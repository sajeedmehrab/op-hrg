import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks"))
from benchmark_datasets import DATASET_CHOICES, batched, compute_iou, iter_records, mask_area, result_paths

QUESTION_TEMPLATE = (
    'Please find "{Question}" with bboxs and points.'
    "Compare the difference between object(s) and find the most closely matched object(s)."
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    "Output the bbox(es) and point(s) inside the interested object(s) in JSON format."
    "i.e., <think> thinking process here </think>"
    "<answer>{Answer}</answer>"
)

ANSWER_EXEMPLAR = (
    '[{"bbox_2d": [10,100,200,210], "point_2d": [30,110]}, '
    '{"bbox_2d": [225,296,706,786], "point_2d": [302,410]}]'
)


def parse_answer(output_text, x_factor, y_factor):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", output_text, re.DOTALL)
    if not match:
        raise ValueError("no <answer> block in model output")
    data = json.loads(match.group(1))
    bboxes = [
        [
            int(item["bbox_2d"][0] * x_factor + 0.5),
            int(item["bbox_2d"][1] * y_factor + 0.5),
            int(item["bbox_2d"][2] * x_factor + 0.5),
            int(item["bbox_2d"][3] * y_factor + 0.5),
        ]
        for item in data
    ]
    points = [[int(item["point_2d"][0] * x_factor + 0.5), int(item["point_2d"][1] * y_factor + 0.5)] for item in data]
    think_match = re.search(r"<think>([^<]+)</think>", output_text)
    think = think_match.group(1).strip() if think_match else ""
    return bboxes, points, think


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p.add_argument("--benchmark_root", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--segmentation_model_path", default="facebook/sam2-hiera-large")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--chunk_id", type=int, default=None)
    p.add_argument("--num_chunks", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--resize_size", type=int, default=840)
    p.add_argument("--max_new_tokens", type=int, default=2000)
    p.add_argument("--max_units", type=int, default=None)
    p.add_argument("--save_every", type=int, default=200)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    paths = result_paths(args.save_dir, args.chunk_id)
    rows = {"parts": [], "objects": []}

    def flush():
        for kind, path in paths.items():
            if rows[kind]:
                with open(path, "w") as f:
                    json.dump(rows[kind], f)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")
    segmenter = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)

    records = iter_records(args.dataset, args.benchmark_root, chunk_id=args.chunk_id, num_chunks=args.num_chunks, max_units=args.max_units)

    n_written = 0
    n_failed = 0
    current_image_key = None
    t_start = time.time()
    for batch in batched(records, args.batch_size):
        messages = []
        for rec in batch:
            resized = rec["image"].resize((args.resize_size, args.resize_size), Image.BILINEAR)
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": resized},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=rec["query"].lower().strip("."), Answer=ANSWER_EXEMPLAR)},
                ],
            }])
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            generated = model.generate(**inputs, use_cache=True, max_new_tokens=args.max_new_tokens, do_sample=False)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            output_texts = processor.batch_decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        for rec, output_text, gen_ids in zip(batch, output_texts, trimmed):
            w, h = rec["image"].size
            x_factor, y_factor = w / args.resize_size, h / args.resize_size
            row = {
                "image_id": rec["image_id"],
                "query": rec["query"],
                "object_name": rec.get("object_name"),
                "part_name": rec.get("part_name"),
                "n_generated_tokens": int(len(gen_ids)),
                "raw_output": output_text,
            }
            if args.dataset == "partimagenet":
                row["annotation_idx"] = rec.get("annotation_idx")
            try:
                bboxes, points, think = parse_answer(output_text, x_factor, y_factor)
                image_key = (rec["image_id"], w, h)
                if image_key != current_image_key:
                    segmenter.set_image(rec["image"])
                    current_image_key = image_key
                mask_all = np.zeros((h, w), dtype=bool)
                for bbox, point in zip(bboxes, points):
                    masks, scores, _ = segmenter.predict(point_coords=[point], point_labels=[1], box=bbox)
                    mask_all = np.logical_or(mask_all, masks[np.argsort(scores)[::-1]][0].astype(bool))
                intersection, union = compute_iou(mask_all, rec["gt_mask"])
                row.update({
                    "think": think,
                    "n_predicted_bboxes": len(bboxes),
                    "intersection": intersection,
                    "union": union,
                    "iou": (intersection / union) if union > 0 else 0.0,
                })
            except Exception as e:
                n_failed += 1
                current_image_key = None
                row.update({
                    "think": "",
                    "n_predicted_bboxes": 0,
                    "error": f"{type(e).__name__}: {e}",
                    "intersection": 0,
                    "union": mask_area(rec["gt_mask"]),
                    "iou": 0.0,
                })
            rows[rec["result_type"]].append(row)
            n_written += 1

        if n_written % args.save_every < args.batch_size:
            flush()
            print(f"[{args.dataset} chunk {args.chunk_id}] {n_written} rows ({n_failed} failed) {(time.time()-t_start)/60:.1f} min", flush=True)
        del inputs, generated, trimmed
        torch.cuda.empty_cache()

    flush()
    print(f"[{args.dataset} chunk {args.chunk_id}] done: {n_written} rows, {n_failed} failed", flush=True)


if __name__ == "__main__":
    main()
