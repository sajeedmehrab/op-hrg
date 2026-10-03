import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from eval_base import compute_first_answer_iou, compute_iou, extract_information_ophrg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--benchmark_root', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--prompt_path', type=str, default=os.path.join(SCRIPT_DIR, "..", "training", "prompts", "ophrg_prompt.txt"))
parser.add_argument('--segmentation_model_path', type=str, default="facebook/sam2-hiera-large")
parser.add_argument('--chunk_id', type=int, default=None)
parser.add_argument('--num_chunks', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_samples', type=int, default=None)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
max_response_length = 2048
resize_size = 1024
prediction_grid_size = 1000

images_dir = os.path.join(args.benchmark_root, "InstructPart", "test", "images")
masks_dir = os.path.join(args.benchmark_root, "InstructPart", "test", "masks")
filenames = sorted(os.listdir(images_dir))
if args.max_samples is not None:
    filenames = filenames[: args.max_samples]

if args.chunk_id is not None:
    chunk_size = len(filenames) // args.num_chunks
    start_idx = args.chunk_id * chunk_size
    end_idx = len(filenames) if args.chunk_id == args.num_chunks - 1 else (args.chunk_id + 1) * chunk_size
    filenames = filenames[start_idx:end_idx]
    results_filepath = os.path.join(args.save_dir, f"parts_results_{args.chunk_id}.json")
else:
    results_filepath = os.path.join(args.save_dir, "parts_results.json")

with open(args.prompt_path, 'r') as f:
    QUESTION_TEMPLATE = f.read()

model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    local_files_only=True,
)
model.eval()
segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)
processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")


def save_results(new_results):
    existing = []
    if os.path.exists(results_filepath):
        with open(results_filepath, 'r') as f:
            existing = json.load(f)
    existing.extend(new_results)
    with open(results_filepath, 'w') as f:
        json.dump(existing, f)


all_results = []
for batch_start in tqdm(range(0, len(filenames), args.batch_size)):
    batch_filenames = filenames[batch_start: batch_start + args.batch_size]
    batch_messages = []
    batch_metadata = []
    for image_name in batch_filenames:
        image = Image.open(os.path.join(images_dir, image_name)).convert("RGB")
        original_width, original_height = image.size
        basename = os.path.splitext(image_name)[0]
        gt_mask = np.array(Image.open(os.path.join(masks_dir, f"{basename}.png")))
        name_parts = basename.split('-')
        object_name, part_name = name_parts[-2], name_parts[-1]
        query_text = f"{object_name}'s {part_name}"
        batch_messages.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": image.resize((resize_size, resize_size), Image.BILINEAR)},
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=query_text.lower().strip("."))},
            ],
        }])
        batch_metadata.append({
            "image_name": image_name,
            "image": image,
            "original_width": original_width,
            "original_height": original_height,
            "x_factor": original_width / prediction_grid_size,
            "y_factor": original_height / prediction_grid_size,
            "gt_mask": gt_mask,
            "object_name": object_name,
            "part_name": part_name,
            "query_text": query_text,
        })

    inputs = processor.apply_chat_template(
        batch_messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=max_response_length, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    for output_text, metadata in zip(output_texts, batch_metadata):
        try:
            bboxes, points, parsed_output = extract_information_ophrg(output_text, metadata['x_factor'], metadata['y_factor'])
            segmentation_model.set_image(metadata['image'])
            mask_all = np.zeros((metadata['original_height'], metadata['original_width']), dtype=bool)
            for bbox, point in zip(bboxes, points):
                masks, scores, _ = segmentation_model.predict(point_coords=[point], point_labels=[1], box=bbox)
                mask_all = np.logical_or(mask_all, masks[np.argsort(scores)[::-1]][0].astype(bool))
            intersection, union = compute_iou(mask_all, metadata['gt_mask'])
            first_intersection, first_union, first_answer_iou, first_bboxes, first_points = compute_first_answer_iou(
                segmentation_model, parsed_output, metadata['x_factor'], metadata['y_factor'], metadata['gt_mask']
            )
            all_results.append({
                "image_id": metadata['image_name'],
                "object_name": metadata['object_name'],
                "part_name": metadata['part_name'],
                "query": metadata['query_text'],
                "think": parsed_output,
                "raw_output": output_text,
                "intersection": int(intersection),
                "union": int(union),
                "iou": float(intersection / union) if union > 0 else 0.0,
                "bboxes": bboxes,
                "points": points,
                "first_intersection": first_intersection,
                "first_union": first_union,
                "first_answer_iou": first_answer_iou,
                "first_bboxes": first_bboxes,
                "first_points": first_points,
            })
        except Exception as e:
            print(f"Error processing {metadata['image_name']}: {e}")
            all_results.append({
                "image_id": metadata['image_name'],
                "object_name": metadata['object_name'],
                "part_name": metadata['part_name'],
                "query": metadata['query_text'],
                "error": str(e),
                "raw_output": output_text,
                "intersection": 0,
                "union": int(metadata['gt_mask'].sum()),
                "iou": 0.0,
            })
        if len(all_results) % 100 == 0:
            save_results(all_results)
            all_results = []

    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

save_results(all_results)
print(f"results saved to {args.save_dir}")
