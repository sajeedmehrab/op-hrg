import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image as PILImage
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from eval_base import combine_masks, compute_first_answer_iou, compute_iou, extract_information_ophrg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "benchmarks"))
from benchmark_datasets import PASCALPART_VAL_TXT, read_txt_file
from pascalpart import get_pascalpart_masks

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
parser.add_argument('--save_every', type=int, default=100)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
max_response_length = 2048
resize_size = 1024
prediction_grid_size = 1000

pascal_image_dir = os.path.join(args.benchmark_root, "Pascal_VOC_2012", "VOCdevkit", "VOC2012", "JPEGImages")
annotations_path = os.path.join(args.benchmark_root, "PascalPart", "Annotations_Part")

filenames = read_txt_file(PASCALPART_VAL_TXT)
if args.max_samples is not None:
    filenames = filenames[: args.max_samples]

if args.chunk_id is not None:
    chunk_size = len(filenames) // args.num_chunks
    start_idx = args.chunk_id * chunk_size
    end_idx = len(filenames) if args.chunk_id == args.num_chunks - 1 else (args.chunk_id + 1) * chunk_size
    filenames = filenames[start_idx:end_idx]
    object_results_filepath = os.path.join(args.save_dir, f"objects_results_{args.chunk_id}.json")
    parts_results_filepath = os.path.join(args.save_dir, f"parts_results_{args.chunk_id}.json")
else:
    object_results_filepath = os.path.join(args.save_dir, "objects_results.json")
    parts_results_filepath = os.path.join(args.save_dir, "parts_results.json")

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

all_object_outputs = []
all_parts_outputs = []


def flush_results():
    global all_object_outputs, all_parts_outputs
    for filepath, buffer in ((object_results_filepath, all_object_outputs), (parts_results_filepath, all_parts_outputs)):
        existing = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing = json.load(f)
        existing.extend(buffer)
        with open(filepath, 'w') as f:
            json.dump(existing, f)
    all_object_outputs = []
    all_parts_outputs = []


work_items = []
for filename in filenames:
    anno_dict = get_pascalpart_masks(filename + '.mat', annotations_path, images_path=pascal_image_dir)
    queries = []
    for obj_name, anno in anno_dict.items():
        queries.append((obj_name, np.asarray(combine_masks(anno['object_maps'])), True))
        for part_name, masks in anno['parts'].items():
            queries.append((obj_name + "'s " + part_name, np.asarray(combine_masks(masks)), False))
    for q_i, (query, gt_mask, is_object) in enumerate(queries):
        original_height, original_width = gt_mask.shape
        work_items.append({
            "filename": filename,
            "query": query,
            "gt_mask": gt_mask,
            "is_object": is_object,
            "x_factor": original_width / prediction_grid_size,
            "y_factor": original_height / prediction_grid_size,
            "original_width": original_width,
            "original_height": original_height,
            "is_last_in_image": q_i == len(queries) - 1,
        })

image_cache = {}


def get_image(filename):
    if filename not in image_cache:
        img = PILImage.open(os.path.join(pascal_image_dir, filename + '.jpg')).convert("RGB")
        image_cache[filename] = {"orig": img, "resized": img.resize((resize_size, resize_size), PILImage.BILINEAR)}
    return image_cache[filename]


images_done = 0
last_set_image = None
for batch_start in tqdm(range(0, len(work_items), args.batch_size)):
    batch = work_items[batch_start: batch_start + args.batch_size]
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": get_image(item["filename"])["resized"]},
            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=item["query"].lower().strip("."))},
        ],
    }] for item in batch]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=max_response_length, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        batch_output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for item, output_text in zip(batch, batch_output_text):
            filename = item["filename"]
            gt_mask = item["gt_mask"]
            out_list = all_object_outputs if item["is_object"] else all_parts_outputs
            row = None
            try:
                bboxes, points, think = extract_information_ophrg(output_text, item["x_factor"], item["y_factor"])
            except Exception as e:
                print("Reasoning error: ", e, filename, item["query"])
                row = {
                    "image_id": filename,
                    "query": item["query"],
                    "error": str(e),
                    "raw_output": output_text,
                    "intersection": 0,
                    "union": int(gt_mask.sum()),
                    "iou": 0.0,
                }
            else:
                try:
                    if last_set_image != filename:
                        segmentation_model.set_image(get_image(filename)["orig"])
                        last_set_image = filename
                    mask_all = np.zeros((item["original_height"], item["original_width"]), dtype=bool)
                    for bbox, point in zip(bboxes, points):
                        masks, scores, _ = segmentation_model.predict(point_coords=[point], point_labels=[1], box=bbox)
                        mask_all = np.logical_or(mask_all, masks[np.argsort(scores)[::-1]][0].astype(bool))
                    intersection, union = compute_iou(mask_all, gt_mask)
                except Exception as e:
                    print("Segmentation/IoU error: ", e, filename)
                    row = {
                        "image_id": filename,
                        "query": item["query"],
                        "error": str(e),
                        "raw_output": output_text,
                        "intersection": 0,
                        "union": int(gt_mask.sum()),
                        "iou": 0.0,
                    }
                else:
                    first_intersection, first_union, first_answer_iou, first_bboxes, first_points = compute_first_answer_iou(
                        segmentation_model, think, item["x_factor"], item["y_factor"], gt_mask
                    )
                    row = {
                        "image_id": filename,
                        "query": item["query"],
                        "think": think,
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
                    }
            if row is not None:
                out_list.append(row)
            if item["is_last_in_image"]:
                images_done += 1
                image_cache.pop(filename, None)
                if last_set_image == filename:
                    last_set_image = None
                if images_done % args.save_every == 0:
                    flush_results()

    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

flush_results()
print(f"results saved to {args.save_dir}")
