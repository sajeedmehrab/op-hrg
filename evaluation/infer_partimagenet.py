import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from eval_base import compute_first_answer_iou, compute_iou, extract_information_ophrg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "benchmarks"))
from partimagenet_dataset import PartImageNetDataset

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

dataset = PartImageNetDataset(
    json_path=os.path.join(args.benchmark_root, "PartImageNet", "test.json"),
    images_dir=os.path.join(args.benchmark_root, "PartImageNet", "test"),
)
num_annotations = dataset.num_annotations
if args.max_samples is not None:
    num_annotations = min(num_annotations, args.max_samples)

if args.chunk_id is not None:
    chunk_size = num_annotations // args.num_chunks
    start_idx = args.chunk_id * chunk_size
    end_idx = num_annotations if args.chunk_id == args.num_chunks - 1 else (args.chunk_id + 1) * chunk_size
    results_filepath = os.path.join(args.save_dir, f"parts_results_{args.chunk_id}.json")
else:
    start_idx, end_idx = 0, num_annotations
    results_filepath = os.path.join(args.save_dir, "parts_results.json")
if os.path.exists(results_filepath):
    raise ValueError(f"{results_filepath} already exists. Choose a different save directory.")
annotation_ids = list(range(start_idx, end_idx))

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
for j in tqdm(range(0, len(annotation_ids), args.batch_size)):
    batch_messages = []
    batch_metadata = []
    for i in annotation_ids[j: j + args.batch_size]:
        try:
            annotation = dataset.get_annotation(i)
        except Exception as e:
            print(f"Error loading annotation {i}: {e}")
            continue
        gt_mask = annotation['mask']
        query_text = annotation['class_name']
        if gt_mask is None or query_text is None:
            print(f"Skipping annotation {i}")
            continue
        image = annotation['image']
        original_width, original_height = image.size
        batch_messages.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": image.resize((resize_size, resize_size), Image.BILINEAR)},
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=query_text.lower().strip("."))},
            ],
        }])
        batch_metadata.append({
            "image_name": annotation['image_filename'],
            "image": image,
            "original_width": original_width,
            "original_height": original_height,
            "x_factor": original_width / prediction_grid_size,
            "y_factor": original_height / prediction_grid_size,
            "gt_mask": gt_mask,
            "query_text": query_text,
            "annotation_idx": i,
        })
    if len(batch_messages) == 0:
        continue

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
                "annotation_idx": metadata['annotation_idx'],
            })
        except Exception as e:
            print(f"Error processing {metadata['image_name']}: {e}")
            all_results.append({
                "image_id": metadata['image_name'],
                "query": metadata['query_text'],
                "error": str(e),
                "raw_output": output_text,
                "intersection": 0,
                "union": int(metadata['gt_mask'].sum()),
                "iou": 0.0,
                "annotation_idx": metadata['annotation_idx'],
            })
        if len(all_results) % 100 == 0:
            save_results(all_results)
            all_results = []

    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

save_results(all_results)
print(f"results saved to {args.save_dir}")
