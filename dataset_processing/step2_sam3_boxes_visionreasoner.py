import argparse
import json
import os

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

from box_utils import sam3_text_boxes

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--chunk_id", type=int, required=True)
parser.add_argument("--total_chunks", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = load_from_disk(args.dataset_dir)["train"]

chunk_size = len(dataset) // args.total_chunks
start = args.chunk_id * chunk_size
end = len(dataset) if args.chunk_id == args.total_chunks - 1 else start + chunk_size
rows = [dataset[i] for i in range(start, end)]

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

results = []
for b in tqdm(range(0, len(rows), args.batch_size)):
    batch = rows[b: b + args.batch_size]
    texts = [row["problem"] for row in batch]
    boxes = sam3_text_boxes(model, processor, [row["image"] for row in batch], texts, device)
    for row, text, box in zip(batch, texts, boxes):
        results.append({"image_name": row["id"], "text_prompt": text, "pred_bboxes": box})

os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, f"sam3_bboxes_{args.chunk_id}.json"), "w") as f:
    json.dump(results, f)
