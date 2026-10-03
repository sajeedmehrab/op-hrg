import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

from box_utils import sam3_text_boxes

parser = argparse.ArgumentParser()
parser.add_argument("--instructpart_train_dir", required=True)
parser.add_argument("--order_file", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--chunk_id", type=int, required=True)
parser.add_argument("--total_chunks", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
images_dir = os.path.join(args.instructpart_train_dir, "images")
image_names = [line.strip() for line in open(args.order_file) if line.strip()]

chunk_size = len(image_names) // args.total_chunks
start = args.chunk_id * chunk_size
end = len(image_names) if args.chunk_id == args.total_chunks - 1 else start + chunk_size
image_names = image_names[start:end]

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

results = []
for b in tqdm(range(0, len(image_names), args.batch_size)):
    batch = image_names[b: b + args.batch_size]
    images = [Image.open(os.path.join(images_dir, name)).convert("RGB") for name in batch]
    texts = []
    for name in batch:
        name_parts = os.path.splitext(name)[0].split("-")
        texts.append(f"{name_parts[-2]}'s {name_parts[-1]}")
    boxes = sam3_text_boxes(model, processor, images, texts, device)
    for name, text, box in zip(batch, texts, boxes):
        results.append({"image_name": name, "text_prompt": text, "pred_bboxes": box})

os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, f"sam3_bboxes_{args.chunk_id}.json"), "w") as f:
    json.dump(results, f)
