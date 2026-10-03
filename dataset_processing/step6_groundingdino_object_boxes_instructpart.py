import argparse
import json
import os

import torch
from groundingdino.util.inference import load_image, load_model, predict
from torchvision.ops import box_convert
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--instructpart_train_dir", required=True)
parser.add_argument("--order_file", required=True)
parser.add_argument("--groundingdino_config", required=True)
parser.add_argument("--groundingdino_weights", required=True)
parser.add_argument("--output_json", required=True)
args = parser.parse_args()

images_dir = os.path.join(args.instructpart_train_dir, "images")
image_names = [line.strip() for line in open(args.order_file) if line.strip()]

model = load_model(args.groundingdino_config, args.groundingdino_weights)
model.to(device="cuda")

results = []
for image_name in tqdm(image_names):
    object_name = os.path.splitext(image_name)[0].split("-")[-2]
    image_source, image = load_image(os.path.join(images_dir, image_name))
    boxes, logits, phrases = predict(model=model, image=image, caption=object_name, box_threshold=0.35, text_threshold=0.25)
    h, w, _ = image_source.shape
    xyxy = box_convert(boxes=boxes * torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy").numpy().tolist()
    confs = logits.tolist()
    if len(xyxy) == 0:
        xyxy, confs, phrases = [[0, 0, 0, 0]], [0.0], ["no_box_found"]
    results.append({
        "image_filename": image_name,
        "object_name": object_name,
        "boxes": xyxy,
        "conf_scores": confs,
        "pred_phrases": phrases,
    })

with open(args.output_json, "w") as f:
    json.dump(results, f)
