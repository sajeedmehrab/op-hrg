import argparse
import glob
import json
import os

import cv2
import numpy as np
from datasets import Dataset, DatasetDict, Features, Image, Value
from tqdm import tqdm

from box_utils import hungarian_mean_iou, scale_box, scale_point

parser = argparse.ArgumentParser()
parser.add_argument("--instructpart_train_dir", required=True)
parser.add_argument("--metadata_json", required=True)
parser.add_argument("--sam3_boxes_dir", required=True)
parser.add_argument("--groundingdino_json", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

IMAGE_RESIZE = 1024
PREDICTION_GRID = 1000

images_dir = os.path.join(args.instructpart_train_dir, "images")
data = json.load(open(args.metadata_json))

object_boxes = {}
for item in json.load(open(args.groundingdino_json)):
    object_boxes[item["image_filename"]] = None if item["boxes"] == [[0, 0, 0, 0]] else item["boxes"]

sam3_boxes = {}
for path in sorted(glob.glob(os.path.join(args.sam3_boxes_dir, "sam3_bboxes_*.json"))):
    for item in json.load(open(path)):
        sam3_boxes[item["image_name"]] = None if item["pred_bboxes"] == [0, 0, 0, 0] else item["pred_bboxes"]

columns = {k: [] for k in ["id", "problem", "solution", "image", "img_height", "img_width", "object_part", "object_hint_boxes", "baseline_iou"]}
for item in tqdm(data):
    name = item["image_name"]
    pred = sam3_boxes[name]
    baseline_iou = 0.0 if pred is None else hungarian_mean_iou(np.array(pred), np.array(item["bboxes"]))

    image = cv2.imread(os.path.join(images_dir, name))
    height, width = image.shape[:2]
    x_factor, y_factor = PREDICTION_GRID / width, PREDICTION_GRID / height

    if item["object_name"].strip().endswith("s"):
        problem = f"{item['object_name']}' {item['part_name']}"
    else:
        problem = f"{item['object_name']}'s {item['part_name']}"

    hint = object_boxes[name]

    columns["id"].append(item["image_id"])
    columns["problem"].append(problem)
    columns["solution"].append(json.dumps([
        {"bbox_2d": scale_box(box, x_factor, y_factor), "point_2d": scale_point(point, x_factor, y_factor)}
        for box, point in zip(item["bboxes"], item["midpoints"])
    ]))
    columns["image"].append(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (IMAGE_RESIZE, IMAGE_RESIZE), interpolation=cv2.INTER_AREA))
    columns["img_height"].append(height)
    columns["img_width"].append(width)
    columns["object_part"].append(True)
    columns["object_hint_boxes"].append(None if hint is None else json.dumps([scale_box(b, x_factor, y_factor) for b in hint]))
    columns["baseline_iou"].append(baseline_iou)

features = Features({
    "id": Value("string"),
    "problem": Value("string"),
    "solution": Value("string"),
    "image": Image(),
    "img_height": Value("int64"),
    "img_width": Value("int64"),
    "object_part": Value("bool"),
    "object_hint_boxes": Value("string"),
    "baseline_iou": Value("float32"),
})
DatasetDict({"train": Dataset.from_dict(columns, features=features)}).save_to_disk(args.output_dir)
