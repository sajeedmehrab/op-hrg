import argparse
import glob
import json
import os

import numpy as np
from datasets import Dataset, DatasetDict, Features, Image, Value, load_from_disk
from PIL import Image as PILImage
from tqdm import tqdm

from box_utils import hungarian_mean_iou, scale_box, scale_point

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", required=True)
parser.add_argument("--sam3_boxes_dir", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

IMAGE_RESIZE = 1024
PREDICTION_GRID = 1000
SOURCE_GRID = 840

dataset = load_from_disk(args.dataset_dir)["train"]

sam3_boxes = {}
for path in sorted(glob.glob(os.path.join(args.sam3_boxes_dir, "sam3_bboxes_*.json"))):
    for item in json.load(open(path)):
        sam3_boxes[item["image_name"]] = None if item["pred_bboxes"] == [0, 0, 0, 0] else item["pred_bboxes"]

factor = PREDICTION_GRID / SOURCE_GRID
columns = {k: [] for k in ["id", "problem", "solution", "image", "img_height", "img_width", "object_part", "object_hint_boxes", "baseline_iou"]}
for item in tqdm(dataset):
    solution = json.loads(item["solution"])
    gt_boxes = [s["bbox_2d"] for s in solution]
    pred = sam3_boxes[item["id"]]
    baseline_iou = 0.0 if pred is None else hungarian_mean_iou(np.array(pred), np.array(gt_boxes))

    columns["id"].append(item["id"])
    columns["problem"].append(item["problem"])
    columns["solution"].append(json.dumps([
        {"bbox_2d": scale_box(s["bbox_2d"], factor, factor), "point_2d": scale_point(s["point_2d"], factor, factor)}
        for s in solution
    ]))
    columns["image"].append(item["image"].resize((IMAGE_RESIZE, IMAGE_RESIZE), PILImage.Resampling.LANCZOS))
    columns["img_height"].append(item["img_height"])
    columns["img_width"].append(item["img_width"])
    columns["object_part"].append(False)
    columns["object_hint_boxes"].append(None)
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
