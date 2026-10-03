import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from box_utils import get_bboxes_from_mask


def box_center_points(bboxes):
    return [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in bboxes]


def deepest_interior_points(mask):
    labeled_mask, num_segments = ndimage.label(mask > 0)
    points = []
    for segment_id in range(1, num_segments + 1):
        segment = np.pad(labeled_mask == segment_id, 1)
        distance = ndimage.distance_transform_edt(segment)
        y, x = np.unravel_index(distance.argmax(), distance.shape)
        points.append([int(x) - 1, int(y) - 1])
    return points


parser = argparse.ArgumentParser()
parser.add_argument("--instructpart_train_dir", required=True)
parser.add_argument("--order_file", required=True)
parser.add_argument("--output_json", required=True)
parser.add_argument("--point_type", choices=["deepest_interior", "box_center"], default="deepest_interior")
args = parser.parse_args()

images_dir = os.path.join(args.instructpart_train_dir, "images")
masks_dir = os.path.join(args.instructpart_train_dir, "masks")
image_names = [line.strip() for line in open(args.order_file) if line.strip()]

data = []
for image_name in tqdm(image_names):
    width, height = Image.open(os.path.join(images_dir, image_name)).size
    basename = os.path.splitext(image_name)[0]
    mask = np.array(Image.open(os.path.join(masks_dir, f"{basename}.png")))
    name_parts = basename.split("-")
    bboxes = get_bboxes_from_mask(mask)
    if args.point_type == "deepest_interior":
        points = deepest_interior_points(mask)
    else:
        points = box_center_points(bboxes)
    data.append({
        "image_name": image_name,
        "image_id": name_parts[0],
        "object_name": name_parts[-2],
        "part_name": name_parts[-1],
        "bboxes": bboxes,
        "midpoints": points,
        "image_width": width,
        "image_height": height,
    })

with open(args.output_json, "w") as f:
    json.dump(data, f, indent=2)
