import os

import numpy as np
from PIL import Image

from pascalpart import get_pascalpart_masks
from partimagenet_dataset import PartImageNetDataset

DATASET_CHOICES = ("instructpart", "pascalpart", "partimagenet")
PASCALPART_VAL_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pascalpart_val.txt")


def benchmark_paths(benchmark_root):
    return {
        "instructpart_test_dir": os.path.join(benchmark_root, "InstructPart", "test"),
        "pascal_image_dir": os.path.join(benchmark_root, "Pascal_VOC_2012", "VOCdevkit", "VOC2012", "JPEGImages"),
        "pascal_annot_dir": os.path.join(benchmark_root, "PascalPart", "Annotations_Part"),
        "partimagenet_images_dir": os.path.join(benchmark_root, "PartImageNet", "test"),
        "partimagenet_json": os.path.join(benchmark_root, "PartImageNet", "test.json"),
    }


def read_txt_file(file_path):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    while lines[-1] == "":
        lines = lines[:-1]
    return lines


def as_bool_mask(mask):
    return np.asarray(mask) > 0


def mask_area(mask):
    return int(as_bool_mask(mask).sum())


def compute_iou(pred_mask, gt_mask):
    pred = as_bool_mask(pred_mask)
    gt = as_bool_mask(gt_mask)
    return int(np.logical_and(pred, gt).sum()), int(np.logical_or(pred, gt).sum())


def combine_masks(masks):
    out = as_bool_mask(masks[0])
    for m in masks[1:]:
        out = np.logical_or(out, as_bool_mask(m))
    return out


def chunk_units(units, chunk_id, num_chunks):
    if chunk_id is None:
        return units, 0
    size = len(units) // num_chunks
    start = chunk_id * size
    end = len(units) if chunk_id == num_chunks - 1 else (chunk_id + 1) * size
    return units[start:end], start


def _instructpart_units(paths):
    return sorted(os.listdir(os.path.join(paths["instructpart_test_dir"], "images")))


def _iter_instructpart(paths, units, offset):
    images_dir = os.path.join(paths["instructpart_test_dir"], "images")
    masks_dir = os.path.join(paths["instructpart_test_dir"], "masks")
    for i, image_name in enumerate(units):
        basename = os.path.splitext(image_name)[0]
        image = Image.open(os.path.join(images_dir, image_name)).convert("RGB")
        gt_mask = as_bool_mask(Image.open(os.path.join(masks_dir, f"{basename}.png")))
        name_parts = basename.split("-")
        object_name, part_name = name_parts[-2], name_parts[-1]
        yield {
            "image_id": image_name,
            "query": f"{object_name}'s {part_name}",
            "image": image,
            "gt_mask": gt_mask,
            "result_type": "parts",
            "unit_index": offset + i,
            "object_name": object_name,
            "part_name": part_name,
        }


def _pascalpart_units(paths):
    return sorted(read_txt_file(PASCALPART_VAL_TXT))


def _iter_pascalpart(paths, units, offset):
    for i, filename in enumerate(units):
        anno_dict = get_pascalpart_masks(filename + ".mat", paths["pascal_annot_dir"], images_path=paths["pascal_image_dir"])
        if len(anno_dict) == 0:
            continue
        image = Image.open(os.path.join(paths["pascal_image_dir"], filename + ".jpg")).convert("RGB")
        for obj_name, anno in anno_dict.items():
            yield {
                "image_id": filename,
                "query": obj_name,
                "image": image,
                "gt_mask": combine_masks(anno["object_maps"]),
                "result_type": "objects",
                "unit_index": offset + i,
                "object_name": obj_name,
                "part_name": None,
            }
            for part_name, part_masks in anno["parts"].items():
                yield {
                    "image_id": filename,
                    "query": f"{obj_name}'s {part_name}",
                    "image": image,
                    "gt_mask": combine_masks(part_masks),
                    "result_type": "parts",
                    "unit_index": offset + i,
                    "object_name": obj_name,
                    "part_name": part_name,
                }


_PARTIMAGENET_CACHE = {}


def _partimagenet_dataset(paths):
    key = paths["partimagenet_json"]
    if key not in _PARTIMAGENET_CACHE:
        _PARTIMAGENET_CACHE[key] = PartImageNetDataset(
            json_path=paths["partimagenet_json"], images_dir=paths["partimagenet_images_dir"]
        )
    return _PARTIMAGENET_CACHE[key]


def _partimagenet_units(paths):
    return list(range(_partimagenet_dataset(paths).num_annotations))


def _iter_partimagenet(paths, units, offset):
    ds = _partimagenet_dataset(paths)
    for i, ann_idx in enumerate(units):
        try:
            ann = ds.get_annotation(ann_idx)
        except FileNotFoundError as e:
            print(f"[partimagenet] skipping annotation {ann_idx}: {e}", flush=True)
            continue
        if ann["class_name"] is None or ann["mask"] is None:
            continue
        yield {
            "image_id": ann["image_filename"],
            "query": ann["class_name"].lower().strip("."),
            "image": ann["image"],
            "gt_mask": as_bool_mask(ann["mask"]),
            "result_type": "parts",
            "unit_index": offset + i,
            "annotation_idx": ann_idx,
            "annotation_id": int(ann["id"]) if not isinstance(ann["id"], (list, tuple)) else None,
        }


_UNIT_FNS = {
    "instructpart": _instructpart_units,
    "pascalpart": _pascalpart_units,
    "partimagenet": _partimagenet_units,
}

_ITER_FNS = {
    "instructpart": _iter_instructpart,
    "pascalpart": _iter_pascalpart,
    "partimagenet": _iter_partimagenet,
}


def iter_records(dataset, benchmark_root, chunk_id=None, num_chunks=4, max_units=None):
    paths = benchmark_paths(benchmark_root)
    units = _UNIT_FNS[dataset](paths)
    if max_units is not None:
        units = units[:max_units]
    units, offset = chunk_units(units, chunk_id, num_chunks)
    print(f"[{dataset}] chunk={chunk_id} units={len(units)} (offset {offset})", flush=True)
    return _ITER_FNS[dataset](paths, units, offset)


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def result_paths(save_dir, chunk_id):
    suffix = "" if chunk_id is None else f"_{chunk_id}"
    return {
        "parts": os.path.join(save_dir, f"parts_results{suffix}.json"),
        "objects": os.path.join(save_dir, f"objects_results{suffix}.json"),
    }
