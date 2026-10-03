import json
import re

import numpy as np


def extract_information_ophrg(output_text, x_factor, y_factor):
    locate_match = re.search(r'<locate>([^<]+)</locate>', output_text)
    decide_match = re.search(r'<target>([^<]+)</target>', output_text)
    first_answer_match = re.search(r'<first_answer>\s*(.*?)\s*</first_answer>', output_text, re.DOTALL)
    criticism_match = re.search(r'<criticism>([^<]+)</criticism>', output_text)
    final_answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)

    output_text_parsed = {
        "locate": locate_match.group(1).strip() if locate_match else "",
        "decide": decide_match.group(1).strip() if decide_match else "",
        "first_answer": first_answer_match.group(1).strip() if first_answer_match else "",
        "criticism": criticism_match.group(1).strip() if criticism_match else "",
        "final_answer": final_answer_match.group(1).strip() if final_answer_match else "",
    }

    pred_bboxes = []
    pred_points = []
    if final_answer_match:
        data = json.loads(final_answer_match.group(1))
        pred_bboxes = [[
            int(item['bbox_2d'][0] * x_factor + 0.5),
            int(item['bbox_2d'][1] * y_factor + 0.5),
            int(item['bbox_2d'][2] * x_factor + 0.5),
            int(item['bbox_2d'][3] * y_factor + 0.5)
        ] for item in data]
        pred_points = [[
            int(item['point_2d'][0] * x_factor + 0.5),
            int(item['point_2d'][1] * y_factor + 0.5)
        ] for item in data]

    return pred_bboxes, pred_points, output_text_parsed


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0, 0
    return intersection, union


def parse_first_answer_to_bboxes_points(parsed_output, x_factor, y_factor):
    if not isinstance(parsed_output, dict):
        return [], []
    fa = parsed_output.get('first_answer', '') or ''
    if not fa.strip():
        return [], []
    try:
        data = json.loads(fa)
        bboxes = [[
            int(item['bbox_2d'][0] * x_factor + 0.5),
            int(item['bbox_2d'][1] * y_factor + 0.5),
            int(item['bbox_2d'][2] * x_factor + 0.5),
            int(item['bbox_2d'][3] * y_factor + 0.5)
        ] for item in data]
        points = [[
            int(item['point_2d'][0] * x_factor + 0.5),
            int(item['point_2d'][1] * y_factor + 0.5)
        ] for item in data]
    except Exception:
        return [], []
    return bboxes, points


def compute_first_answer_iou(segmentation_model, parsed_output, x_factor, y_factor, gt_mask):
    first_bboxes, first_points = parse_first_answer_to_bboxes_points(parsed_output, x_factor, y_factor)
    gt = np.asarray(gt_mask)
    first_mask = np.zeros(gt.shape[:2], dtype=bool)
    for bbox, point in zip(first_bboxes, first_points):
        try:
            masks, scores, _ = segmentation_model.predict(point_coords=[point], point_labels=[1], box=bbox)
            first_mask = np.logical_or(first_mask, masks[np.argsort(scores)[::-1]][0].astype(bool))
        except Exception as exc:
            print(f"first-answer SAM2 predict error: {exc}")
            continue
    first_intersection, first_union = compute_iou(first_mask, gt)
    first_iou = float(first_intersection / first_union) if first_union > 0 else 0.0
    return int(first_intersection), int(first_union), first_iou, first_bboxes, first_points


def combine_masks(masks):
    combined_mask = masks[0]
    for i in range(1, len(masks)):
        combined_mask = combined_mask | masks[i]
    return combined_mask
