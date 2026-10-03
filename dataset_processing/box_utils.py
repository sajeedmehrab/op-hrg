import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


def get_bboxes_from_mask(mask):
    labeled_mask, num_segments = ndimage.label(mask > 0)
    bboxes = []
    for segment_id in range(1, num_segments + 1):
        coords = np.argwhere(labeled_mask == segment_id)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
    return bboxes


def calculate_intersection_area(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_width = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_height = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    return inter_width * inter_height


def remove_overlapping_boxes(xyxy_boxes, intersection_threshold=0.9):
    if len(xyxy_boxes) == 0:
        return []
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in xyxy_boxes]
    sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
    keep_indices = []
    for i in sorted_indices:
        area_i = areas[i]
        if area_i == 0:
            continue
        should_keep = True
        for j in keep_indices:
            if calculate_intersection_area(xyxy_boxes[i], xyxy_boxes[j]) / area_i > intersection_threshold:
                should_keep = False
                break
        if should_keep:
            keep_indices.append(i)
    keep_indices.sort()
    return [xyxy_boxes[i] for i in keep_indices]


def boxes_intersect(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    return (x_min1 < x_max2 and x_max1 > x_min2) and (y_min1 < y_max2 and y_max1 > y_min2)


def merge_intersecting_boxes(bboxes):
    if not bboxes:
        return []
    n = len(bboxes)
    box_to_group = [-1] * n
    groups = {}
    group_id = 0
    for i in range(n):
        if box_to_group[i] == -1:
            groups[group_id] = [i]
            box_to_group[i] = group_id
            changed = True
            while changed:
                changed = False
                for j in range(n):
                    if box_to_group[j] == -1:
                        for box_idx in groups[group_id]:
                            if boxes_intersect(bboxes[j], bboxes[box_idx]):
                                groups[group_id].append(j)
                                box_to_group[j] = group_id
                                changed = True
                                break
            group_id += 1
    merged_boxes = []
    for group_indices in groups.values():
        group_boxes = [bboxes[i] for i in group_indices]
        merged_boxes.append([
            min(box[0] for box in group_boxes),
            min(box[1] for box in group_boxes),
            max(box[2] for box in group_boxes),
            max(box[3] for box in group_boxes),
        ])
    return merged_boxes


def sam3_text_boxes(model, processor, images, texts, device):
    inputs = processor.image_processor(images, return_tensors="pt")
    inputs.update(processor.tokenizer(texts, return_tensors="pt", padding="max_length", max_length=32, truncation=True))
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.5, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
    )
    all_boxes = []
    for res in results:
        if len(res["masks"]) == 0:
            all_boxes.append([0, 0, 0, 0])
            continue
        pred_bboxes = get_bboxes_from_mask(res["masks"].any(dim=0).cpu().numpy())
        if len(pred_bboxes) > 1:
            pred_bboxes = remove_overlapping_boxes(pred_bboxes)
        if len(pred_bboxes) > 1:
            pred_bboxes = merge_intersecting_boxes(pred_bboxes)
        all_boxes.append(pred_bboxes)
    return all_boxes


def batch_iou(boxes1, boxes2):
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)
    unionArea = box1Area + np.transpose(box2Area) - interArea
    return interArea / np.clip(unionArea, a_min=1e-9, a_max=None)


def hungarian_mean_iou(pred_bboxes, gt_bboxes):
    M, N = len(pred_bboxes), len(gt_bboxes)
    if M == 0 or N == 0:
        return 1.0 if (M == 0 and N == 0) else 0.0
    iou_matrix = batch_iou(pred_bboxes, gt_bboxes)
    row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)
    return float(np.mean([iou_matrix[i, j] for i, j in zip(row_ind, col_ind)]))


def scale_box(bbox_2d, x_factor, y_factor):
    return [
        int(bbox_2d[0] * x_factor + 0.5),
        int(bbox_2d[1] * y_factor + 0.5),
        int(bbox_2d[2] * x_factor + 0.5),
        int(bbox_2d[3] * y_factor + 0.5),
    ]


def scale_point(point_2d, x_factor, y_factor):
    return [int(point_2d[0] * x_factor + 0.5), int(point_2d[1] * y_factor + 0.5)]
