import re
import json
import logging
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Any

logger = logging.getLogger(__name__)

def batch_iou(boxes1, boxes2):
    """Compute IoU between each box in boxes1 and each box in boxes2."""
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)  # (M,1) each
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)  # (N,1)
    # Intersection coords
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    # Areas of boxes
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)
    # Union
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / np.clip(unionArea, a_min=1e-9, a_max=None)  # avoid division by zero
    return iou

def batch_l1_distance(boxes1, boxes2):
    """Compute mean L1 distance between each predicted box and each ground truth box."""
    # L1 distance on bounding box coordinates (average absolute difference per coordinate)
    boxes1 = boxes1[:, np.newaxis, :]  # (M,1,4)
    boxes2 = boxes2[np.newaxis, :, :]  # (1,N,4)
    return np.mean(np.abs(boxes1 - boxes2), axis=2)  # (M,N)

def batch_points_distance(points1, points2):
    """Compute Euclidean distance between each predicted point and each ground truth point."""
    points1 = points1[:, np.newaxis, :]  # (M,1,2)
    points2 = points2[np.newaxis, :, :]  # (1,N,2)
    dist = np.sqrt(np.sum((points1 - points2) ** 2, axis=2))
    return dist  # (M,N)

def batch_points_in_box(points, boxes):
    """Check if each point lies inside the corresponding box (by index)."""
    # points: (M,2), boxes: (M,4)
    if len(points) == 0 or len(boxes) == 0:
        return np.array([], dtype=bool)
    try:
        x_check = (points[:, 0] >= boxes[:, 0]) & (points[:, 0] <= boxes[:, 2])
        y_check = (points[:, 1] >= boxes[:, 1]) & (points[:, 1] <= boxes[:, 3])
        return x_check & y_check  # (M,) boolean array
    except Exception as e:
        logger.debug("Error in batch_points_in_box: %s", e)
        return np.array([], dtype=bool)

# Reward components:
def vision_reasoner_format_reward(predict_str: str) -> float:
    """
        Reward for producing the correct output format (reasoning and answer tags with JSON).
        The reward is normalized to be out of 1.0 based on format correctness and content validity

        Args:
            predict_str: Full prediction string
        Returns:
            format_reward: float reward [0,1]
        
    """
    pattern = (
        r"^<locate>.*?</locate>\s*"
        r"<target>(object|part)</target>\s*"
        r"<object_hint>\s*(\[.*?\])\s*</object_hint>\s*"
        r"<first_answer>\s*\[.*?\]\s*</first_answer>\s*"
        r"<criticism>.*?ADJUSTMENT:\s*(YES|NO)\s*</criticism>\s*"
        r"<answer>\s*\[.*?\]\s*</answer>$"
    )
    format_correct = 1.0 if re.fullmatch(pattern, predict_str.strip(), re.DOTALL | re.IGNORECASE) else 0.0
    # Check JSON content format for object_hint, first_answer and answer
    content_reward = 0.0
    try:
        # Extract <target> to check if it's "part"
        target_match = re.search(r'<target>\s*(object|part)\s*</target>', predict_str, re.IGNORECASE)
        is_part = target_match and target_match.group(1).lower() == 'part' if target_match else False
        
        # Extract and validate object_hint
        # We expect object_hint to be a list of objects if target is "part", and empty list if target is "object"
        # Maximum reward for object_hint is 1.0
        hint_match = re.search(r'<object_hint>\s*(\[.*?\])\s*</object_hint>', predict_str, re.DOTALL)
        if hint_match:
            hint_data = json.loads(hint_match.group(1))
            if isinstance(hint_data, list):
                # If target is "part", object_hint should have objects; if "object", should be empty
                if is_part and len(hint_data) > 0:
                    # Validate structure of hint objects
                    per_item_scores = []
                    for item in hint_data:
                        score = 0.0
                        if 'bbox_2d' in item and isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4:
                            score += 1.0
                        per_item_scores.append(score)
                    if per_item_scores:
                        content_reward += (sum(per_item_scores) / len(per_item_scores))
                elif not is_part and len(hint_data) == 0:
                    content_reward += 1.0  # Correct: empty list for object target
                elif is_part and len(hint_data) == 0:
                    content_reward += 0.0  # Incorrect: should have hints for parts
                elif not is_part and len(hint_data) > 0:
                    content_reward += 0.0  # Incorrect: shouldn't have hints for objects

        # Extract and validate first_answer
        # Maximum reward for first_answer is 1.0
        first_match = re.search(r'<first_answer>\s*(\[.*?\])\s*</first_answer>', predict_str, re.DOTALL)
        if first_match:
            first_data = json.loads(first_match.group(1))
            if isinstance(first_data, list) and len(first_data) > 0:
                per_item_scores = []
                for item in first_data:
                    score = 0.0
                    if 'bbox_2d' in item and isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4:
                        score += 0.5
                    if 'point_2d' in item and isinstance(item['point_2d'], list) and len(item['point_2d']) == 2:
                        score += 0.5
                    per_item_scores.append(score)
                if per_item_scores:
                    content_reward += (sum(per_item_scores) / len(per_item_scores))

        # Extract and validate answer (final answer)
        # Maximum reward for final answer is 1.0
        final_match = re.search(r'<answer>\s*(\[.*?\])\s*</answer>', predict_str, re.DOTALL)
        if final_match:
            final_data = json.loads(final_match.group(1))
            if isinstance(final_data, list) and len(final_data) > 0:
                per_item_scores = []
                for item in final_data:
                    score = 0.0
                    if 'bbox_2d' in item and isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4:
                        score += 0.5
                    if 'point_2d' in item and isinstance(item['point_2d'], list) and len(item['point_2d']) == 2:
                        score += 0.5
                    per_item_scores.append(score)
                if per_item_scores:
                    content_reward += (sum(per_item_scores) / len(per_item_scores))
        
        # Check for <criticism> tags and the ADJUSTMENT flag within criticism
        # Maximum reward for criticism tag along with adjustment flag is 1.0
        criticism_match = re.search(r'<criticism>.*?ADJUSTMENT:\s*(YES|NO)\s*</criticism>', predict_str, re.DOTALL | re.IGNORECASE)
        if criticism_match:
            content_reward += 1.0 

    except Exception as e:
        # JSON parsing failure -> format content likely incorrect
        content_reward = 0.0
        logger.debug("Caught error in format reward calculation: %s", e)

    # Max Total: format_correct (1.0) + content_reward (up to 4.0)

    # Normalize the format_reward to be out of 1.0
    normalized_format_reward = (format_correct + content_reward) / 5.0
    return normalized_format_reward

def vision_reasoner_decision_reward(predict_str: str, ground_truth_type: bool = None) -> float:
    """Reward if the model correctly decides 'object' vs 'part'."""
    # ground_truth_type: True if query is asking for a part, False if asking for a whole object.
    try:
        target_match = re.search(r'<target>\s*(.*?)\s*</target>', predict_str, re.IGNORECASE)
        if not target_match:
            return 0.0

        target_text = target_match.group(1).strip().lower()
        # Determine model's decision: assume it mentions "part" if it's an object part query.
        predicted_is_part = "part" in target_text  # (this works if output contains "object part" or similar phrasing)
        actual_is_part = bool(ground_truth_type)
        return 1.0 if predicted_is_part == actual_is_part else 0.0
    except Exception as e:
        logger.debug("Error in vision_reasoner_decision_reward: %s", e)
        return 0.0
    
def compute_hungarian_match(pred_bboxes: np.ndarray, gt_bboxes: np.ndarray) -> float:
    iou_matrix = batch_iou(pred_bboxes, gt_bboxes)  # (M,N)

    cost_matrix = 1.0 - iou_matrix  # hungarian match on iou
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return iou_matrix, row_ind, col_ind

def compute_iou_reward(iou_matrix, matched_row_ind, matched_col_ind, M, N, weight) -> float:
    """
        Compute average iou given an iou matrix (M x N) between predicted and ground truth boxes, and the matched indices from Hungarian algorithm.
        Args:
            iou_matrix: (M x N) matrix of IoU scores
            matched_row_ind: List of matched row indices
            matched_col_ind: List of matched column indices
            M: Number of predicted boxes
            N: Number of ground truth boxes
            weight: Weighting factor for the IoU reward
    """
    avg_iou = 0.0

    for i, j in zip(matched_row_ind, matched_col_ind):
        iou_val = iou_matrix[i, j]
        avg_iou += iou_val # -> [0,1]

    max_count = max(M, N)
    avg_iou = avg_iou / max_count  # average iou over max(M,N)
    iou_reward = (weight * avg_iou) # weighted average iou as reward

    return iou_reward, avg_iou


def vision_reasoner_object_hint_reward(predict_str: str, gt_str: str = None, ground_truth_type: bool = None) -> float:
    """
        Gives object IoU reward for object_hint boxes if the target is "part".
        Maximum reward is 1.0
        Args:
            predict_str: Full prediction string
            gt_object_boxes: np.ndarray of shape (N,4) ground truth object boxes (not part boxes) if we are predicting parts. Otherwise, None 
            ground_truth_type: True if query is asking for a part, False if asking for a whole object
    """

    # if ground_truth_type is False (object), no reward for object_hint, just return
    if ground_truth_type is False:
        return 0.0

    # If ground_truth_type is True (part), and gt_object_boxes is None, this means we don't have object boxes to compare to. Just provide the whole reward of 1.0 if object_hint is non-empty
    if ground_truth_type is True and gt_str is None:
        try:
            hint_match = re.search(r'<object_hint>\s*(\[.*?\])\s*</object_hint>', predict_str, re.DOTALL)
            if not hint_match:
                return 0.0
            hint_data = json.loads(hint_match.group(1))
            if isinstance(hint_data, list) and len(hint_data) > 0:
                return 1.0
            else:
                return 0.0
        except Exception as e:
            logger.debug("Caught error in object hint reward calculation (no gt boxes): %s", e)
            return 0.0

    # if ground_truth_type is True (part), and gt_str is provided, compute IoU reward for object_hint boxes
    if gt_str is not None:
        try:
            hint_match = re.search(r'<object_hint>\s*(\[.*?\])\s*</object_hint>', predict_str, re.DOTALL)
            if not hint_match:
                return 0.0
            hint_data = json.loads(hint_match.group(1))
            if isinstance(hint_data, list) and len(hint_data) > 0:
                pred_object_boxes = np.array([item['bbox_2d'] for item in hint_data], dtype=np.float64)
                gt_object_boxes = np.array(json.loads(gt_str), dtype=np.float64)
                
                # Compute IoU reward for object_hint boxes
                iou_matrix, row_ind, col_ind = compute_hungarian_match(pred_object_boxes, gt_object_boxes)

                M, N = len(pred_object_boxes), len(gt_object_boxes)
                if M == 0 or N == 0:
                    if M == 0 and N == 0:
                        return 1.0  # no objects quried and none predicted, so its correct.
                    return 0.0
                
                _, object_hint_iou = compute_iou_reward(
                    iou_matrix, 
                    row_ind, 
                    col_ind, M, N, 
                    weight=1
                )
                
                return object_hint_iou
            else:
                return 0.0
        except Exception as e:
            logger.debug("Caught error in object hint reward calculation (gt boxes provided): %s", e)
            return 0.0
    
def parse_answer_predictions(predict_str: str, tag_name: str):
    """
    parse bounding boxes and points from a specified tag in the prediction string.
    for example, tag_name can be 'answer' or 'first_answer'.

    Args:
        predict_str: The full prediction string containing tagged predictions
        tag_name: Name of the tag to parse (e.g., 'answer', 'first_answer')
    
    Returns:
        boxes, points as numpy arrays:
        - boxes: np.ndarray of shape (N, 4) containing bounding boxes
        - points: np.ndarray of shape (N, 2) containing points
        empty arrays if parsing fails or no predictions found
    """
    boxes = []
    points = []
    
    try:
        # regex
        pattern = rf'<{tag_name}>\s*(.*?)\s*</{tag_name}>'
        match = re.search(pattern, predict_str, re.DOTALL)
        
        if match:
            data = json.loads(match.group(1))
            if isinstance(data, list) and len(data) > 0:
                # only add items that have both bbox and point
                for item in data:
                    if 'bbox_2d' in item and 'point_2d' in item:
                        boxes.append(np.array(item['bbox_2d'], dtype=np.float64))
                        points.append(np.array(item['point_2d'], dtype=np.float64))
    except Exception as e:
        logger.debug(f"Error parsing {tag_name} predictions into boxes and points: %s", e)
    
    # convert to np array
    boxes = np.array(boxes, dtype=np.float64) if len(boxes) > 0 else np.empty((0, 4))
    points = np.array(points, dtype=np.float64) if len(points) > 0 else np.empty((0, 2))
    
    return boxes, points

def compute_l1_threshold_from_boxes(gt_boxes: np.ndarray, percentage: float = 0.10) -> float:
    """
    Compute adaptive L1 threshold based on ground truth box sizes.
    Uses the mean diagonal of GT boxes.
    
    Args:
        gt_boxes: np.ndarray of shape (N, 4) containing ground truth boxes [x1, y1, x2, y2]
        percentage: Percentage of the mean diagonal to use as threshold (default: 0.10 or 10%)
    
    Returns:
        L1 threshold value (float)
    """
    if len(gt_boxes) == 0 or gt_boxes is None:
        return 10.0  # default if no GT boxes
    
    try:
        # width and height for each gt box
        widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1  # x2 - x1 + 1
        heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1  # y2 - y1 + 1
        
        # diagonal for each box: sqrt(w^2 + h^2)
        diagonals = np.sqrt(widths**2 + heights**2)
        
        # mean diagonal across all GT boxes
        mean_diagonal = np.mean(diagonals)
        
        threshold = percentage * mean_diagonal

        # clamp to reasonable range (min 3.0 to handle very small boxes, max 10.0 (default))
        threshold = np.clip(threshold, 3.0, 10.0)
    except Exception as e:
        logger.debug("Error in compute_l1_threshold: %s", e)
        threshold = 10.0  # default on error

    return float(threshold)

def compute_l1_threshold(gt_box: np.ndarray, percentage: float = 0.10) -> float:
    """
    Compute adaptive L1 threshold based on single ground truth box.
    Uses the diagonal of the GT box.
    
    Args:
        gt_box: np.ndarray of shape (4,) containing ground truth box [x1, y1, x2, y2]
        percentage: Percentage of diagonal to use as threshold (default: 0.10 or 10%)
    
    Returns:
        L1 threshold value (float)
    """
    if gt_box is None or len(gt_box) == 0:
        return 10.0  # default if no GT boxes
    
    try:
        width = gt_box[2] - gt_box[0] + 1
        height = gt_box[3] - gt_box[1] + 1
        
        # diagonal
        diagonal = np.sqrt(width**2 + height**2)

        threshold = int(percentage * diagonal)
        
        # clamp 
        threshold = np.clip(threshold, 3.0, 10.0)
    except Exception as e:
        logger.debug("Error in compute_l1_threshold: %s", e)
        threshold = 10.0  # default on error

    return float(threshold)

def compute_l1_reward(l1_matrix, matched_row_ind, matched_col_ind, M, N, gt_bboxes, weight) -> float:
    """
        Compute average l1 distance reward given an l1 distance matrix (M x N) between predicted and ground truth boxes, and the matched indices from Hungarian algorithm.
        Args:
            same as compute_iou_reward above except:
            gt_bboxes: ground truth boxes to compute adaptive l1 threshold
    """
    avg_l1 = 0.0

    for i, j in zip(matched_row_ind, matched_col_ind):
        l1_val = l1_matrix[i, j]
        l1_threshold = compute_l1_threshold(gt_bboxes[j])  # adaptive threshold per GT box
        avg_l1 += max(0, 1.0 - (l1_val / l1_threshold))  # scaled -> [0,1]. no reward if l1 > l1_threshold

    max_count = max(M, N)

    return (weight * avg_l1) / max_count  # weighted average l1 as reward

def compute_point_threshold(gt_box: np.ndarray, percentage: float = 0.20) -> float:
    """
    Compute point distance threshold based on ground truth box size.
    Uses the diagonal of the GT box as the scale.
    
    Args:
        gt_box: np.ndarray of shape (4,) containing ground truth box [x1, y1, x2, y2]
        percentage: Percentage of diagonal to use as threshold (default: 0.20 or 20%)
    
    Returns:
        Point distance threshold value (float)
    """
    if gt_box is None or len(gt_box) == 0:
        return 30.0  # fallback to default
    
    try:
        width = gt_box[2] - gt_box[0] + 1
        height = gt_box[3] - gt_box[1] + 1
        
        # diagonal
        diagonal = np.sqrt(width**2 + height**2)
    
        threshold = percentage * diagonal
        
        # clamp 
        threshold = np.clip(threshold, 5.0, 30.0)
        
        return float(threshold)
    except Exception as e:
        logger.debug("Error in compute_point_threshold: %s", e)
        return 30.0  # default on error

def compute_point_reward(
        pred_points, 
        gt_points, 
        pred_bboxes,
        gt_bboxes,
        row_ind, 
        col_ind, 
        weight
    ) -> float:

    if len(pred_points) == 0 and len(gt_points) == 0:
        return weight * 1.0  # both empty, full reward
    
    point_reward = 0.0 # necessary if no points or boxes, or no points inside boxes  

    success_count = 0
    if len(pred_points) > 0 and len(gt_points) > 0 and len(pred_bboxes) > 0 :
        # Ensure predicted points are inside their boxes for matched pairs
        points_inside = batch_points_in_box(pred_points, pred_bboxes)
        for i, j in zip(row_ind, col_ind):
            if i < len(pred_points) and j < len(gt_points):
                if not points_inside[i]:
                    continue  # skip if point not in its box
                dist = np.linalg.norm(pred_points[i] - gt_points[j])
                # adaptive threshold 
                # limitations with using points. we want to use a larger threshold so the chance of getting reward is higher
                threshold = compute_point_threshold(gt_bboxes[j])
                if dist < threshold:  
                    success_count += 1
        point_reward = success_count / max(len(pred_points), len(gt_points))

    return weight * float(point_reward)

def vision_reasoner_answer_reward(
        pred_bboxes: np.ndarray, 
        gt_bboxes: np.ndarray,
        final_points,
        gt_points,
        iou_reward_weight,
        l1_reward_weight,
        point_reward_weight
    ) -> float:
    """
    Reward for answer. Includes three components:
    - IoU reward on answer bounding boxes
    - L1 distance reward on answer bounding boxes
    - Point matching reward on answer points

    Args:
        pred_bboxes: np.ndarray of shape (M,4) predicted final bounding boxes
        gt_bboxes: np.ndarray of shape (N,4) ground truth bounding boxes
        final_points: np.ndarray of shape (M,2) predicted final points
        gt_points: np.ndarray of shape (N,2) ground truth points
        {iou/l1/point}_reward_weight: float weights for each component
    """
    # try boxes reward and iou 
    try:
        M, N = len(pred_bboxes), len(gt_bboxes)
        if M == 0 or N == 0:
            # if both empty, full reward
            if M == 0 and N == 0:
                return 1.0, 1.0, 1.0  # no objects asked and none predicted, so its correct.
            # otherwise no reward
            return 0.0, 0.0, 0.0

        # iou_matrix = batch_iou(pred_bboxes, gt_bboxes)
        # # cost matrix and hungarian matching based on iou only
        # cost_matrix = 1.0 - iou_matrix
        # row_ind, col_ind = linear_sum_assignment(cost_matrix)

        iou_matrix, row_ind, col_ind = compute_hungarian_match(pred_bboxes, gt_bboxes)
    except Exception as e:
        logger.debug("Error in final bbox hungarian matching: %s", e)
        # no reward if no matching can be done
        return 0.0, 0.0, 0.0

    try:
        # iou reward
        iou_reward, _ = compute_iou_reward(
            iou_matrix, 
            row_ind, 
            col_ind, 
            M, N, 
            iou_reward_weight
        )
    except Exception as e:
        logger.debug("Error in final answer iou reward: %s", e)
        iou_reward = 0.0

    try:
        # l1 reward
        l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes) 

        l1_reward = compute_l1_reward(
            l1_matrix, 
            row_ind, 
            col_ind, M=M, N=N, 
            gt_bboxes=gt_bboxes,
            weight=l1_reward_weight
        )
    except Exception as e:
        logger.debug("Error in final answer l1 reward: %s", e)
        l1_reward = 0.0

    try:
        point_reward = compute_point_reward(
            final_points, 
            gt_points, 
            pred_bboxes, 
            gt_bboxes,
            row_ind, 
            col_ind, 
            point_reward_weight
        )
    except Exception as e:
        logger.debug("Error in vision_reasoner_final_point_reward: %s", e)
        point_reward = 0.0

    return iou_reward, l1_reward, float(point_reward)


def vision_reasoner_compactness_reward(pred_bboxes: np.ndarray, gt_bboxes: np.ndarray, alpha=1.0) -> float:
    """Additional reward for tight bounding boxes (precision of coverage)."""
    try:
        M, N = len(pred_bboxes), len(gt_bboxes)
        if M == 0 or N == 0:
            return 0.0
        
        # Intersection and areas:
        x11, y11, x12, y12 = np.split(pred_bboxes, 4, axis=1)
        x21, y21, x22, y22 = np.split(gt_bboxes, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        inter = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)  # intersection area matrix
        pred_areas = ((x12 - x11 + 1) * (y12 - y11 + 1)).reshape(-1)  # area of each pred box
        gt_areas = ((x22 - x21 + 1) * (y22 - y21 + 1)).reshape(-1)    # area of each gt box

        # Compute IoU and intersection areas
        iou_matrix, row_ind, col_ind = compute_hungarian_match(pred_bboxes, gt_bboxes)
        
        precisions = []
        over_penalties = []
        compact_rewards = []
        for i, j in zip(row_ind, col_ind):
            prec = float(inter[i, j] / pred_areas[i])  # fraction of pred box that intersects ground truth
            precisions.append(prec)

            over = max(0, float(pred_areas[i] / gt_areas[j]) - 1)
            over_penalty = min(1.0, over) * -1.0
            over_penalties.append(over_penalty)

            compact_rewards.append(prec + over_penalty)

        return alpha * float(np.mean(compact_rewards))
    
    except Exception as e:
        logger.debug("Error in vision_reasoner_compactness_reward: %s", e)
        return 0.0

def vision_reasoner_non_repeat_reward(predict_str: str) -> float:
    """Penalize repeated sentences in the response."""
    non_repeat_reward = 1.0
    try:
        sentences = [s.strip() for s in predict_str.split('.') if s.strip()]
        seen = set()
        repeats = 0
        for sentence in sentences:
            if sentence in seen:
                repeats += 1
            if repeats >= 2:  # allow one repetition maybe, but >=2 repeated sentences -> fail
                non_repeat_reward = 0.0
                break
            seen.add(sentence)
    except Exception as e:
        logger.debug("caught error in vision_reasoner_non_repeat_reward: %s", e)
    return non_repeat_reward

def vision_reasoner_adjustment_consistency_reward(predict_str: str) -> float:
    """
    Penalty for inconsistency between ADJUSTMENT flag and actual changes made.
    - ADJUSTMENT: YES should result in different first_answer vs answer (penalize if same)
    - ADJUSTMENT: NO should result in same first_answer vs answer (penalize if different)

    Returns: reward (penalty) value (negative value to indicate penalty)
    """
    try:
        # Extract ADJUSTMENT flag
        criticism_match = re.search(r'<criticism>.*?ADJUSTMENT:\s*(YES|NO)', predict_str, re.DOTALL | re.IGNORECASE)
        if not criticism_match:
            return 0.0  # No adjustment flag found, no reward/penalty.
        
        adjustment_flag = criticism_match.group(1).strip().upper()
        
        # Extract first_answer and final answer
        first_match = re.search(r'<first_answer>\s*(\[.*?\])\s*</first_answer>', predict_str, re.DOTALL)
        final_match = re.search(r'<answer>\s*(\[.*?\])\s*</answer>', predict_str, re.DOTALL)
        
        if not first_match or not final_match:
            return 0.0  # Missing answers, can't evaluate
        
        first_data = json.loads(first_match.group(1))
        final_data = json.loads(final_match.group(1))
        
        # Check if answers are identical (simple equality check)
        answers_identical = (first_data == final_data)
        
        # Compute reward/penalty based on consistency
        if adjustment_flag == "YES":
            if answers_identical:
                # penalty: model said it would adjust but didn't
                return -1.0
               
        elif adjustment_flag == "NO":
            if not answers_identical:
                return -1.0
        
        return 0.0

    except Exception as e:
        logger.debug("Caught error in adjustment consistency reward calculation: %s", e)
        return 0.0

def check_containment(part_box, object_boxes) -> bool:
    """
    Check if a part box is properly contained within any of the given list of object boxes with additional constraints:
    - Part bbox must be fully contained within at least one object bbox
    - Part bbox must not be the same as the object bbox
    - Part bbox area must be smaller than the object bbox area

    Args:
        part_box: A np array or list [x1, y1, x2, y2] representing the part bounding box
        object_boxes: A list of nparrays or lists [[x1, y1, x2, y2], ...] representing the object bounding boxes

    Returns True if properly contained, False otherwise.
    """
    px1, py1, px2, py2 = part_box
    part_area = (px2 - px1 + 1) * (py2 - py1 + 1)
    
    # Check if this part is properly contained in any object box
    is_properly_contained = False
    
    for obj_box in object_boxes:
        ox1, oy1, ox2, oy2 = obj_box
        obj_area = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)
        
        is_spatially_contained = (px1 >= ox1 and py1 >= oy1 and px2 <= ox2 and py2 <= oy2)
        
        if not is_spatially_contained:
            continue
        
        # Not identical boxes 
        is_identical = (
            abs(px1 - ox1) < 1e-6 and 
            abs(py1 - oy1) < 1e-6 and 
            abs(px2 - ox2) < 1e-6 and 
            abs(py2 - oy2) < 1e-6
        )
        
        if is_identical:
            continue
        
        # Part area should be smaller than object area
        if part_area >= obj_area:
            continue
        
        # All constraints satisfied
        is_properly_contained = True
        break

    return is_properly_contained   

def vision_reasoner_part_containment_reward(predict_str: str, object_hint_gt_str: str , ground_truth_type: bool = None) -> float:
    """
    Reward for ensuring predicted part bboxes are contained within object_hint bboxes (predicted object boxes) and ground truth object boxes
    Only applies when target is "part" (ground_truth_type is True).
    
    This method should only be called on an example where the ground truth query is for a part. The method also checks and returns 0.0 if the ground truth type is not part. 
    
    Constraints:
    - Part bbox must be fully contained within at least one object bbox
    - Part bbox must not be the same as the object bbox
    - Part bbox area must be smaller than the object bbox area
    
    Args:
        predict_str: Full prediction string
        ground_truth_type: True if query is asking for a part, False if asking for a whole object
        object_hint_gt_str: JSON string of ground truth object boxes for when predicting parts
    
    Returns:
        Containment reward (0.0 to 1.0):
        - 1.0 if all part boxes satisfy containment constraints
        - Proportional score based on how many parts are properly contained
        - 0.0 if not a part query or no proper containment
    """
    try:
        # Only apply if target is "part"
        target_match = re.search(r'<target>\s*(object|part)\s*</target>', predict_str, re.IGNORECASE)
        if not target_match or target_match.group(1).strip().lower() != 'part' or not ground_truth_type:
            return 0.0
        
        # if part, but no object hints, then no reward
        hint_match = re.search(r'<object_hint>\s*(\[.*?\])\s*</object_hint>', predict_str, re.DOTALL)
        if not hint_match:
            return 0.0
        
        hint_data = json.loads(hint_match.group(1))
        if not isinstance(hint_data, list) or len(hint_data) == 0:
            return 0.0  # No object hints provided for part query

        predicted_object_boxes = np.array([item['bbox_2d'] for item in hint_data], dtype=np.float64)

        # object_hint_gt_str can be None if we were not able to extract object boxes. Also, it can potentially be an empty list too
        if object_hint_gt_str:
            ground_truth_object_boxes = np.array(json.loads(object_hint_gt_str), dtype=np.float64)
        else:
            ground_truth_object_boxes = None

        answer_match = re.search(r'<answer>\s*(\[.*?\])\s*</answer>', predict_str, re.DOTALL)
        if not answer_match:
            return 0.0
        
        answer_data = json.loads(answer_match.group(1))
        if not isinstance(answer_data, list) or len(answer_data) == 0:
            return 0.0
        
        part_boxes = np.array([item['bbox_2d'] for item in answer_data], dtype=np.float64)

        # Check containment: each part box should be fully inside at least one predicted object box and one ground truth object box  
        # with additional constraints: not identical and smaller area
        num_properly_contained = 0
        
        for part_box in part_boxes:

            is_properly_contained_in_predicted_object_boxes = check_containment(part_box, predicted_object_boxes)

            if ground_truth_object_boxes is not None:
                is_properly_contained_in_gt_object_boxes = check_containment(part_box, ground_truth_object_boxes)

                if is_properly_contained_in_predicted_object_boxes and is_properly_contained_in_gt_object_boxes:
                    num_properly_contained += 1
            else:
                # if no ground truth object boxes provided, only check predicted object boxes
                if is_properly_contained_in_predicted_object_boxes:
                    num_properly_contained += 1

        # reward = proportion of parts that are properly contained
        containment_reward = num_properly_contained / len(part_boxes)
        return float(containment_reward)
        
    except Exception as e:
        logger.debug("Caught error in part containment reward calculation: %s", e)
        return 0.0

def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """
    Compute the total reward for a batch of model outputs against the ground truth annotations.
    reward_inputs: a batch (list) of dicts. Keys include: 'response' (model output string), 'ground_truth' (JSON string of ground truth boxes and points), 'ground_truth_type' (boolean indicating if query is for part), 'object_hint_boxes' (ground truth object boxes for when predicting parts), 'baseline_iou' (baseline iou to improve upon. baseline boxes are obtained using SAM3).

    'object_hint_boxes' is a json string of list of boxes, e.g. '[ [x1,y1,x2,y2], ... ]' or it is None. 'object_hint_boxes' can be None when the target is 'object' (not part), or when we do not have gt boxes for the objects when predicting parts. 

    'baseline_iou' is a float indicating the baseline IoU score to improve upon. It will be 0.0 if there are no baseline boxes (e.g. SAM3 did not predict any boxes).
    """
    # Set score weights for each component
    weights = {
        'format_reward': 1.0, 
        'decision_reward': 1.0, 
        'object_hint_reward': 1.0, 
        'iou_reward': 2.0, 
        'point_reward': 1.0, 
        'l1_reward': 1.0, 
        'non_repeat_reward': 1.0, 
        'adjustment_consistency_reward': 1.0, 
        'improvement_reward': 1.0, 
        'part_containment_reward': 1.0,
        'compactness_reward': 1.0, 
    }

    scores = []
    for reward_input in reward_inputs:
        predict_str = reward_input['response']
        ground_truth = reward_input['ground_truth']
        ground_truth_type = reward_input.get('ground_truth_type', None)
        object_hint_boxes_str = reward_input['object_hint_boxes']
        baseline_iou = reward_input['baseline_iou'] # will be 0 if sam3 did not predict any boxes

        # Parse ground truth data. These are the gt boxes and points
        try:
            gt_data = json.loads(ground_truth)
            # ground truth could be a list of dicts or a dict with keys
            if isinstance(gt_data, dict):
                # ground_truth provided as {"bboxes": [...], "points": [...], "is_part": ...} for example
                if 'is_part' in gt_data:
                    ground_truth_type = bool(gt_data['is_part'])
                if 'objects' in gt_data:
                    # e.g {"objects": [{"bbox_2d": [...], "point_2d": [...]}, ...], ...}
                    gt_list = gt_data['objects']
                elif 'bboxes' in gt_data:
                    # e.g {"bboxes": [...], "points": [...]} separate lists
                    bboxes = gt_data['bboxes']
                    points = gt_data.get('points', [])
                    gt_list = []
                    for bb, pt in zip(bboxes, points):
                        gt_list.append({"bbox_2d": bb, "point_2d": pt})
                else:
                    # If dict but doesn't match expected format, try to interpret it as list
                    gt_list = gt_data
            else:
                gt_list = gt_data
        except Exception as e:
            logger.debug("Error parsing ground truth JSON, assuming empty ground truth: %s", e)
            gt_list = []

        # Convert ground truth list to numpy arrays for bboxes and points
        # ideally, gt should always have 'bbox_2d' and 'point_2d' keys for each item

        try:
            gt_bboxes = np.array([item['bbox_2d'] for item in gt_list], dtype=np.float64) if len(gt_list) > 0 else np.empty((0,4))
        except Exception as e:
            logger.debug("Error extracting ground truth bounding boxes, assuming empty: %s", e)
            gt_bboxes = np.empty((0,4))
        
        try:
            gt_points = np.array([item['point_2d'] for item in gt_list], dtype=np.float64) if len(gt_list) > 0 else np.empty((0,2))
        except Exception as e:
            logger.debug("Error extracting ground truth points, assuming empty: %s", e)
            gt_points = np.empty((0,2))


        # Compute each reward component

        # Max format reward returned here is 1.0
        format_reward = vision_reasoner_format_reward(predict_str)

        # Max decision reward returned here is 1.0
        decision_reward = vision_reasoner_decision_reward(predict_str, ground_truth_type)

        # Max object hint reward returned here is 1.0
        # reward only applies to parts, not objects
        object_hint_reward = vision_reasoner_object_hint_reward(predict_str, object_hint_boxes_str, ground_truth_type)

        # initial predictions parsing
        initial_boxes = []
        initial_points = []
        try:
            initial_boxes, initial_points = parse_answer_predictions(predict_str, tag_name='first_answer')
        except Exception as e:
            logger.debug("Error parsing initial predictions into boxes and points: %s", e)

        # Max iou reward  returned here is weights['iou_reward']
        # Max l1 reward returned here is weights['l1_reward']
        # Max point reward returned here is weights['point_reward']

        # use initial boxes and ground truth to compute initial prediction rewards
        initial_iou_reward = 0.0
        initial_l1_reward = 0.0
        initial_point_reward = 0.0

        if len(gt_bboxes) == 0 and len(initial_boxes) == 0:
            initial_iou_reward = 1.0
            initial_l1_reward = 1.0
            initial_point_reward = 1.0
        else:
            # Compute initial prediction rewards
            initial_iou_reward, initial_l1_reward, initial_point_reward = vision_reasoner_answer_reward(
                initial_boxes,
                gt_bboxes, 
                initial_points, 
                gt_points, 
                iou_reward_weight=weights['iou_reward'],
                l1_reward_weight=weights['l1_reward'],
                point_reward_weight=weights['point_reward']
            )

        initial_prediction_reward = initial_iou_reward + initial_l1_reward + initial_point_reward

        # final predictions parsing
        final_boxes = []
        final_points = []
        try:
            final_boxes, final_points = parse_answer_predictions(predict_str, tag_name='answer')
        except Exception as e:
            logger.debug("Error parsing final predictions into boxes and points: %s", e)

        # Use final boxes and ground truth to compute final prediction rewards
        # three components: iou, l1, point
        final_iou_reward = 0.0
        final_l1_reward = 0.0
        final_point_reward = 0.0
        if len(gt_bboxes) == 0 and len(final_boxes) == 0:
            # no objects to find and none predicted, consider it correct
            # give full credit
            final_iou_reward = 1.0
            final_l1_reward = 1.0
            final_point_reward = 1.0 
        else:
            # Compute final prediction rewards
            final_iou_reward, final_l1_reward, final_point_reward = vision_reasoner_answer_reward(
                final_boxes,
                gt_bboxes, 
                final_points, 
                gt_points, 
                iou_reward_weight=weights['iou_reward'],
                l1_reward_weight=weights['l1_reward'],
                point_reward_weight=weights['point_reward']
            )
        final_prediction_reward = final_iou_reward + final_l1_reward + final_point_reward  # combined final prediction reward.

        # improvement reward: based on improvement on iou, l1 and points
        # for l1: l1_1 - l1_0 (if negative, no reward)
        # for points: point_1 - point_0 (if negative, no reward)
        # combine these three into one improvement reward

        try:
            # iou improvement
            iou_improvement_reward = max(
                0, final_iou_reward - max(
                        initial_iou_reward, weights['iou_reward'] * baseline_iou
                    )
                )

            # iou_improvement reward is non-negative, and therefore does not penalize for drops in iou
            ## so we add an explicit penalty for iou drops
            if final_iou_reward < initial_iou_reward:
                iou_drop_penalty = final_iou_reward - initial_iou_reward  # negative value
            else:
                iou_drop_penalty = 0.0

            # l1 improvement
            l1_improvement_reward = max(0.0, final_l1_reward - initial_l1_reward)

            # point improvement
            point_improvement_reward = max(0.0, final_point_reward - initial_point_reward)

            # total improvement reward
            improvement_reward = iou_improvement_reward + l1_improvement_reward + point_improvement_reward + iou_drop_penalty  # iou_drop_penalty is negative or zero
           
        except Exception as e:
            logger.debug("Caught error computing improvement reward: %s", e)
            initial_iou_reward = 0.0
            improvement_reward = 0.0

        # Compactness reward for final boxes
        # Max compactness reward returned here is 1.0
        compact_reward = 0.0
        try:
            if len(final_boxes) > 0 and len(gt_bboxes) > 0:
                compact_reward = vision_reasoner_compactness_reward(final_boxes, gt_bboxes, alpha=1.0)
        except Exception as e:
            logger.debug("Caught error computing compactness reward: %s", e)
            compact_reward = 0.0

        # adjustment consistency reward/penalty
        # max adjustment consistency reward is 0.0 (no penalty)
        adjustment_reward = vision_reasoner_adjustment_consistency_reward(predict_str)

        # part containment reward (only for part queries)
        # this should only be called when ground_truth_type is True (part)
        # this creates a mismatch in the max total reward possible between part and object queries. We handle this via normalization
        if ground_truth_type:
            # Max containment reward returned here is 1.0
            containment_reward = vision_reasoner_part_containment_reward(predict_str, object_hint_boxes_str, ground_truth_type)
        else:
            containment_reward = 0.0

        # Max non-repetition reward returned here is 1.0
        non_repeat = vision_reasoner_non_repeat_reward(predict_str)

        # weigh the rewards according to the predefined weights at the start of this method 
        weighed_format_reward = weights['format_reward'] * format_reward
        weighed_decision_reward = weights['decision_reward'] * decision_reward
        weighed_object_hint_reward = weights['object_hint_reward'] * object_hint_reward
        
        # initial prediction reward
        initial_prediction_reward = (
            initial_iou_reward + 
            initial_l1_reward +
            initial_point_reward
        )

        # final prediction reward
        final_prediction_reward = (
            final_iou_reward +
            final_l1_reward +
            final_point_reward
        )

        weighed_compact_reward = weights['compactness_reward'] * compact_reward
        weighed_adjustment_consistency_reward = weights['adjustment_consistency_reward'] * adjustment_reward
        weighed_containment_reward = weights['part_containment_reward'] * containment_reward
        weighed_non_repeat = weights['non_repeat_reward'] * non_repeat

        # max total rewards 
        if ground_truth_type:
            # part query
            max_total_reward = (
                weights['format_reward'] +
                weights['decision_reward'] +
                weights['object_hint_reward'] +
                2 * (weights['iou_reward'] + weights['l1_reward'] + weights['point_reward']) +
                weights['compactness_reward'] +
                weights['part_containment_reward'] +
                weights['non_repeat_reward']
            )
        else:
            # object query
            max_total_reward = (
                weights['format_reward'] +
                weights['decision_reward'] +
                2 * (weights['iou_reward'] + weights['l1_reward'] + weights['point_reward']) +  # initial, final, improvement
                weights['compactness_reward'] +
                weights['non_repeat_reward']
            )

        # reward obtained in example
        total_reward = (
            weighed_format_reward +
            weighed_decision_reward +
            weighed_object_hint_reward +
            initial_prediction_reward +
            improvement_reward +
            final_prediction_reward +
            weighed_compact_reward +
            weighed_adjustment_consistency_reward +
            weighed_containment_reward +
            weighed_non_repeat
        )

        # overall normalized reward
        overall_normalized_reward = total_reward / max_total_reward

        scores.append(
            {
                'overall': float(overall_normalized_reward),
                'format': float(weighed_format_reward),
                'decision': float(weighed_decision_reward),
                'object_hint': float(weighed_object_hint_reward),
                'initial_iou': float(initial_iou_reward),
                'initial_l1': float(initial_l1_reward),
                'initial_point': float(initial_point_reward),
                'final_iou': float(final_iou_reward),
                'final_l1': float(final_l1_reward),
                'final_point': float(final_point_reward),
                'improvement': float(improvement_reward),
                'compactness': float(weighed_compact_reward),
                'adjustment_consistency': float(weighed_adjustment_consistency_reward),
                'part_containment': float(weighed_containment_reward),
                'non_repetition': float(weighed_non_repeat)
            }
        )

    return scores