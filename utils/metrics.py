"""
Metrics utilities for SAD Detector
"""
import numpy as np
import torch
from tqdm import tqdm
from colorama import Fore, Style


def bbox_iou(box1, box2):
    """
    Calculate IoU between box1 and multiple box2
    
    Args:
        box1: Single box [x1, y1, x2, y2]
        box2: Multiple boxes [[x1, y1, x2, y2], ...]
        
    Returns:
        numpy.ndarray: IoU values
    """
    # Ensure box2 is an array
    if not isinstance(box2, np.ndarray):
        box2 = np.array(box2)
    
    # Add batch dimension to box1 if needed
    if len(box2.shape) == 2 and len(np.array(box1).shape) == 1:
        box1 = np.array([box1])
    
    # Calculate intersection area
    x1 = np.maximum(box1[:, 0], box2[:, 0])
    y1 = np.maximum(box1[:, 1], box2[:, 1])
    x2 = np.minimum(box1[:, 2], box2[:, 2])
    y2 = np.minimum(box1[:, 3], box2[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-10)
    
    return iou


def calculate_map(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5):
    """
    Calculate mAP (mean Average Precision) for object detection
    
    Args:
        pred_boxes: List of predicted bounding boxes [x1, y1, x2, y2]
        pred_scores: List of confidence scores for each prediction
        pred_classes: List of predicted class indices
        gt_boxes: List of ground truth bounding boxes [x1, y1, x2, y2]
        gt_classes: List of ground truth class indices
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        dict: Dictionary with mAP metrics (mAP50, mAP50-95, precision, recall)
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return {
            'mAP50': 0.0,
            'mAP50-95': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    # Convert lists to numpy arrays if they aren't already
    if not isinstance(pred_boxes, np.ndarray):
        pred_boxes = np.array(pred_boxes)
    if not isinstance(pred_scores, np.ndarray):
        pred_scores = np.array(pred_scores)
    if not isinstance(pred_classes, np.ndarray):
        pred_classes = np.array(pred_classes)
    if not isinstance(gt_boxes, np.ndarray):
        gt_boxes = np.array(gt_boxes)
    if not isinstance(gt_classes, np.ndarray):
        gt_classes = np.array(gt_classes)
    
    # Get unique class IDs in ground truth
    unique_classes = np.unique(np.concatenate([gt_classes, pred_classes]))
    
    # For each class, calculate AP
    ap_per_class = {}
    ap_per_class_50_95 = {}
    
    for class_id in unique_classes:
        # Get detections and ground truths for this class
        class_pred_indices = np.where(pred_classes == class_id)[0]
        class_gt_indices = np.where(gt_classes == class_id)[0]
        
        # Skip if no ground truth or predictions for this class
        if len(class_pred_indices) == 0 or len(class_gt_indices) == 0:
            ap_per_class[class_id] = 0.0
            ap_per_class_50_95[class_id] = 0.0
            continue
        
        # Get predictions and ground truths for this class
        class_pred_boxes = pred_boxes[class_pred_indices]
        class_pred_scores = pred_scores[class_pred_indices]
        class_gt_boxes = gt_boxes[class_gt_indices]
        
        # Sort predictions by score (highest first)
        score_indices = np.argsort(class_pred_scores)[::-1]
        class_pred_boxes = class_pred_boxes[score_indices]
        class_pred_scores = class_pred_scores[score_indices]
        
        # Initialize matches array
        gt_matched = np.zeros(len(class_gt_boxes), dtype=bool)
        
        # Calculate precision and recall at different IoU thresholds
        aps = []
        
        for iou_t in np.linspace(0.5, 0.95, 10):  # 0.5, 0.55, 0.6, ..., 0.95
            # Reset matches for each IoU threshold
            gt_matched = np.zeros(len(class_gt_boxes), dtype=bool)
            
            # Initialize true and false positives
            tp = np.zeros(len(class_pred_boxes))
            fp = np.zeros(len(class_pred_boxes))
            
            # Match each prediction to ground truth
            for pred_idx, pred_box in enumerate(class_pred_boxes):
                # Skip if all ground truths have been matched
                if gt_matched.all():  # Fixed: Use .all() instead of np.all(gt_matched)
                    fp[pred_idx] = 1
                    continue
                
                # Calculate IoU with all ground truths
                ious = bbox_iou(pred_box, class_gt_boxes)
                
                # Find best matching ground truth
                best_iou_idx = np.argmax(ious)
                best_iou = ious[best_iou_idx]
                
                # Check if IoU passes threshold and ground truth isn't matched yet
                if best_iou >= iou_t and not gt_matched[best_iou_idx]:
                    tp[pred_idx] = 1
                    gt_matched[best_iou_idx] = True
                else:
                    fp[pred_idx] = 1
            
            # Calculate cumulative true and false positives
            cumsum_tp = np.cumsum(tp)
            cumsum_fp = np.cumsum(fp)
            
            # Calculate precision and recall
            precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-10)
            recall = cumsum_tp / (len(class_gt_boxes) + 1e-10)
            
            # Ensure precision is monotonically decreasing
            for i in range(len(precision) - 1, 0, -1):
                precision[i - 1] = max(precision[i - 1], precision[i])
            
            # Calculate average precision using all points interpolation
            indices = np.where(np.diff(np.concatenate(([0], recall))))[0]  # Points where recall changes
            ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1]) if len(indices) > 0 else 0.0
            
            aps.append(ap)
        
        # Store AP for this class at IoU 0.5
        ap_per_class[class_id] = aps[0]  # AP at IoU=0.5
        
        # Store mAP for this class (average over all IoU thresholds)
        ap_per_class_50_95[class_id] = np.mean(aps)
    
    # Calculate mAP as mean of APs over all classes
    mAP50 = np.mean(list(ap_per_class.values()))
    mAP50_95 = np.mean(list(ap_per_class_50_95.values()))
    
    # Calculate overall precision and recall (at IoU 0.5)
    total_predictions = len(pred_boxes)
    total_gt = len(gt_boxes)
    
    # Match each prediction to ground truth at IoU 0.5
    gt_matched = np.zeros(total_gt, dtype=bool)
    total_tp = 0
    
    # Sort all predictions by score
    all_indices = np.argsort(pred_scores)[::-1]
    sorted_pred_boxes = pred_boxes[all_indices]
    sorted_pred_classes = pred_classes[all_indices]
    
    for pred_idx, (pred_box, pred_class) in enumerate(zip(sorted_pred_boxes, sorted_pred_classes)):
        # Find ground truths of same class
        matching_gt_indices = np.where(gt_classes == pred_class)[0]
        
        if len(matching_gt_indices) == 0:
            continue
        
        matching_gt_boxes = gt_boxes[matching_gt_indices]
        
        # Calculate IoU with matching ground truths
        ious = bbox_iou(pred_box, matching_gt_boxes)
        
        # Find best matching ground truth
        if len(ious) > 0:
            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]
            
            # Check if IoU passes threshold and ground truth isn't matched yet
            actual_gt_idx = matching_gt_indices[best_iou_idx]
            if best_iou >= iou_threshold and not gt_matched[actual_gt_idx]:
                total_tp += 1
                gt_matched[actual_gt_idx] = True
    
    # Calculate precision and recall
    precision = total_tp / (total_predictions + 1e-10)
    recall = total_tp / (total_gt + 1e-10)
    
    return {
        'mAP50': float(mAP50),
        'mAP50-95': float(mAP50_95),
        'precision': float(precision),
        'recall': float(recall)
    }


def evaluate_detections(model, dataloader, device, config, epoch, max_samples=100, post_process_fn=None):
    """
    Evaluate model by calculating mAP metrics
    
    Args:
        model: SAD model
        dataloader: Validation data loader
        device: Device to use
        config: Configuration
        epoch: Current epoch number for logging
        max_samples: Maximum number of samples to evaluate (for speed)
        post_process_fn: Function to post-process model outputs
        
    Returns:
        dict: Evaluation metrics
    """
    if post_process_fn is None:
        from utils.post_processing import post_process
        post_process_fn = post_process
    
    model.eval()
    
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_classes = []
    all_gt_boxes = []
    all_gt_classes = []
    
    samples_processed = 0
    
    progress_bar = tqdm(dataloader, 
                     desc=f"{Fore.BLUE}Eval{Style.RESET_ALL} Calculating mAP metrics", 
                     bar_format="{l_bar}{bar:10}{r_bar}",
                     unit=" batch")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(progress_bar):
            if samples_processed >= max_samples:
                break
                
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            outputs = model(images)
            
            # Post-process outputs
            processed_outputs = post_process_fn(
                outputs,
                confidence_threshold=config.eval['confidence_threshold'],
                nms_threshold=config.eval['nms_threshold'],
                max_detections=config.eval['max_detections']
            )
            
            # Debug: print keys in processed_outputs
            print(f"Processed outputs keys: {processed_outputs.keys()}")
            
            # Extract predictions and ground truths for each image in batch
            batch_size = images.size(0)
            for i in range(batch_size):
                if samples_processed >= max_samples:
                    break
                
                # Get predictions for this image
                try:
                    # Check for different possible key names
                    if 'pred_boxes' in processed_outputs:
                        pred_boxes = processed_outputs['pred_boxes'][i].cpu().numpy()
                        
                        # Get scores
                        if 'pred_scores' in processed_outputs:
                            pred_scores = processed_outputs['pred_scores'][i].cpu().numpy()
                        else:
                            print(f"{Fore.YELLOW}⚠ No pred_scores found. Using default scores.{Style.RESET_ALL}")
                            pred_scores = np.ones(len(pred_boxes))
                        
                        # Get class labels
                        if 'pred_labels' in processed_outputs:
                            pred_classes = processed_outputs['pred_labels'][i].cpu().numpy()
                        elif 'pred_classes' in processed_outputs:
                            pred_classes = processed_outputs['pred_classes'][i].cpu().numpy()
                        else:
                            print(f"{Fore.YELLOW}⚠ No pred_labels or pred_classes found. Using zeros.{Style.RESET_ALL}")
                            pred_classes = np.zeros(len(pred_boxes), dtype=np.int32)
                    
                    elif 'boxes' in processed_outputs:
                        pred_boxes = processed_outputs['boxes'][i].cpu().numpy()
                        pred_scores = processed_outputs['scores'][i].cpu().numpy()
                        pred_classes = processed_outputs['labels'][i].cpu().numpy()
                    else:
                        # If we don't have boxes, skip this batch
                        print(f"{Fore.YELLOW}⚠ Post-processing output doesn't contain boxes. Skipping batch.{Style.RESET_ALL}")
                        continue
                except Exception as e:
                    print(f"{Fore.YELLOW}⚠ Error processing predictions: {e}{Style.RESET_ALL}")
                    continue
                
                # Get ground truths for this image
                try:
                    # Get valid object mask for this image
                    object_mask = targets['object_masks'][i].cpu().numpy()
                    
                    # Apply mask to get valid ground truth
                    valid_boxes = targets['boxes'][i][object_mask].cpu().numpy()
                    valid_labels = targets['labels'][i][object_mask].cpu().numpy()
                    
                    gt_boxes = valid_boxes
                    gt_classes = valid_labels
                    
                    # Skip images with no ground truth
                    if len(gt_boxes) == 0:
                        continue
                    
                    # Convert ground truth boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format
                    gt_boxes_xyxy = np.zeros_like(gt_boxes)
                    gt_boxes_xyxy[:, 0] = gt_boxes[:, 0] - gt_boxes[:, 2] / 2  # x1 = cx - w/2
                    gt_boxes_xyxy[:, 1] = gt_boxes[:, 1] - gt_boxes[:, 3] / 2  # y1 = cy - h/2
                    gt_boxes_xyxy[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2] / 2  # x2 = cx + w/2
                    gt_boxes_xyxy[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3] / 2  # y2 = cy + h/2
                    gt_boxes = gt_boxes_xyxy
                    
                except Exception as e:
                    print(f"{Fore.YELLOW}⚠ Error processing ground truth: {e}{Style.RESET_ALL}")
                    continue
                
                # Append to global lists
                all_pred_boxes.append(pred_boxes)
                all_pred_scores.append(pred_scores)
                all_pred_classes.append(pred_classes)
                all_gt_boxes.append(gt_boxes)
                all_gt_classes.append(gt_classes)
                
                samples_processed += 1
    
    # Initialize metrics
    metrics = {
        'mAP50': 0.0,
        'mAP50-95': 0.0,
        'precision': 0.0,
        'recall': 0.0
    }
    
    # If no samples processed, return empty metrics
    if samples_processed == 0:
        print(f"{Fore.YELLOW}⚠ No valid samples for evaluation. Returning zero metrics.{Style.RESET_ALL}")
        return metrics
    
    # Flatten prediction and ground truth lists for mAP calculation
    try:
        # Check if we have any valid predictions and ground truths
        if not all_pred_boxes or not all_gt_boxes:
            print(f"{Fore.YELLOW}⚠ No valid predictions or ground truths. Returning zero metrics.{Style.RESET_ALL}")
            return metrics
        
        # Handle potentially empty arrays
        non_empty_pred = [p for p in all_pred_boxes if p.size > 0]
        non_empty_scores = [s for s in all_pred_scores if s.size > 0]
        non_empty_pred_classes = [c for c in all_pred_classes if c.size > 0]
        non_empty_gt = [g for g in all_gt_boxes if g.size > 0]
        non_empty_gt_classes = [c for c in all_gt_classes if c.size > 0]
        
        # If any category is empty after filtering, return zero metrics
        if not non_empty_pred or not non_empty_scores or not non_empty_pred_classes or not non_empty_gt or not non_empty_gt_classes:
            print(f"{Fore.YELLOW}⚠ No valid data after filtering empty arrays. Returning zero metrics.{Style.RESET_ALL}")
            return metrics
        
        flat_pred_boxes = np.concatenate(non_empty_pred)
        flat_pred_scores = np.concatenate(non_empty_scores)
        flat_pred_classes = np.concatenate(non_empty_pred_classes)
        flat_gt_boxes = np.concatenate(non_empty_gt)
        flat_gt_classes = np.concatenate(non_empty_gt_classes)
        
    except Exception as e:
        print(f"{Fore.YELLOW}⚠ Error concatenating boxes: {e}{Style.RESET_ALL}")
        if all_pred_boxes:
            print(f"Pred boxes shapes: {[box.shape for box in all_pred_boxes if hasattr(box, 'shape')]}")
        if all_gt_boxes:
            print(f"GT boxes shapes: {[box.shape for box in all_gt_boxes if hasattr(box, 'shape')]}")
        return metrics
    
    # Calculate mAP metrics
    try:
        metrics = calculate_map(
            flat_pred_boxes, 
            flat_pred_scores, 
            flat_pred_classes, 
            flat_gt_boxes, 
            flat_gt_classes,
            iou_threshold=0.5
        )
    except Exception as e:
        print(f"{Fore.YELLOW}⚠ Error calculating mAP: {e}{Style.RESET_ALL}")
    
    return metrics


def compute_per_class_ap(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, class_names, iou_threshold=0.5):
    """
    Compute per-class Average Precision
    
    Args:
        pred_boxes: List of predicted bounding boxes [x1, y1, x2, y2]
        pred_scores: List of confidence scores for each prediction
        pred_classes: List of predicted class indices
        gt_boxes: List of ground truth bounding boxes [x1, y1, x2, y2]
        gt_classes: List of ground truth class indices
        class_names: List of class names
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        dict: Dictionary with per-class AP
    """
    # Convert to numpy arrays
    if not isinstance(pred_boxes, np.ndarray):
        pred_boxes = np.array(pred_boxes)
    if not isinstance(pred_scores, np.ndarray):
        pred_scores = np.array(pred_scores)
    if not isinstance(pred_classes, np.ndarray):
        pred_classes = np.array(pred_classes)
    if not isinstance(gt_boxes, np.ndarray):
        gt_boxes = np.array(gt_boxes)
    if not isinstance(gt_classes, np.ndarray):
        gt_classes = np.array(gt_classes)
    
    # Get unique class IDs in the data
    unique_classes = np.unique(np.concatenate([gt_classes, pred_classes]))
    
    # For each class, calculate AP
    ap_per_class = {}
    
    for class_id in unique_classes:
        # Get detections and ground truths for this class
        class_pred_indices = np.where(pred_classes == class_id)[0]
        class_gt_indices = np.where(gt_classes == class_id)[0]
        
        # Skip if no ground truth for this class
        if len(class_gt_indices) == 0:
            ap_per_class[class_names[class_id]] = 0.0
            continue
        
        # Skip if no predictions for this class
        if len(class_pred_indices) == 0:
            ap_per_class[class_names[class_id]] = 0.0
            continue
        
        # Get predictions and ground truths for this class
        class_pred_boxes = pred_boxes[class_pred_indices]
        class_pred_scores = pred_scores[class_pred_indices]
        class_gt_boxes = gt_boxes[class_gt_indices]
        
        # Sort predictions by score (highest first)
        score_indices = np.argsort(class_pred_scores)[::-1]
        class_pred_boxes = class_pred_boxes[score_indices]
        class_pred_scores = class_pred_scores[score_indices]
        
        # Initialize matches array
        gt_matched = np.zeros(len(class_gt_boxes), dtype=bool)
        
        # Initialize true and false positives
        tp = np.zeros(len(class_pred_boxes))
        fp = np.zeros(len(class_pred_boxes))
        
        # Match each prediction to ground truth
        for pred_idx, pred_box in enumerate(class_pred_boxes):
            # Skip if there are no unmatched ground truths
            if gt_matched.all():  # Fixed: Use .all() instead of np.all(gt_matched)
                fp[pred_idx] = 1
                continue
            
            # Calculate IoU with all ground truths
            ious = bbox_iou(pred_box, class_gt_boxes)
            
            # Find best matching ground truth
            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]
            
            # Check if IoU passes threshold and ground truth isn't matched yet
            if best_iou >= iou_threshold and not gt_matched[best_iou_idx]:
                tp[pred_idx] = 1
                gt_matched[best_iou_idx] = True
            else:
                fp[pred_idx] = 1
        
        # Calculate cumulative true and false positives
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        
        # Calculate precision and recall
        precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-10)
        recall = cumsum_tp / (len(class_gt_boxes) + 1e-10)
        
        # Ensure precision is monotonically decreasing
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        
        # Calculate average precision using all points interpolation
        indices = np.where(np.diff(np.concatenate(([0], recall))))[0]  # Points where recall changes
        ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1]) if len(indices) > 0 else 0.0
        
        # Store AP for this class
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        ap_per_class[class_name] = float(ap)
    
    return ap_per_class