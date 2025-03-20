import torch
import torch.nn.functional as F

def filter_predictions(outputs, confidence_threshold=0.5):
    """
    根据置信度阈值过滤预测
    
    改进:
    1. 添加更多调试信息
    2. 正确处理多类别情况
    3. 确保数值稳定性
    4. 添加类别平衡后处理
    
    Args:
        outputs (dict): 模型输出
        confidence_threshold (float): 置信度阈值
        
    Returns:
        dict: 过滤后的预测
    """
    # 获取输出分量
    pred_boxes = outputs['pred_boxes']  # [batch_size, num_spotlights, 4]
    pred_logits = outputs['pred_logits']  # [batch_size, num_spotlights, num_classes]
    attention_maps = outputs.get('attention_maps', None)  # [batch_size, num_spotlights, H, W]
    objectness = outputs.get('objectness', None)  # [batch_size, num_spotlights, 1]
    
    print(f"[DEBUG] filter_predictions - input shapes:")
    print(f"  - pred_boxes: {pred_boxes.shape}")
    print(f"  - pred_logits: {pred_logits.shape}")
    if attention_maps is not None:
        print(f"  - attention_maps: {attention_maps.shape}")
    print(f"  - confidence_threshold: {confidence_threshold}")
    
    batch_size = pred_boxes.shape[0]
    num_spotlights = pred_boxes.shape[1]
    num_classes = pred_logits.shape[-1]
    
    # 始终使用softmax处理多类别情况，让类别间竞争
    # 应用softmax获取类别概率
    pred_probs = F.softmax(pred_logits, dim=-1)
    
    # 打印一些类别概率分布信息
    if batch_size > 0:
        first_sample_probs = pred_probs[0]  # 第一个批次的概率
        cls_means = first_sample_probs.mean(dim=0)  # 各类别平均概率
        top5_classes = torch.topk(cls_means, min(5, num_classes))
        print(f"[DEBUG] Top-5 class probabilities: {top5_classes.values.cpu().numpy()}")
        print(f"[DEBUG] Top-5 class indices: {top5_classes.indices.cpu().numpy()}")
    
    # 应用如下更标准的方法压缩概率差距做出更有决断力的预测
    # 对每个框，找到最高类别概率和对应的类别ID
    max_probs, class_ids = pred_probs.max(dim=-1)  # [batch_size, num_spotlights]
    
    # 如果有objectness，将其与类别概率相乘得到最终置信度
    if objectness is not None:
        objectness = objectness.squeeze(-1)  # [batch_size, num_spotlights]
        confidence = max_probs * objectness
    else:
        confidence = max_probs
    
    # 创建结果列表
    batch_boxes, batch_scores, batch_labels, batch_attention = [], [], [], []
    
    # 处理每个批次
    for b in range(batch_size):
        # 获取该批次数据
        boxes = pred_boxes[b]  # [num_spotlights, 4]
        scores = confidence[b]  # [num_spotlights]
        labels = class_ids[b]  # [num_spotlights]
        
        # 添加调试信息
        if b == 0:  # 只显示第一个批次的调试信息
            print(f"[DEBUG] Batch {b} scores distribution:")
            print(f"  - min: {scores.min():.4f}, max: {scores.max():.4f}")
            print(f"  - mean: {scores.mean():.4f}, std: {scores.std():.4f}")
            print(f"  - Above threshold: {(scores > confidence_threshold).sum()}/{len(scores)}")
        
        # 根据置信度阈值过滤
        keep_indices = torch.nonzero(scores > confidence_threshold).squeeze(-1)
        
        # 如果没有预测，添加空张量
        if len(keep_indices) == 0:
            batch_boxes.append(torch.zeros((0, 4), device=boxes.device))
            batch_scores.append(torch.zeros((0,), device=scores.device))
            batch_labels.append(torch.zeros((0,), device=labels.device, dtype=torch.int64))
            if attention_maps is not None:
                batch_attention.append(torch.zeros((0, attention_maps.shape[2], attention_maps.shape[3]), 
                                                  device=attention_maps.device))
            continue
        
        # 提取保留的预测
        filtered_boxes = boxes[keep_indices]  # [n_kept, 4]
        filtered_scores = scores[keep_indices]  # [n_kept]
        filtered_labels = labels[keep_indices]  # [n_kept]
        
        if attention_maps is not None:
            filtered_attention = attention_maps[b][keep_indices]  # [n_kept, H, W]
        else:
            filtered_attention = None
        
        # 添加到批次结果
        batch_boxes.append(filtered_boxes)
        batch_scores.append(filtered_scores)
        batch_labels.append(filtered_labels)
        if attention_maps is not None:
            batch_attention.append(filtered_attention)
    
    if b == 0:
        print(f"[DEBUG] After confidence filtering: {len(batch_boxes[0])}/{num_spotlights} boxes kept")
    
    # 创建结果字典
    filtered_outputs = {
        'pred_boxes': batch_boxes,
        'pred_scores': batch_scores,
        'pred_labels': batch_labels
    }
    
    if attention_maps is not None:
        filtered_outputs['attention_maps'] = batch_attention
    
    return filtered_outputs


def convert_to_xyxy(boxes):
    """
    将框从(cx, cy, w, h)格式转换为(x1, y1, x2, y2)格式
    
    Args:
        boxes (torch.Tensor): (cx, cy, w, h)格式的框
        
    Returns:
        torch.Tensor: (x1, y1, x2, y2)格式的框
    """
    print(f"[DEBUG] convert_to_xyxy - input shape: {boxes.shape}")
    
    if len(boxes) > 1:
        print(f"[DEBUG] Sample boxes before conversion: {boxes[:2]}")
    
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    
    result = torch.stack([x1, y1, x2, y2], dim=-1)
    
    if len(boxes) > 1:
        print(f"[DEBUG] Sample boxes after conversion: {result[:2]}")
    
    return result


def apply_nms(filtered_outputs, nms_threshold=0.5, max_detections=100):
    """
    应用非极大值抑制(NMS)
    
    改进:
    1. 添加调试输出
    2. 按类别应用NMS
    3. 限制每个类别的最大检测数
    4. 处理空预测
    
    Args:
        filtered_outputs (dict): 过滤后的预测
        nms_threshold (float): NMS IoU阈值
        max_detections (int): 每个图像最大检测数
        
    Returns:
        dict: 应用NMS后的预测
    """
    batch_boxes = filtered_outputs['pred_boxes']
    batch_scores = filtered_outputs['pred_scores']
    batch_labels = filtered_outputs['pred_labels']
    batch_attention = filtered_outputs.get('attention_maps', None)
    
    print(f"[DEBUG] apply_nms - nms_threshold: {nms_threshold}")
    print(f"[DEBUG] Number of boxes before NMS: {[len(boxes) for boxes in batch_boxes]}")
    
    # 创建结果列表
    processed_boxes, processed_scores, processed_labels, processed_attention = [], [], [], []
    
    # 处理每个批次
    for b in range(len(batch_boxes)):
        boxes = batch_boxes[b]
        scores = batch_scores[b]
        labels = batch_labels[b]
        
        if batch_attention is not None:
            attention = batch_attention[b]
        else:
            attention = None
        
        # 显示调试信息
        if b == 0:
            print(f"[DEBUG] Batch {b} before NMS:")
            print(f"  - boxes shape: {boxes.shape}")
            print(f"  - scores shape: {scores.shape}")
            print(f"  - labels shape: {labels.shape}")
            if len(boxes) > 0:
                print(f"  - First 3 boxes: {boxes[:3]}")
                print(f"  - First 3 scores: {scores[:3]}")
                print(f"  - First 3 labels: {labels[:3]}")
        
        # 如果没有预测，添加空张量
        if len(boxes) == 0:
            processed_boxes.append(torch.zeros((0, 4), device=boxes.device))
            processed_scores.append(torch.zeros((0,), device=scores.device))
            processed_labels.append(torch.zeros((0,), device=labels.device, dtype=torch.int64))
            if attention is not None:
                processed_attention.append(torch.zeros((0, attention.shape[1], attention.shape[2]), 
                                                      device=attention.device))
            continue
        
        # 将框转换为(x1, y1, x2, y2)格式用于NMS
        boxes_xyxy = convert_to_xyxy(boxes)
        
        # 按类别应用NMS
        unique_labels = torch.unique(labels)
        
        nms_boxes, nms_scores, nms_labels, nms_attention = [], [], [], []
        
        for class_id in unique_labels:
            # 获取该类别的预测
            class_mask = labels == class_id
            class_boxes = boxes_xyxy[class_mask]
            class_scores = scores[class_mask]
            
            if b == 0:
                print(f"[DEBUG] Batch {b}, Class {class_id}: {len(class_boxes)} boxes before NMS")
            
            # 如果该类别没有预测，跳过
            if len(class_boxes) == 0:
                continue
            
            # 应用NMS
            keep_indices = torch.ops.torchvision.nms(class_boxes, class_scores, nms_threshold)
            
            # 限制每个类别的最大检测数
            keep_indices = keep_indices[:max_detections]
            
            if b == 0:
                print(f"[DEBUG] Batch {b}, Class {class_id}: {len(keep_indices)}/{len(class_boxes)} boxes after NMS")
            
            # 获取保留的预测
            nms_boxes.append(boxes[class_mask][keep_indices])
            nms_scores.append(scores[class_mask][keep_indices])
            nms_labels.append(labels[class_mask][keep_indices])
            
            if attention is not None:
                nms_attention.append(attention[class_mask][keep_indices])
        
        # 合并所有类别的保留预测
        if nms_boxes:
            nms_boxes = torch.cat(nms_boxes)
            nms_scores = torch.cat(nms_scores)
            nms_labels = torch.cat(nms_labels)
            
            if attention is not None and nms_attention:
                nms_attention = torch.cat(nms_attention)
            else:
                nms_attention = None
            
            # 按置信度排序
            sorted_indices = torch.argsort(nms_scores, descending=True)
            sorted_indices = sorted_indices[:max_detections]  # 限制最大检测数
            
            processed_boxes.append(nms_boxes[sorted_indices])
            processed_scores.append(nms_scores[sorted_indices])
            processed_labels.append(nms_labels[sorted_indices])
            
            if nms_attention is not None:
                processed_attention.append(nms_attention[sorted_indices])
        else:
            # 没有保留的预测
            processed_boxes.append(torch.zeros((0, 4), device=boxes.device))
            processed_scores.append(torch.zeros((0,), device=scores.device))
            processed_labels.append(torch.zeros((0,), device=labels.device, dtype=torch.int64))
            
            if attention is not None:
                processed_attention.append(torch.zeros((0, attention.shape[1], attention.shape[2]), 
                                                      device=attention.device))
    
    print(f"[DEBUG] Number of boxes after NMS: {[len(boxes) for boxes in processed_boxes]}")
    
    # 创建结果字典
    processed_outputs = {
        'pred_boxes': processed_boxes,
        'pred_scores': processed_scores,
        'pred_labels': processed_labels
    }
    
    if batch_attention is not None:
        processed_outputs['attention_maps'] = processed_attention
    
    return processed_outputs


def post_process(outputs, confidence_threshold=0.5, nms_threshold=0.5, max_detections=100):
    """
    后处理模型输出
    
    Args:
        outputs (dict): 模型输出
        confidence_threshold (float): 置信度阈值
        nms_threshold (float): NMS IoU阈值
        max_detections (int): 每个图像最大检测数
        
    Returns:
        dict: 后处理后的预测
    """
    # 1. 根据置信度阈值过滤预测
    filtered_outputs = filter_predictions(outputs, confidence_threshold)
    
    # 2. 应用NMS
    processed_outputs = apply_nms(filtered_outputs, nms_threshold, max_detections)
    
    print(f"Processed outputs keys: {processed_outputs.keys()}")
    
    return processed_outputs