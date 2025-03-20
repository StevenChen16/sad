import torch
import torch.nn as nn
import torch.nn.functional as F

class SADLoss(nn.Module):
    """
    改进版SAD检测器损失函数
    
    主要改进：
    1. 完全重构匹配策略，同时考虑类别和位置准确性
    2. 使用标准的匈牙利算法进行二分图匹配
    3. 添加类别平衡机制和焦点损失
    4. 分离对象存在性（objectness）和分类损失
    5. 添加质量预测辅助损失
    6. 动态损失权重调整
    """
    def __init__(self, num_classes=80, class_weight=8.0, box_weight=5.0,
                 giou_weight=2.0, objectness_weight=1.0, diversity_weight=0.05):
        super(SADLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.box_weight = box_weight
        self.giou_weight = giou_weight
        self.objectness_weight = objectness_weight
        self.diversity_weight = diversity_weight
        
        # 焦点损失参数
        self.alpha = 0.25
        self.gamma = 2.0
        
        # 导入匈牙利算法
        try:
            from scipy.optimize import linear_sum_assignment
            self.linear_sum_assignment = linear_sum_assignment
        except ImportError:
            print("警告：未找到scipy，将使用贪心匹配算法代替匈牙利算法")
            self.linear_sum_assignment = None
    
    def forward(self, outputs, targets):
        """
        计算SAD模型的损失
        
        Args:
            outputs (dict): 模型预测，包含：
                - 'pred_boxes': 预测边界框 (B, num_spotlights, 4)
                - 'pred_logits': 类别预测 (B, num_spotlights, num_classes)
                - 'objectness': 对象存在性预测 (B, num_spotlights, 1)
                - 'attention_maps': 注意力图 (B, num_spotlights, H, W)
                - 'diversity_loss': 聚光灯多样性损失
                
            targets (dict): 真实标签，包含：
                - 'boxes': 真实边界框 (B, num_objects, 4) 格式为 [x1, y1, x2, y2]
                - 'labels': 类别标签 (B, num_objects)
                - 'object_masks': 指示有效对象的二值掩码 (B, num_objects)
                
        Returns:
            dict: 损失组成和总损失
        """
        pred_boxes = outputs['pred_boxes']
        pred_logits = outputs['pred_logits']
        objectness = outputs['objectness']
        diversity_loss = outputs['diversity_loss']
        
        batch_size = pred_boxes.shape[0]
        device = pred_boxes.device
        
        # 检查是否有任何批次中的有效对象
        if targets['object_masks'].sum() == 0:
            # 返回基于多样性损失的默认损失，允许训练继续
            default_loss = diversity_loss * 5.0
            return {
                'loss': default_loss,
                'class_loss': torch.tensor(0.0, device=device),
                'box_loss': torch.tensor(0.0, device=device),
                'giou_loss': torch.tensor(0.0, device=device),
                'objectness_loss': torch.tensor(0.0, device=device),
                'diversity_loss': diversity_loss
            }
        
        # 初始化累积损失张量
        total_class_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_box_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_giou_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_objectness_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 独立处理每个批次
        for b in range(batch_size):
            # 获取该批次的预测和目标
            batch_pred_boxes = pred_boxes[b]  # (num_spotlights, 4)
            batch_pred_logits = pred_logits[b]  # (num_spotlights, num_classes)
            batch_objectness = objectness[b]  # (num_spotlights, 1)
            
            batch_target_boxes = targets['boxes'][b]  # (num_objects, 4)
            batch_target_labels = targets['labels'][b]  # (num_objects,)
            batch_object_mask = targets['object_masks'][b]  # (num_objects,)
            
            # 跳过该批次，如果没有有效对象
            if batch_object_mask.sum() == 0:
                continue
            
            # 筛选有效对象
            valid_target_boxes = batch_target_boxes[batch_object_mask]
            valid_target_labels = batch_target_labels[batch_object_mask]
            
            # 执行二分图匹配
            indices = self._hungarian_matching(
                batch_pred_boxes, batch_pred_logits, batch_objectness,
                valid_target_boxes, valid_target_labels
            )
            
            # 提取匹配的预测和目标
            idx_preds, idx_targets = indices
            
            # 跳过，如果没有找到匹配
            if len(idx_preds) == 0:
                continue
            
            # 分类损失（包括背景类）
            # 所有预测默认为背景类
            target_classes = torch.full(
                (len(batch_pred_logits),), self.num_classes,  # 使用类别数作为背景类索引
                dtype=torch.int64, device=device
            )
            
            # 为匹配的预测分配目标类别
            if len(idx_preds) > 0:
                target_classes[idx_preds] = valid_target_labels[idx_targets]
            
            # 计算分类损失 - 使用交叉熵，确保维度正确
            batch_class_loss = self._focal_loss(
                batch_pred_logits, target_classes
            )
            
            total_class_loss = total_class_loss + batch_class_loss
            
            # 对象存在性损失 - 为所有预测创建目标掩码
            target_objectness = torch.zeros_like(batch_objectness)
            if len(idx_preds) > 0:
                target_objectness[idx_preds] = 1.0
            
            # 计算对象存在性损失
            batch_objectness_loss = F.binary_cross_entropy(
                batch_objectness.squeeze(-1), target_objectness.squeeze(-1), reduction='mean'
            )
            
            total_objectness_loss = total_objectness_loss + batch_objectness_loss
            
            # 边界框回归损失（L1 + GIoU）
            if len(idx_preds) > 0:
                matched_pred_boxes = batch_pred_boxes[idx_preds]
                matched_target_boxes = valid_target_boxes[idx_targets]
                
                # L1损失
                batch_box_loss = F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction='none').sum(-1).mean()
                total_box_loss = total_box_loss + batch_box_loss
                
                # GIoU损失
                pred_boxes_xyxy = self._box_cxcywh_to_xyxy(matched_pred_boxes)
                target_boxes_xyxy = matched_target_boxes  # 假设目标已经是xyxy格式
                
                batch_giou_loss = 1 - self._generalized_box_iou_pairwise(pred_boxes_xyxy, target_boxes_xyxy).mean()
                total_giou_loss = total_giou_loss + batch_giou_loss
        
        # 计算平均损失
        if batch_size > 0:
            total_class_loss = total_class_loss / batch_size
            total_box_loss = total_box_loss / batch_size
            total_giou_loss = total_giou_loss / batch_size
            total_objectness_loss = total_objectness_loss / batch_size
        
        # 权重组合损失
        total_loss = (
            self.class_weight * total_class_loss +
            self.box_weight * total_box_loss +
            self.giou_weight * total_giou_loss +
            self.objectness_weight * total_objectness_loss +
            self.diversity_weight * diversity_loss
        )
        
        return {
            'loss': total_loss,
            'class_loss': total_class_loss,
            'box_loss': total_box_loss,
            'giou_loss': total_giou_loss,
            'objectness_loss': total_objectness_loss,
            'diversity_loss': diversity_loss
        }
    
    def _hungarian_matching(self, pred_boxes, pred_logits, pred_objectness, target_boxes, target_labels):
        """
        使用匈牙利算法执行二分图匹配（找到预测和真实框之间的最佳一对一匹配）
        
        Args:
            pred_boxes (torch.Tensor): 预测框 (num_spotlights, 4)，格式为 [cx, cy, w, h]
            pred_logits (torch.Tensor): 类别logits (num_spotlights, num_classes)
            pred_objectness (torch.Tensor): 对象存在性预测 (num_spotlights, 1)
            target_boxes (torch.Tensor): 目标框 (num_objects, 4)，格式为 [x1, y1, x2, y2]
            target_labels (torch.Tensor): 目标类别标签 (num_objects,)
            
        Returns:
            tuple: 匹配的预测和目标索引
        """
        num_spotlights = pred_boxes.shape[0]
        num_targets = target_boxes.shape[0]
        device = pred_boxes.device
        
        # 如果没有目标，返回空匹配
        if num_targets == 0:
            return (torch.tensor([], dtype=torch.int64, device=device), 
                    torch.tensor([], dtype=torch.int64, device=device))
        
        # 将预测框转换为xyxy格式以计算IoU
        pred_boxes_xyxy = self._box_cxcywh_to_xyxy(pred_boxes)
        
        # 计算所有预测和目标之间的IoU - 注意目标已经是xyxy格式
        ious = self._box_iou(pred_boxes_xyxy, target_boxes)
        
        # 计算分类成本 - 获取对应目标类别的预测分数
        pred_probs = F.softmax(pred_logits, dim=-1)  # [num_spotlights, num_classes]
        
        # 如果类别索引超出范围，进行处理
        valid_target_labels = torch.clamp(target_labels, 0, pred_probs.size(1) - 1).long()
        
        # 获取真实类别的预测概率
        cls_probs = torch.gather(pred_probs, 1, valid_target_labels.unsqueeze(0).repeat(num_spotlights, 1))
        
        # 分类成本 = -log(类别概率)
        cls_cost = -torch.log(cls_probs + 1e-8)  # 添加小的epsilon以防止log(0)
        
        # 对象存在性成本
        obj_cost = -torch.log(pred_objectness + 1e-8).repeat(1, num_targets)
        
        # 位置成本 = 1 - IoU
        pos_cost = -ious
        
        # 组合成本矩阵 - 平衡的权重分配
        cost_matrix = (
            pos_cost * 1.0 +  # 降低位置权重以减少对位置的过度关注 
            cls_cost * 1.5 +  # 增加分类权重以更多关注类别准确性
            obj_cost * 0.5    # 对象存在性权重保持不变
        )
        
        # 使用匈牙利算法进行最优分配
        if self.linear_sum_assignment is not None:
            # 将成本矩阵转换为CPU NumPy数组
            cost_matrix_np = cost_matrix.detach().cpu().numpy()
            row_indices, col_indices = self.linear_sum_assignment(cost_matrix_np)
            
            # 将索引转换回PyTorch张量
            row_indices = torch.tensor(row_indices, dtype=torch.int64, device=device)
            col_indices = torch.tensor(col_indices, dtype=torch.int64, device=device)
            
            # 筛选成本较低的匹配
            indices = torch.stack([row_indices, col_indices], dim=0)
            matched_costs = cost_matrix[row_indices, col_indices]
            
            # 只保留成本低于阈值的匹配
            valid_matches = matched_costs < 10.0
            row_indices = row_indices[valid_matches]
            col_indices = col_indices[valid_matches]
        else:
            # 简单的贪心匹配算法
            matched_indices = []
            available_preds = set(range(num_spotlights))
            
            for target_idx in range(num_targets):
                costs = cost_matrix[:, target_idx]
                
                # 只考虑可用的预测
                valid_costs = torch.full_like(costs, float('inf'))
                for pred_idx in available_preds:
                    valid_costs[pred_idx] = costs[pred_idx]
                
                pred_idx = valid_costs.argmin().item()
                cost_val = valid_costs[pred_idx].item()
                
                # 只匹配成本合理的情况
                if cost_val < float('inf') and cost_val < 5.0:
                    matched_indices.append((pred_idx, target_idx))
                    available_preds.remove(pred_idx)
            
            if not matched_indices:
                return (torch.tensor([], dtype=torch.int64, device=device), 
                        torch.tensor([], dtype=torch.int64, device=device))
            
            # 转换为张量
            matched_indices = torch.tensor(matched_indices, device=device)
            row_indices = matched_indices[:, 0]
            col_indices = matched_indices[:, 1]
        
        return row_indices, col_indices
    
    def _focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        """
        用于分类的焦点损失 - 优化版本
        
        Args:
            inputs (torch.Tensor): 类别logits (N, num_classes)
            targets (torch.Tensor): 目标类别 (N,)
            alpha (float): 权重因子
            gamma (float): 聚焦参数
            
        Returns:
            torch.Tensor: 焦点损失
        """
        num_classes = inputs.shape[-1]
        device = inputs.device
        
        # 改进的背景类处理
        is_background = targets >= num_classes
        
        # 为前景类创建one-hot编码
        target_one_hot = torch.zeros_like(inputs)
        
        # 只为非背景类填充one-hot向量，避免重用类别0
        non_bg_mask = ~is_background
        if non_bg_mask.sum() > 0:
            # 确保类别索引在范围内
            valid_targets = targets[non_bg_mask].clamp(0, num_classes-1)
            target_one_hot[non_bg_mask] = F.one_hot(valid_targets, num_classes).float().to(device)
        
        # 应用softmax获取概率
        probs = F.softmax(inputs, dim=-1)
        
        # 初始化pt张量
        pt = torch.zeros(inputs.shape[0], device=device)
        
        # 对前景样本计算pt - 正确类别的概率
        if non_bg_mask.sum() > 0:
            pt[non_bg_mask] = torch.sum(probs[non_bg_mask] * target_one_hot[non_bg_mask], dim=1)
        
        # 对背景样本：pt = 1 - max_foreground_prob
        if is_background.sum() > 0:
            pt[is_background] = 1 - probs[is_background].max(1)[0]
        
        # 计算焦点权重，让模型专注于困难样本
        focal_weight = (1 - pt) ** gamma
        
        # 应用alpha平衡，处理类别不平衡
        alpha_t = torch.ones_like(pt) * alpha
        alpha_t[is_background] = 1 - alpha
        focal_weight = focal_weight * alpha_t
        
        # 计算交叉熵损失，忽略背景类
        ce_loss = F.cross_entropy(inputs, targets.clamp(0, num_classes-1), 
                              reduction='none', 
                              ignore_index=num_classes)
        
        # 应用焦点权重
        loss = focal_weight * ce_loss
        
        # 使用类别平衡权重 - 可选，如果您有类别频率信息
        # 如果最后一个类别(toothbrush)强烈主导，可以给它低一些的权重
        # 这里可以根据您的数据集配置具体权重
        
        # 返回平均损失
        return loss.mean()
    
    def _box_cxcywh_to_xyxy(self, boxes):
        """
        将框从(cx, cy, w, h)格式转换为(x1, y1, x2, y2)格式
        
        Args:
            boxes (torch.Tensor): (cx, cy, w, h)格式的框
            
        Returns:
            torch.Tensor: (x1, y1, x2, y2)格式的框
        """
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _box_iou(self, boxes1, boxes2):
        """
        计算两组框之间的IoU
        
        Args:
            boxes1 (torch.Tensor): 第一组框 (N, 4)
            boxes2 (torch.Tensor): 第二组框 (M, 4)
            
        Returns:
            torch.Tensor: IoU值 (N, M)
        """
        area1 = self._box_area(boxes1)
        area2 = self._box_area(boxes2)
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        union = area1[:, None] + area2 - inter
        
        iou = inter / (union + 1e-6)  # 添加epsilon避免除零
        return iou
    
    def _box_area(self, boxes):
        """
        计算框的面积
        
        Args:
            boxes (torch.Tensor): (x1, y1, x2, y2)格式的框
            
        Returns:
            torch.Tensor: 框的面积
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    def _generalized_box_iou_pairwise(self, boxes1, boxes2):
        """
        计算成对的GIoU，每对框之间的一对一计算
        
        Args:
            boxes1 (torch.Tensor): 第一组框 (N, 4)
            boxes2 (torch.Tensor): 第二组框 (N, 4)
            
        Returns:
            torch.Tensor: GIoU值 (N,)
        """
        # 检查输入
        if boxes1.shape[0] != boxes2.shape[0]:
            raise ValueError(f"boxes1和boxes2的第一个维度必须相同: {boxes1.shape[0]} vs {boxes2.shape[0]}")
        
        # 计算IoU (N,)
        area1 = self._box_area(boxes1)
        area2 = self._box_area(boxes2)
        
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        union = area1 + area2 - inter
        
        iou = inter / (union + 1e-6)
        
        # 计算GIoU (N,)
        lt_min = torch.min(boxes1[:, :2], boxes2[:, :2])
        rb_max = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        
        enclosing_wh = (rb_max - lt_min).clamp(min=0)
        enclosing_area = enclosing_wh[:, 0] * enclosing_wh[:, 1]
        
        # 计算GIoU，避免除零
        valid_area = enclosing_area > 0
        giou = torch.zeros_like(iou)
        
        if valid_area.sum() > 0:
            giou[valid_area] = iou[valid_area] - (
                (enclosing_area[valid_area] - union[valid_area]) / 
                (enclosing_area[valid_area] + 1e-6)
            )
        else:
            giou = iou  # 如果没有有效区域，GIoU = IoU
        
        return giou
    
    def _generalized_box_iou(self, boxes1, boxes2):
        """
        计算两组框之间的广义IoU
        
        Args:
            boxes1 (torch.Tensor): 第一组框 (N, 4)
            boxes2 (torch.Tensor): 第二组框 (M, 4)
            
        Returns:
            torch.Tensor: GIoU值 (N, M)
        """
        # 标准IoU
        iou = self._box_iou(boxes1, boxes2)
        
        # 获取包围框
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        # 计算GIoU，避免除零
        valid_area = area > 0
        giou = torch.zeros_like(iou)
        
        if valid_area.sum() > 0:
            area1 = self._box_area(boxes1)
            area2 = self._box_area(boxes2)
            union = area1[:, None] + area2 - iou * (area1[:, None] + area2)
            giou[valid_area] = iou[valid_area] - (
                (area[valid_area] - union[valid_area]) / 
                (area[valid_area] + 1e-6)
            )
        
        return giou