import torch

def greedy_assignment(cost_matrix):
    """
    贪心算法执行二分图匹配，避免使用匈牙利算法
    
    Args:
        cost_matrix (torch.Tensor): 成本矩阵 [num_preds, num_targets]
        
    Returns:
        tuple: 匹配的预测和目标索引
    """
    num_preds, num_targets = cost_matrix.shape
    device = cost_matrix.device
    
    # 初始化输出
    matched_pred_indices = []
    matched_target_indices = []
    
    # 创建可用的预测和目标集合
    available_preds = set(range(num_preds))
    available_targets = set(range(num_targets))
    
    # 迭代直到没有更多可用的预测或目标
    while available_preds and available_targets:
        # 创建可用预测和目标的成本子矩阵
        available_pred_list = list(available_preds)
        available_target_list = list(available_targets)
        
        # 提取可用预测和目标的成本子矩阵
        available_costs = cost_matrix[available_pred_list][:, available_target_list]
        
        # 找到最小成本的索引
        min_cost, min_index = available_costs.view(-1).min(dim=0)
        
        # 如果最小成本太高，停止匹配
        if min_cost > 5.0:
            break
        
        # 转换为2D索引
        sub_pred_idx = min_index.item() // len(available_target_list)
        sub_target_idx = min_index.item() % len(available_target_list)
        
        # 转换为原始索引
        pred_idx = available_pred_list[sub_pred_idx]
        target_idx = available_target_list[sub_target_idx]
        
        # 添加到结果
        matched_pred_indices.append(pred_idx)
        matched_target_indices.append(target_idx)
        
        # 移除已使用的预测和目标
        available_preds.remove(pred_idx)
        available_targets.remove(target_idx)
    
    # 转换为张量
    if matched_pred_indices:
        return (
            torch.tensor(matched_pred_indices, device=device, dtype=torch.int64),
            torch.tensor(matched_target_indices, device=device, dtype=torch.int64)
        )
    else:
        # 如果没有匹配，返回空张量
        return (
            torch.tensor([], device=device, dtype=torch.int64),
            torch.tensor([], device=device, dtype=torch.int64)
        )
