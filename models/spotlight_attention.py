import torch
import torch.nn as nn
import torch.nn.functional as F

class SpotlightAttention(nn.Module):
    """
    改进版聚光灯注意力机制 - 解决分类问题
    
    主要改进：
    1. 添加多头注意力以增强特征表示
    2. 引入特征归一化以稳定训练
    3. 类别感知的注意力源初始化
    4. 添加位置编码以增强空间敏感性
    5. 改进特征融合策略
    """
    def __init__(self, in_channels, num_spotlights=100, hidden_dim=256, spotlight_dim=64, num_classes=80):
        super(SpotlightAttention, self).__init__()
        self.num_spotlights = num_spotlights
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 保留多头注意力机制
        self.num_heads = 4
        self.head_dim = spotlight_dim // self.num_heads
        
        # 改进1: 更好的初始化 - 使用正态分布初始化聚光灯源，标准差更小以避免初始注意力过于分散
        self.spotlight_sources = nn.Parameter(torch.randn(num_spotlights, spotlight_dim) * 0.02)
        
        # 改进2: 添加位置编码
        self.pos_encoding = nn.Conv2d(2, hidden_dim // 2, kernel_size=1)
        
        # 特征转换层
        self.feature_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),  # 使用BatchNorm2d替换LayerNorm
            nn.ReLU(inplace=True)
        )
        
        # 多头注意力投影
        self.key_proj = nn.Conv2d(hidden_dim, spotlight_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        # 改进4: 添加类别感知聚光灯源的预测
        self.class_token_proj = nn.Linear(hidden_dim, num_classes)
        
        # 改进5: 分离的边界框和类别头，增加深度
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4)  # (x, y, w, h)
        )
        
        # 分类头有更深的网络
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)  # 二值分类分数
        )
        
        # 改进6: 对抗性正则化损失权重
        self.diversity_weight = 0.1
        
        # 类别感知初始化
        self._initialize_class_aware()

    def _initialize_class_aware(self):
        """类别感知的初始化 - 鼓励不同聚光灯专注于不同类别"""
        # 初始化聚光灯的类别偏好 - 使初始分布更加均匀
        nn.init.xavier_uniform_(self.class_head[-1].weight)
        nn.init.zeros_(self.class_head[-1].bias)
        
        # 关键：确保类别投影层也得到正确初始化
        nn.init.xavier_uniform_(self.class_token_proj.weight)
        nn.init.zeros_(self.class_token_proj.bias)  # 为所有类别设置相同的起点
        
    def _make_position_encoding(self, height, width, device):
        """创建位置编码以帮助模型理解空间关系"""
        # 生成坐标网格
        y_coords = torch.linspace(-1, 1, height, device=device).view(height, 1).repeat(1, width)
        x_coords = torch.linspace(-1, 1, width, device=device).view(1, width).repeat(height, 1)
        
        # 堆叠坐标
        coords = torch.stack([y_coords, x_coords], dim=0).unsqueeze(0)  # [1, 2, height, width]
        return coords
    
    def forward(self, x):
        """
        前向传播，修复了维度问题但保留多头注意力机制
        
        Args:
            x (torch.Tensor): 输入特征图 (B, C, H, W)
            
        Returns:
            dict: 包含注意力图、预测框和置信度分数
        """
        batch_size, _, height, width = x.shape
        device = x.device
        
        # 创建位置编码
        pos_encoding = self._make_position_encoding(height, width, device)
        
        # 投影特征
        features = self.feature_proj(x)  # (B, hidden_dim, H, W)
        
        # 融合位置编码
        pos_features = self.pos_encoding(pos_encoding)  # [1, hidden_dim//2, H, W]
        pos_features = pos_features.repeat(batch_size, 1, 1, 1)  # [B, hidden_dim//2, H, W]
        
        # 拼接特征和位置编码
        combined_features = torch.cat([
            features[:, :self.hidden_dim//2, :, :], 
            pos_features
        ], dim=1)  # [B, hidden_dim, H, W]
        
        # 生成键和值
        keys = self.key_proj(combined_features)   # (B, spotlight_dim, H, W)
        values = self.value_proj(combined_features)  # (B, hidden_dim, H, W)
        
        # 为多头注意力重塑张量
        # 批量大小
        B = batch_size
        # 聚光灯数量
        N = self.num_spotlights
        # 头数
        h = self.num_heads
        # 每个头的维度
        d = self.head_dim
        # 特征图高宽
        H, W = height, width
        
        # 将关键字reshape为多头格式: [B, h*d, H, W] -> [B, h, d, H*W]
        keys = keys.view(B, h, d, H*W)
        
        # 将值reshape: [B, hidden_dim, H, W] -> [B, hidden_dim, H*W]
        values = values.flatten(2)  # [B, hidden_dim, H*W]
        
        # 将聚光灯查询reshape为多头格式: [N, h*d] -> [B, h, N, d]
        queries = self.spotlight_sources.view(1, N, h, d).permute(0, 2, 1, 3).expand(B, h, N, d)
        
        # 计算多头注意力分数: [B, h, N, d] x [B, h, d, H*W] -> [B, h, N, H*W]
        attention_logits = torch.einsum('bhnk,bhkm->bhnm', queries, keys) / (d ** 0.5)
        
        # 应用softmax: [B, h, N, H*W] -> [B, h, N, H*W]
        attention_weights_multihead = F.softmax(attention_logits, dim=-1)
        
        # 合并多头注意力: [B, h, N, H*W] -> [B, N, H*W]
        attention_weights = attention_weights_multihead.mean(dim=1)
        
        # 重塑注意力图以便可视化: [B, N, H*W] -> [B, N, H, W]
        attention_maps = attention_weights.view(B, N, H, W)
        
        # 应用注意力到值: [B, N, H*W] x [B, H*W, hidden_dim] -> [B, N, hidden_dim]
        spotlight_features = torch.bmm(attention_weights, values.transpose(1, 2))
        
        # 预测边界框和类别
        boxes = self.bbox_head(spotlight_features)  # (B, N, 4)
        scores = self.class_head(spotlight_features)  # (B, N, 1)
        
        # 类别感知投影
        class_logits = self.class_token_proj(spotlight_features)  # [B, N, num_classes]
        
        # 计算多样性损失以确保聚光灯源关注不同区域
        diversity_loss = self._compute_diversity_loss(attention_weights)
        
        return {
            'pred_boxes': boxes, 
            'pred_scores': scores.sigmoid(), 
            'pred_logits': class_logits,
            'attention_maps': attention_maps,
            'diversity_loss': diversity_loss,
            'spotlight_features': spotlight_features
        }
    
    def _compute_diversity_loss(self, attention_weights):
        """
        计算多样性损失以防止不同聚光灯关注相同区域
        
        Args:
            attention_weights (torch.Tensor): 注意力权重 (B, num_spotlights, H*W)
            
        Returns:
            torch.Tensor: 多样性损失
        """
        # 计算注意力图之间的成对余弦相似度
        attention_weights_norm = F.normalize(attention_weights, p=2, dim=2)
        similarity_matrix = torch.bmm(attention_weights_norm, attention_weights_norm.transpose(1, 2))
        
        # 我们希望非对角元素较小（不同聚光灯应关注不同区域）
        mask = torch.ones_like(similarity_matrix) - torch.eye(self.num_spotlights, device=similarity_matrix.device)
        diversity_loss = (similarity_matrix * mask).sum() / (self.num_spotlights * (self.num_spotlights - 1))
        
        return diversity_loss * self.diversity_weight