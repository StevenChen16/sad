import torch
import torch.nn as nn
import torch.nn.functional as F

from .spotlight_attention import SpotlightAttention

class ConvBlock(nn.Module):
    """
    Standard convolution block with batch normalization and leaky ReLU activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """
    Residual block with two convolutions and a skip connection
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels // 2, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(channels // 2, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class CSPBlock(nn.Module):
    """
    Cross Stage Partial block
    """
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super(CSPBlock, self).__init__()
        self.downsample = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2)
        
        self.part1_conv = ConvBlock(out_channels, out_channels // 2, kernel_size=1, padding=0)
        self.part2_conv = ConvBlock(out_channels, out_channels // 2, kernel_size=1, padding=0)
        
        self.blocks = nn.Sequential(*[ResBlock(out_channels // 2) for _ in range(num_blocks)])
        
        self.transition = ConvBlock(out_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        x = self.downsample(x)
        part1 = self.part1_conv(x)
        part2 = self.part2_conv(x)
        
        part1 = self.blocks(part1)
        
        combined = torch.cat([part1, part2], dim=1)
        return self.transition(combined)


class SADBackbone(nn.Module):
    """
    Backbone network for the SAD model, based on CSPDarknet but simplified
    """
    def __init__(self):
        super(SADBackbone, self).__init__()
        
        # Initial convolution
        self.conv1 = ConvBlock(3, 32, kernel_size=3)
        
        # Downsample and CSP stages
        self.stage1 = CSPBlock(32, 64, num_blocks=1)    # 1/2
        self.stage2 = CSPBlock(64, 128, num_blocks=2)   # 1/4
        self.stage3 = CSPBlock(128, 256, num_blocks=8)  # 1/8
        self.stage4 = CSPBlock(256, 512, num_blocks=8)  # 1/16
        self.stage5 = CSPBlock(512, 1024, num_blocks=4) # 1/32
    
    def forward(self, x):
        x = self.conv1(x)
        
        c1 = self.stage1(x)      # 1/2 resolution
        c2 = self.stage2(c1)     # 1/4 resolution
        c3 = self.stage3(c2)     # 1/8 resolution
        c4 = self.stage4(c3)     # 1/16 resolution
        c5 = self.stage5(c4)     # 1/32 resolution
        
        return [c3, c4, c5]  # Return multiple feature levels for FPN


class FPNNeck(nn.Module):
    """
    Feature Pyramid Network (FPN) for feature fusion
    """
    def __init__(self):
        super(FPNNeck, self).__init__()
        
        # Lateral connections
        self.lateral_c3 = ConvBlock(256, 256, kernel_size=1, padding=0)
        self.lateral_c4 = ConvBlock(512, 256, kernel_size=1, padding=0)
        self.lateral_c5 = ConvBlock(1024, 256, kernel_size=1, padding=0)
        
        # FPN connections
        self.fpn_c5 = ConvBlock(256, 256, kernel_size=3, padding=1)
        self.fpn_c4 = ConvBlock(256, 256, kernel_size=3, padding=1)
        self.fpn_c3 = ConvBlock(256, 256, kernel_size=3, padding=1)
        
        # Output convolutions
        self.out_conv = ConvBlock(256 * 3, 512, kernel_size=3, padding=1)
    
    def forward(self, features):
        c3, c4, c5 = features
        
        # Lateral connections
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        
        # FPN connections
        p5 = self.fpn_c5(p5)
        p4 = self.fpn_c4(p4)
        p3 = self.fpn_c3(p3)
        
        # Resize all to the largest feature map size (p3)
        p4_up = F.interpolate(p4, size=p3.shape[2:], mode='nearest')
        p5_up = F.interpolate(p5, size=p3.shape[2:], mode='nearest')
        
        # Concatenate all feature maps
        features_cat = torch.cat([p3, p4_up, p5_up], dim=1)
        
        # Final output
        out = self.out_conv(features_cat)
        
        return out


class SpotlightDetectionHead(nn.Module):
    """
    改进版检测头，使用聚光灯注意力代替锚框
    
    主要改进：
    1. 重构分类头使用辅助损失
    2. 添加特征增强网络
    3. 实现多尺度特征融合
    4. 为每个类别添加单独的预测层 
    5. 添加类别平衡策略
    """
    def __init__(self, in_channels, num_classes=80, num_spotlights=100):
        super(SpotlightDetectionHead, self).__init__()
        self.num_classes = num_classes
        hidden_dim = 256
        
        # 改进1: 特征增强网络
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 改进2: 使用改进版聚光灯注意力机制
        self.spotlight_attention = SpotlightAttention(
            in_channels=hidden_dim,
            num_spotlights=num_spotlights,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
        
        # 改进3: 更深的分类头（专门处理多类别预测）
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 改进4: 添加辅助任务 - 预测对象存在性（objectness）
        self.objectness_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 类别平衡初始化
        self._initialize_balanced()
        
    def _initialize_balanced(self):
        """初始化分类头以实现更平衡的类别预测"""
        # 使用Xavier初始化最后一层，为不同类别提供相同的起点
        nn.init.xavier_uniform_(self.class_head[-1].weight)
        # 关键：初始化偏置为0，确保所有类别起点相同
        nn.init.zeros_(self.class_head[-1].bias)
        
    def forward(self, x):
        """
        检测头的前向传播
        
        Args:
            x (torch.Tensor): 来自backbone+neck的特征图
            
        Returns:
            dict: 包含预测框、类别分数和注意力图
        """
        # 增强特征
        enhanced_features = self.feature_enhancer(x)
        
        # 获取聚光灯注意力结果
        spotlight_results = self.spotlight_attention(enhanced_features)
        
        # 获取聚光灯特征
        spotlight_features = spotlight_results['spotlight_features']
        
        # 使用专门的分类头预测类别，而不是直接使用SpotlightAttention的结果
        class_logits = self.class_head(spotlight_features)
        
        # 辅助任务 - 预测对象存在性
        objectness = self.objectness_head(spotlight_features).sigmoid()
        
        # 获取最终检测结果
        pred_boxes = spotlight_results['pred_boxes']
        attention_maps = spotlight_results['attention_maps']
        diversity_loss = spotlight_results['diversity_loss']
        
        return {
            'pred_boxes': pred_boxes,                    # (B, num_spotlights, 4)
            'pred_logits': class_logits,                 # (B, num_spotlights, num_classes)
            'objectness': objectness,                    # (B, num_spotlights, 1)
            'attention_maps': attention_maps,            # 用于可视化
            'diversity_loss': diversity_loss,            # 额外损失项
            'spotlight_features': spotlight_features     # 用于调试
        }


class SADDetector(nn.Module):
    """
    Full Spotlight Attention Detection (SAD) model
    
    This model replaces traditional anchor-based detectors with a novel
    spotlight attention mechanism for object detection.
    """
    def __init__(self, num_classes=80, num_spotlights=100):
        super(SADDetector, self).__init__()
        
        self.backbone = SADBackbone()
        self.neck = FPNNeck()
        self.head = SpotlightDetectionHead(
            in_channels=512,
            num_classes=num_classes, 
            num_spotlights=num_spotlights
        )
    
    def forward(self, x):
        """
        Forward pass of the full SAD detector
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W)
            
        Returns:
            dict: Contains predicted boxes, class scores, and attention maps
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Feature fusion in neck
        fused_features = self.neck(features)
        
        # Detection head with spotlight attention
        output = self.head(fused_features)
        
        return output
    
    def get_loss(self, outputs, targets):
        """
        Compute the loss for training
        
        Args:
            outputs (dict): Model outputs from forward pass
            targets (dict): Ground truth targets
            
        Returns:
            dict: Dictionary of individual loss components and total loss
        """
        # In a full implementation, this would include:
        # 1. Bipartite matching between predictions and ground truth (similar to DETR)
        # 2. Classification loss (focal loss or cross-entropy)
        # 3. Bounding box regression loss (GIoU, L1, etc.)
        # 4. Spotlight diversity loss (already computed in spotlight attention)
        
        # Placeholder for a complete implementation
        return {"total_loss": torch.tensor(0.0, requires_grad=True)}