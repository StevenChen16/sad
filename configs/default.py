"""
Configuration for SAD (Spotlight Attention Detection) model.

This file contains the default configuration settings for training, evaluation, 
and inference with the SAD model.
"""

class Config:
    # Model parameters
    model = dict(
        name='sad',
        num_classes=80,  # COCO has 80 classes
        num_spotlights=100,  # Number of spotlight sources
        backbone=dict(
            name='csp_darknet',
        ),
        neck=dict(
            name='fpn',
        ),
        head=dict(
            name='spotlight_detection_head',
            hidden_dim=256,
            spotlight_dim=64,
        ),
    )
    
    # Dataset parameters
    dataset = dict(
        train=dict(
            img_dir='data/coco/train2017',
            ann_file='data/coco/annotations/instances_train2017.json',
            batch_size=8,
            num_workers=4,
        ),
        val=dict(
            img_dir='data/coco/val2017',
            ann_file='data/coco/annotations/instances_val2017.json',
            batch_size=8,
            num_workers=4,
        ),
        # COCO class names
        class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ],
    )
    
    # Training parameters
    train = dict(
        epochs=100,
        optimizer=dict(
            name='adamw',
            lr=0.00003,  # 降低学习率以提高稳定性
            weight_decay=0.0001,
        ),
        scheduler=dict(
            name='cosine',
            warmup_epochs=3,
            warmup_lr_start=0.00001,
        ),
        # Loss weights - 增加类别损失权重以更注重分类准确性
        loss_weights=dict(
            class_weight=8.0,         # 增加分类权重
            box_weight=5.0,           # 保持盒回归权重
            giou_weight=2.0,          # 保持GIoU权重
            objectness_weight=1.0,    # 添加对象存在性权重
            diversity_weight=0.1,     # 保持多样性权重
        ),
        # Checkpointing
        checkpoint_interval=5,  # Save every N epochs
        log_interval=100,  # Log every N iterations
    )
    
    # Evaluation parameters
    eval = dict(
        confidence_threshold=0.5,
        nms_threshold=0.5,
        max_detections=100,
    )
    
    # Inference parameters
    inference = dict(
        confidence_threshold=0.5,
        nms_threshold=0.5,
        max_detections=100,
    )
    
    # Paths
    paths = dict(
        output_dir='output',
        checkpoint_dir='output/checkpoints',
        log_dir='output/logs',
        visualization_dir='output/visualization',
    )
