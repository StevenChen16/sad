import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json

from models.sad_detector import SADDetector
from utils.dataset import build_dataset, build_dataloader
from utils.dataset_yolo import build_yolo_dataset, build_yolo_dataloader
from utils.post_processing import post_process
from utils.visualization import visualize_detections_with_attention, save_visualization
from utils.config import create_config_from_yaml, update_config_with_yaml

from configs.default import Config as DefaultConfig

def convert_to_coco_format(predictions, image_ids, class_names):
    """
    Convert predictions to COCO format for evaluation
    
    Args:
        predictions (dict): Post-processed predictions
        image_ids (list): List of image IDs
        class_names (list): List of class names
        
    Returns:
        list: Predictions in COCO format
    """
    coco_predictions = []
    
    for batch_idx, (boxes, scores, labels) in enumerate(zip(
        predictions['pred_boxes'], predictions['pred_scores'], predictions['pred_labels']
    )):
        image_id = image_ids[batch_idx].item()
        
        for box_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            # Convert box from (cx, cy, w, h) to (x, y, w, h)
            cx, cy, w, h = box.tolist()
            x = cx - w / 2
            y = cy - h / 2
            
            # Clip coordinates to [0, 1]
            x = max(0, min(1-1e-10, x))
            y = max(0, min(1-1e-10, y))
            w = max(0, min(1-x-1e-10, w))
            h = max(0, min(1-y-1e-10, h))
            
            prediction = {
                'image_id': int(image_id),
                'category_id': int(label.item()) + 1,  # COCO categories are 1-indexed
                'bbox': [x, y, w, h],
                'score': float(score.item()),
            }
            
            coco_predictions.append(prediction)
    
    return coco_predictions


def evaluate(model, dataloader, device, config):
    """
    Evaluate model on dataset
    
    Args:
        model: SAD model
        dataloader: Evaluation data loader
        device: Device to use
        config: Configuration
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_image_ids = []
    
    # Visualization directory
    vis_dir = config.paths['visualization_dir']
    os.makedirs(vis_dir, exist_ok=True)
    
    progress_bar = tqdm(dataloader, desc="Evaluation")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            image_ids = targets['image_id']
            
            # Forward pass
            outputs = model(images)
            
            # Post-process outputs
            processed_outputs = post_process(
                outputs,
                confidence_threshold=config.eval['confidence_threshold'],
                nms_threshold=config.eval['nms_threshold'],
                max_detections=config.eval['max_detections']
            )
            
            # Convert to COCO format
            batch_predictions = convert_to_coco_format(
                processed_outputs, image_ids, config.dataset['class_names']
            )
            all_predictions.extend(batch_predictions)
            all_image_ids.extend(image_ids.tolist())
            
            # Visualize some predictions
            if batch_idx < 5:  # Visualize first 5 batches
                for i in range(min(2, len(images))):  # Visualize up to 2 images per batch
                    img = images[i].cpu()
                    
                    # Prepare outputs for visualization
                    vis_outputs = {
                        k: v[i].unsqueeze(0) if i < len(v) else v[0].unsqueeze(0)
                        for k, v in outputs.items()
                    }
                    
                    # Visualize detections with attention
                    fig = visualize_detections_with_attention(
                        img,
                        vis_outputs,
                        class_names=config.dataset['class_names'],
                        threshold=config.eval['confidence_threshold']
                    )
                    
                    # Save visualization
                    image_id = image_ids[i].item()
                    save_visualization(
                        fig,
                        os.path.join(vis_dir, f"eval_img_{image_id}_batch_{batch_idx}_idx_{i}.png")
                    )
    
    # Save predictions to file
    predictions_file = os.path.join(config.paths['output_dir'], 'predictions.json')
    with open(predictions_file, 'w') as f:
        json.dump(all_predictions, f)
    
    print(f"Predictions saved to {predictions_file}")
    print(f"To evaluate on COCO dataset, use the official COCO evaluation script.")
    
    return {"num_predictions": len(all_predictions)}


def main(config):
    """
    Evaluate SAD model
    
    Args:
        config: Configuration
    """
    # Create output directories
    for path in config.paths.values():
        os.makedirs(path, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SADDetector(
        num_classes=config.model['num_classes'],
        num_spotlights=config.model['num_spotlights']
    )
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.paths['checkpoint_dir'], 'best.pth')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(config.paths['checkpoint_dir'], 'latest.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return
    
    # Create dataloader based on dataset format
    dataset_format = config.dataset.get('format', 'coco')
    
    if dataset_format == 'yolo':
        # YOLO format dataset
        val_dataset = build_yolo_dataset(
            img_dir=config.dataset['val']['img_dir'],
            label_dir=config.dataset['val']['label_dir'],
            class_names=config.dataset['class_names'],
            train=False
        )
        
        val_dataloader = build_yolo_dataloader(
            dataset=val_dataset,
            batch_size=config.dataset['val']['batch_size'],
            num_workers=config.dataset['val']['num_workers'],
            shuffle=False
        )
    else:
        # COCO format dataset
        val_dataset = build_dataset(
            img_dir=config.dataset['val']['img_dir'],
            ann_file=config.dataset['val']['ann_file'],
            train=False
        )
        
        val_dataloader = build_dataloader(
            dataset=val_dataset,
            batch_size=config.dataset['val']['batch_size'],
            num_workers=config.dataset['val']['num_workers'],
            shuffle=False
        )
    
    # Evaluate model
    metrics = evaluate(model, val_dataloader, device, config)
    
    print("\nEvaluation results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate SAD model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--yaml', type=str, default='coco.yaml', help='Path to YAML dataset config')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Load configuration
    config = DefaultConfig()
    
    # Update with YAML dataset config if provided
    if args.yaml:
        yaml_path = args.yaml
        if not os.path.isabs(yaml_path):
            yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_path)
        
        if os.path.exists(yaml_path):
            print(f"Loading dataset config from {yaml_path}")
            config = update_config_with_yaml(config, yaml_path)
        else:
            print(f"YAML config file {yaml_path} not found, using default config")
    
    # Update checkpoint path if provided
    if args.checkpoint:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        config.paths['checkpoint_dir'] = checkpoint_dir
    
    # Evaluate model
    main(config)
