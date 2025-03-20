import os
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
from tqdm import tqdm

from models.sad_detector import SADDetector
from utils.post_processing import post_process
from utils.visualization import visualize_detections_with_attention, save_visualization
from utils.config import create_config_from_yaml, update_config_with_yaml

from configs.default import Config as DefaultConfig

def load_image(image_path, size=(512, 512)):
    """
    Load and preprocess image
    
    Args:
        image_path (str): Path to image
        size (tuple): Image size
        
    Returns:
        torch.Tensor: Preprocessed image tensor
        PIL.Image: Original image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform image
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_image


def inference_single_image(model, image_path, device, config):
    """
    Run inference on a single image
    
    Args:
        model: SAD model
        image_path (str): Path to image
        device: Device
        config: Configuration
        
    Returns:
        dict: Inference results
    """
    # Load image
    image_tensor, original_image = load_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Post-process outputs
        processed_outputs = post_process(
            outputs,
            confidence_threshold=config.inference['confidence_threshold'],
            nms_threshold=config.inference['nms_threshold'],
            max_detections=config.inference['max_detections']
        )
    
    # Visualize results
    fig = visualize_detections_with_attention(
        image_tensor[0],
        {k: v for k, v in outputs.items()},
        class_names=config.dataset['class_names'],
        threshold=config.inference['confidence_threshold']
    )
    
    return {
        'original_image': original_image,
        'outputs': outputs,
        'processed_outputs': processed_outputs,
        'visualization': fig
    }


def inference_directory(model, input_dir, output_dir, device, config):
    """
    Run inference on all images in a directory
    
    Args:
        model: SAD model
        input_dir (str): Input directory containing images
        output_dir (str): Output directory for visualizations
        device: Device
        config: Configuration
        
    Returns:
        dict: Inference statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith(image_extensions)
    ]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return {}
    
    # Process each image
    total_time = 0
    num_images = len(image_files)
    
    progress_bar = tqdm(image_files, desc="Processing images")
    
    for image_path in progress_bar:
        # Get image filename
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_result.png")
        
        # Measure inference time
        start_time = time.time()
        
        # Run inference
        results = inference_single_image(model, image_path, device, config)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Save visualization
        save_visualization(results['visualization'], output_path)
        
        # Update progress bar
        progress_bar.set_postfix({
            'time': f"{inference_time:.3f}s",
            'output': os.path.basename(output_path)
        })
    
    # Calculate statistics
    avg_time = total_time / num_images if num_images > 0 else 0
    fps = 1 / avg_time if avg_time > 0 else 0
    
    stats = {
        'num_images': num_images,
        'total_time': total_time,
        'avg_time': avg_time,
        'fps': fps
    }
    
    return stats


def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description='SAD Inference')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, default='output/inference', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--yaml', type=str, default='coco.yaml', help='Path to YAML dataset config')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
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
    
    # Update config with command line arguments
    if args.threshold:
        config.inference['confidence_threshold'] = args.threshold
    
    # Device setup
    device_name = args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_name = 'cpu'
    
    device = torch.device(device_name)
    print(f"Using device: {device}")
    
    # Create model
    num_classes = len(config.dataset['class_names'])
    model = SADDetector(
        num_classes=num_classes,
        num_spotlights=config.model['num_spotlights']
    )
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
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
    
    # Run inference
    input_path = args.input
    output_dir = args.output
    
    if os.path.isdir(input_path):
        # Process directory
        stats = inference_directory(model, input_path, output_dir, device, config)
        
        print("\nInference statistics:")
        print(f"  Number of images: {stats['num_images']}")
        print(f"  Total time: {stats['total_time']:.3f}s")
        print(f"  Average time per image: {stats['avg_time']:.3f}s")
        print(f"  FPS: {stats['fps']:.2f}")
    else:
        # Process single image
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_result.png")
        
        # Run inference
        start_time = time.time()
        results = inference_single_image(model, input_path, device, config)
        inference_time = time.time() - start_time
        
        # Save visualization
        save_visualization(results['visualization'], output_path)
        
        print(f"\nProcessed {input_path}")
        print(f"  Inference time: {inference_time:.3f}s")
        print(f"  FPS: {1/inference_time:.2f}")
        print(f"  Results saved to {output_path}")


if __name__ == '__main__':
    main()
