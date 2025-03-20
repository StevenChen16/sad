"""
Logger utilities for SAD Detector
"""
import os
import csv
import time
import datetime
import psutil
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import json
from colorama import Fore, Style
from tqdm import tqdm


def get_system_info():
    """Get system information for logging"""
    gpu_info = ""
    gpu_mem = ""
    if torch.cuda.is_available():
        gpu_info = f"GPU {torch.cuda.get_device_name(0)}"
        gpu_mem = f"{torch.cuda.memory_reserved(0)/1E9:.3g}G"  # Memory reserved in GB
    
    cpu_info = f"CPU {psutil.cpu_percent()}%"
    ram_info = f"RAM {psutil.virtual_memory().percent}%"
    
    return f"{gpu_info} ({gpu_mem}), {cpu_info}, {ram_info}"


def get_experiment_dir(base_dir='runs/train'):
    """
    Get a unique experiment directory by incrementing exp number
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        str: Path to the experiment directory
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Find all existing exp directories
    exp_dirs = glob.glob(os.path.join(base_dir, 'exp*'))
    exp_nums = []
    
    for d in exp_dirs:
        if os.path.isdir(d):
            base_name = os.path.basename(d)
            if base_name == 'exp':
                exp_nums.append(0)  # Treat 'exp' as exp0
            elif base_name.startswith('exp'):
                # Extract the number part safely
                try:
                    num_part = base_name[3:]  # Remove 'exp' prefix
                    if num_part.isdigit():
                        exp_nums.append(int(num_part))
                except ValueError:
                    # Skip directories with non-numeric suffixes
                    continue
    
    # If no exp directories exist, use exp
    if not exp_nums:
        exp_dir = os.path.join(base_dir, 'exp')
    else:
        # Get the next exp number
        next_exp_num = max(exp_nums) + 1
        exp_dir = os.path.join(base_dir, f'exp{next_exp_num}')
    
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories for logs, plots, etc.
    os.makedirs(os.path.join(exp_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'weights'), exist_ok=True)
    
    return exp_dir


def create_results_csv(output_dir):
    """
    Create CSV file for logging training results
    
    Args:
        output_dir: Directory to save CSV file
        
    Returns:
        str: Path to CSV file
    """
    csv_path = os.path.join(output_dir, 'results.csv')
    
    # Define CSV headers
    headers = [
        'epoch', 
        'train/loss', 'train/class_loss', 'train/box_loss', 'train/giou_loss', 'train/diversity_loss',
        'val/loss', 'val/class_loss', 'val/box_loss', 'val/giou_loss', 'val/diversity_loss',
        'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
        'learning_rate', 'memory', 'time'
    ]
    
    # Create CSV file with headers
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    
    return csv_path


def log_epoch_results(csv_path, epoch, train_stats, val_stats, lr, memory_usage):
    """
    Log epoch results to CSV file
    
    Args:
        csv_path: Path to CSV file
        epoch: Current epoch
        train_stats: Training statistics
        val_stats: Validation statistics
        lr: Current learning rate
        memory_usage: GPU memory usage
    """
    with open(csv_path, 'a', newline='') as f:
        writer_csv = csv.writer(f)
        row = [
            epoch + 1,
            train_stats['loss'], train_stats['class_loss'], train_stats['box_loss'], 
            train_stats['giou_loss'], train_stats['diversity_loss'],
            val_stats['loss'], val_stats['class_loss'], val_stats['box_loss'], 
            val_stats['giou_loss'], val_stats['diversity_loss'],
            val_stats['precision'], val_stats['recall'], val_stats['mAP50'], val_stats['mAP50-95'],
            lr,
            memory_usage,
            f"{train_stats['time'] + val_stats['time']:.1f}s"
        ]
        writer_csv.writerow(row)


class TensorboardLogger:
    """Tensorboard Logger for SAD Detector"""
    
    def __init__(self, log_dir):
        """
        Initialize TensorboardLogger
        
        Args:
            log_dir: Directory for tensorboard logs
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
    
    def log_hyperparameters(self, hparams):
        """
        Log hyperparameters
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        self.writer.add_hparams(hparams, {})
    
    def log_model(self, model, input_size=(1, 3, 640, 640)):
        """
        Log model graph
        
        Args:
            model: Model to log
            input_size: Input size for model graph
        """
        try:
            device = next(model.parameters()).device
            dummy_input = torch.randn(input_size).to(device)
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Could not log model graph to TensorBoard: {e}{Style.RESET_ALL}")
    
    def log_train_batch(self, losses, epoch, batch_idx, num_batches):
        """
        Log training batch
        
        Args:
            losses: Dictionary of losses
            epoch: Current epoch
            batch_idx: Current batch index
            num_batches: Number of batches per epoch
        """
        global_step = epoch * num_batches + batch_idx
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                self.writer.add_scalar(f'train/batch/{k}', v.item(), global_step)
            else:
                self.writer.add_scalar(f'train/batch/{k}', v, global_step)
    
    def log_train_epoch(self, stats, epoch):
        """
        Log training epoch
        
        Args:
            stats: Dictionary of epoch statistics
            epoch: Current epoch
        """
        for k, v in stats.items():
            if k != 'time':
                self.writer.add_scalar(f'train/epoch/{k}', v, epoch)
    
    def log_val_epoch(self, stats, epoch):
        """
        Log validation epoch
        
        Args:
            stats: Dictionary of epoch statistics
            epoch: Current epoch
        """
        for k, v in stats.items():
            if k in ['loss', 'class_loss', 'box_loss', 'giou_loss', 'diversity_loss']:
                self.writer.add_scalar(f'val/epoch/{k}', v, epoch)
            elif k in ['precision', 'recall', 'mAP50', 'mAP50-95']:
                self.writer.add_scalar(f'metrics/{k}', v, epoch)
    
    def log_learning_rate(self, lr, epoch):
        """
        Log learning rate
        
        Args:
            lr: Learning rate
            epoch: Current epoch
        """
        self.writer.add_scalar('train/epoch/learning_rate', lr, epoch)
    
    def log_image(self, tag, img_tensor, epoch):
        """
        Log image
        
        Args:
            tag: Image tag
            img_tensor: Image tensor
            epoch: Current epoch
        """
        self.writer.add_image(tag, img_tensor, epoch, dataformats='HWC')
    
    def close(self):
        """Close writer"""
        self.writer.close()


def visualize_class_distribution(class_names, class_counts, output_dir):
    """
    Create class distribution plot
    
    Args:
        class_names: List of class names
        class_counts: Dictionary with class counts
        output_dir: Directory to save visualization
    """
    plt.figure(figsize=(12, 8))
    # Sort by count
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    class_ids, counts = zip(*sorted_items)
    
    # Replace class IDs with names
    names = [class_names[class_id] for class_id in class_ids]
    
    # Create bar plot
    bars = plt.bar(names, counts)
    plt.title('Class Distribution', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height+1,
                 f'{height:.0f}',
                 ha='center', va='bottom', fontsize=10)
    
    # Save figure
    output_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path


def visualize_boxes_per_image(boxes_per_image, output_dir):
    """
    Create boxes per image histogram
    
    Args:
        boxes_per_image: List of box counts per image
        output_dir: Directory to save visualization
    """
    plt.figure(figsize=(10, 6))
    plt.hist(boxes_per_image, bins=20, alpha=0.7, color='blue')
    plt.title('Objects per Image Distribution', fontsize=16)
    plt.xlabel('Number of Objects', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.grid(alpha=0.3)
    
    # Calculate statistics
    avg_boxes = np.mean(boxes_per_image)
    median_boxes = np.median(boxes_per_image)
    max_boxes = np.max(boxes_per_image)
    
    # Add statistics as text
    plt.text(0.95, 0.95, f'Mean: {avg_boxes:.1f}\nMedian: {median_boxes:.1f}\nMax: {max_boxes:.0f}',
             transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'boxes_per_image.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path


def visualize_image_sizes(image_sizes, output_dir):
    """
    Create image size distribution visualization
    
    Args:
        image_sizes: Dictionary of image sizes and counts
        output_dir: Directory to save visualization
    """
    # Sort by count
    sizes = []
    counts = []
    for size, count in sorted(image_sizes.items(), key=lambda x: x[1], reverse=True):
        sizes.append(size)
        counts.append(count)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sizes, counts)
    plt.title('Image Size Distribution', fontsize=16)
    plt.xlabel('Image Size (width x height)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height+1,
                 f'{height:.0f}',
                 ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'image_size_distribution.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path


def visualize_results(results_csv, output_dir):
    """
    Create training and validation results plots
    
    Args:
        results_csv: Path to results CSV file
        output_dir: Directory to save visualizations
    """
    # Read results CSV
    data = []
    with open(results_csv, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            data.append(row)
    
    if not data:
        print(f"{Fore.YELLOW}⚠ No data in results CSV file: {results_csv}{Style.RESET_ALL}")
        return
    
    # Convert to numpy arrays
    data = np.array(data)
    epochs = data[:, 0].astype(int)
    
    # Plot losses
    plt.figure(figsize=(12, 8))
    
    # Training losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs, data[:, 1].astype(float), label='Train Loss')
    plt.plot(epochs, data[:, 6].astype(float), label='Val Loss')
    plt.title('Total Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Class losses
    plt.subplot(2, 2, 2)
    plt.plot(epochs, data[:, 2].astype(float), label='Train Class Loss')
    plt.plot(epochs, data[:, 7].astype(float), label='Val Class Loss')
    plt.title('Class Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Box losses
    plt.subplot(2, 2, 3)
    plt.plot(epochs, data[:, 3].astype(float), label='Train Box Loss')
    plt.plot(epochs, data[:, 8].astype(float), label='Val Box Loss')
    plt.title('Box Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # GIoU and Diversity losses
    plt.subplot(2, 2, 4)
    plt.plot(epochs, data[:, 4].astype(float), label='Train GIoU Loss')
    plt.plot(epochs, data[:, 9].astype(float), label='Val GIoU Loss')
    plt.plot(epochs, data[:, 5].astype(float), label='Train Diversity Loss')
    plt.plot(epochs, data[:, 10].astype(float), label='Val Diversity Loss')
    plt.title('GIoU & Diversity Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, 'loss_plots.png')
    plt.savefig(loss_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot metrics
    plt.figure(figsize=(12, 10))
    
    # mAP50
    plt.subplot(2, 2, 1)
    plt.plot(epochs, data[:, 13].astype(float), 'g-')
    plt.title('mAP@0.5', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('mAP', fontsize=12)
    plt.grid(alpha=0.3)
    
    # mAP50-95
    plt.subplot(2, 2, 2)
    plt.plot(epochs, data[:, 14].astype(float), 'g-')
    plt.title('mAP@0.5:0.95', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('mAP', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Precision
    plt.subplot(2, 2, 3)
    plt.plot(epochs, data[:, 11].astype(float), 'b-')
    plt.title('Precision', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, data[:, 12].astype(float), 'b-')
    plt.title('Recall', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    metrics_plot_path = os.path.join(output_dir, 'metrics_plots.png')
    plt.savefig(metrics_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Learning rate plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data[:, 15].astype(float), 'r-')
    plt.title('Learning Rate', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    lr_plot_path = os.path.join(output_dir, 'learning_rate_plot.png')
    plt.savefig(lr_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return [loss_plot_path, metrics_plot_path, lr_plot_path]


def analyze_dataset(dataset, class_names, output_dir):
    """
    Analyze dataset and generate statistics
    
    Args:
        dataset: Dataset to analyze
        class_names: List of class names
        output_dir: Directory to save statistics
        
    Returns:
        dict: Dataset statistics
    """
    print(f"{Fore.GREEN}█ Analyzing dataset...{Style.RESET_ALL}")
    
    stats = {
        'total_images': len(dataset),
        'classes': {name: 0 for name in class_names},
        'image_sizes': {},
        'label_distribution': {i: 0 for i in range(len(class_names))},
        'boxes_per_image': [],
    }
    
    # Sample a subset of images for detailed analysis if dataset is large
    sample_size = min(1000, len(dataset))
    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in tqdm(sample_indices, desc=f"{Fore.BLUE}Dataset Analysis{Style.RESET_ALL}"):
        try:
            # Get image and targets
            _, targets = dataset[idx]
            
            if 'labels' in targets:
                labels = targets['labels']
                
                # Count labels
                for label in labels:
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    if 0 <= label < len(class_names):
                        stats['label_distribution'][label] += 1
                        stats['classes'][class_names[label]] += 1
                
                # Count boxes per image
                stats['boxes_per_image'].append(len(labels))
            
            if 'img_size' in targets:
                img_size = targets['img_size']
                size_key = f"{img_size[0]}x{img_size[1]}"
                stats['image_sizes'][size_key] = stats['image_sizes'].get(size_key, 0) + 1
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Error analyzing dataset item {idx}: {e}{Style.RESET_ALL}")
    
    # Calculate average boxes per image
    if stats['boxes_per_image']:
        stats['avg_boxes_per_image'] = sum(stats['boxes_per_image']) / len(stats['boxes_per_image'])
    else:
        stats['avg_boxes_per_image'] = 0
        
    # Calculate total boxes
    stats['total_boxes'] = sum(stats['label_distribution'].values())
    
    # Save statistics to JSON file
    stats_file = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Create class distribution plot data
    class_dist_file = os.path.join(output_dir, 'labels', 'class_distribution.txt')
    with open(class_dist_file, 'w') as f:
        for i, name in enumerate(class_names):
            count = stats['label_distribution'][i]
            f.write(f"{i}\t{name}\t{count}\n")
    
    # Create boxes per image distribution file
    boxes_dist_file = os.path.join(output_dir, 'labels', 'boxes_per_image.txt')
    with open(boxes_dist_file, 'w') as f:
        for count in stats['boxes_per_image']:
            f.write(f"{count}\n")
    
    # Generate visualizations
    if stats['label_distribution']:
        visualize_class_distribution(class_names, stats['label_distribution'], os.path.join(output_dir, 'labels'))
    
    if stats['boxes_per_image']:
        visualize_boxes_per_image(stats['boxes_per_image'], os.path.join(output_dir, 'labels'))
    
    if stats['image_sizes']:
        visualize_image_sizes(stats['image_sizes'], os.path.join(output_dir, 'labels'))
    
    print(f"{Fore.GREEN}▶ Dataset statistics saved to {stats_file}{Style.RESET_ALL}")
    return stats