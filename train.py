import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import time
import datetime
import numpy as np
import sys
import colorama
from colorama import Fore, Style

from models.sad_detector import SADDetector
from utils.sad_loss import SADLoss
from utils.dataset import build_dataset, build_dataloader
from utils.dataset_yolo import build_yolo_dataset, build_yolo_dataloader
from utils.post_processing import post_process
from utils.visualization import visualize_detections_with_attention, save_visualization
from utils.config import create_config_from_yaml, update_config_with_yaml
from utils.logger import TensorboardLogger, analyze_dataset, create_results_csv, log_epoch_results, get_experiment_dir, get_system_info, visualize_results
from utils.metrics import evaluate_detections

from configs.default import Config as DefaultConfig

# Initialize colorama for colored terminal output
colorama.init()


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, config, logger=None):
    """
    Train model for one epoch
    
    Args:
        model: SAD model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        epoch: Current epoch
        config: Configuration
        logger: TensorboardLogger instance
    
    Returns:
        dict: Training metrics
    """
    model.train()
    
    loss_stats = {
        'loss': 0.0,
        'class_loss': 0.0,
        'box_loss': 0.0,
        'giou_loss': 0.0,
        'objectness_loss': 0.0,
        'diversity_loss': 0.0,
    }
    
    num_batches = len(dataloader)
    
    # Start timing
    start_time = time.time()
    
    # Create progress bar with ultralytics-style formatting
    progress_bar = tqdm(dataloader, 
                         desc=f"{Fore.BLUE}Train{Style.RESET_ALL} Epoch {epoch+1}/{config.train['epochs']}", 
                         bar_format="{l_bar}{bar:10}{r_bar}",
                         unit=" batch")
    
    batch_times = []
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        batch_start = time.time()
        
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # Forward pass
        outputs = model(images)
        
        # 监控类别预测分布，检查是否有偏向特定类别的问题
        if batch_idx % 100 == 0:
            with torch.no_grad():
                # 计算类别预测分布
                class_preds = F.softmax(outputs['pred_logits'], dim=-1)
                class_dist = class_preds.mean(dim=(0, 1))  # 平均每个类别的预测概率
                top_classes = torch.topk(class_dist, 5)
                print(f"\n{Fore.CYAN}类别统计: 前5个预测类别: {top_classes.indices.cpu().numpy()}, "
                      f"概率: {top_classes.values.cpu().numpy()}{Style.RESET_ALL}")
        
        # Compute losses
        losses = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()
        
        # Update loss stats
        for k, v in losses.items():
            # Check if the loss value is a tensor or already a float
            if isinstance(v, torch.Tensor):
                loss_stats[k] += v.item()
            else:
                # If it's already a float, add it directly
                loss_stats[k] += v
        
        # Calculate batch time and update list of batch times
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Calculate ETA
        if len(batch_times) > 0:
            avg_batch_time = np.mean(batch_times[-min(len(batch_times), 10):])  # Average of last 10 batches
            eta = avg_batch_time * (num_batches - (batch_idx + 1))
            eta_str = str(datetime.timedelta(seconds=int(eta)))
        else:
            eta_str = "?"
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update progress bar with ultralytics-style information
        progress_bar.set_postfix({
            'loss': f"{losses['loss'].item() if isinstance(losses['loss'], torch.Tensor) else losses['loss']:.3f}",
            'cls': f"{losses['class_loss'] if isinstance(losses['class_loss'], float) else losses['class_loss'].item():.3f}",
            'box': f"{losses['box_loss'] if isinstance(losses['box_loss'], float) else losses['box_loss'].item():.3f}",
            'lr': f"{current_lr:.6f}",
            'img_size': f"{images.shape[2]}x{images.shape[3]}",
            'GPU': f"{torch.cuda.memory_reserved(0)/1E9:.1f}G" if torch.cuda.is_available() else "N/A",
            'ETA': eta_str
        })
        
        # Log to TensorBoard (per batch)
        if logger:
            logger.log_train_batch(losses, epoch, batch_idx, num_batches)
    
    # Average loss stats
    for k in loss_stats:
        loss_stats[k] /= num_batches
    
    # Log to TensorBoard (per epoch)
    if logger:
        logger.log_train_epoch(loss_stats, epoch)
    
    # Calculate total epoch time
    epoch_time = time.time() - start_time
    
    # Print epoch summary
    print(f"\n{Fore.GREEN}█ Epoch {epoch+1} Training Summary {Style.RESET_ALL}")
    print(f"  Time: {epoch_time:.2f}s ({epoch_time / num_batches:.2f}s/batch)")
    print(f"  Loss: {loss_stats['loss']:.4f}, Class Loss: {loss_stats['class_loss']:.4f}, Box Loss: {loss_stats['box_loss']:.4f}")
    print(f"  GIoU Loss: {loss_stats['giou_loss']:.4f}, Objectness Loss: {loss_stats['objectness_loss']:.4f}, Diversity Loss: {loss_stats['diversity_loss']:.4f}")
    
    loss_stats['time'] = epoch_time
    
    return loss_stats


def validate(model, dataloader, criterion, device, config, epoch, logger=None):
    """
    Validate model
    
    Args:
        model: SAD model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use
        config: Configuration
        epoch: Current epoch number for logging
        logger: TensorboardLogger instance
    
    Returns:
        dict: Validation metrics
    """
    model.eval()
    
    loss_stats = {
        'loss': 0.0,
        'class_loss': 0.0,
        'box_loss': 0.0,
        'giou_loss': 0.0,
        'objectness_loss': 0.0,
        'diversity_loss': 0.0,
    }
    
    num_batches = len(dataloader)
    
    # Create progress bar with ultralytics-style formatting
    progress_bar = tqdm(dataloader, 
                         desc=f"{Fore.BLUE}Val{Style.RESET_ALL} Epoch {epoch+1}/{config.train['epochs']}", 
                         bar_format="{l_bar}{bar:10}{r_bar}",
                         unit=" batch")
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            outputs = model(images)
            
            # Compute losses
            losses = criterion(outputs, targets)
            
            # Update loss stats
            for k, v in losses.items():
                # Check if the loss value is a tensor or already a float
                if isinstance(v, torch.Tensor):
                    loss_stats[k] += v.item()
                else:
                    # If it's already a float, add it directly
                    loss_stats[k] += v
            
            # Update progress bar with ultralytics-style information
            progress_bar.set_postfix({
                'loss': f"{losses['loss'].item() if isinstance(losses['loss'], torch.Tensor) else losses['loss']:.3f}",
                'cls': f"{losses['class_loss'] if isinstance(losses['class_loss'], float) else losses['class_loss'].item():.3f}",
                'box': f"{losses['box_loss'] if isinstance(losses['box_loss'], float) else losses['box_loss'].item():.3f}",
                'img_size': f"{images.shape[2]}x{images.shape[3]}",
                'GPU': f"{torch.cuda.memory_reserved(0)/1E9:.1f}G" if torch.cuda.is_available() else "N/A"
            })
            
            # Save some visualizations
            if batch_idx == 0 and epoch % config.train['checkpoint_interval'] == 0:
                vis_dir = config.paths['visualization_dir']
                os.makedirs(vis_dir, exist_ok=True)
                
                # Post-process outputs
                processed_outputs = post_process(
                    outputs,
                    confidence_threshold=config.eval['confidence_threshold'],
                    nms_threshold=config.eval['nms_threshold'],
                    max_detections=config.eval['max_detections']
                )
                
                # Visualize first image
                img = images[0].cpu()
                
                # Handle different tensor shapes correctly
                vis_outputs = {}
                for k, v in outputs.items():
                    # Skip non-tensor values
                    if not isinstance(v, torch.Tensor):
                        continue
                        
                    # Handle different tensor dimensions
                    if v.dim() == 0:  # Scalar tensor
                        vis_outputs[k] = v.unsqueeze(0)  # Add batch dimension
                    elif v.dim() == 1:  # 1D tensor
                        vis_outputs[k] = v.unsqueeze(0)  # Add batch dimension
                    else:  # Already has batch dimension
                        # Take first batch item if it exists
                        if v.size(0) > 0:
                            vis_outputs[k] = v[0].unsqueeze(0)
                        else:
                            # Empty tensor, skip
                            continue
                
                # Only visualize if we have necessary outputs
                if 'pred_boxes' in vis_outputs and 'pred_logits' in vis_outputs:
                    fig = visualize_detections_with_attention(
                        img,
                        vis_outputs,
                        class_names=config.dataset['class_names'],
                        threshold=config.eval['confidence_threshold']
                    )
                    
                    vis_path = os.path.join(vis_dir, f"val_epoch_{epoch+1}_batch_{batch_idx}.png")
                    save_visualization(fig, vis_path)
                    
                    # Log visualization to TensorBoard
                    if logger:
                        img_tensor = torch.from_numpy(np.array(fig.canvas.renderer.buffer_rgba()))
                        logger.log_image(f'val/detection_visualization', img_tensor, epoch)
                    
                    print(f"{Fore.GREEN}▶ Visualization saved to {vis_path}{Style.RESET_ALL}")
    
    # Average loss stats
    for k in loss_stats:
        loss_stats[k] /= num_batches
    
    # Calculate mAP metrics
    print(f"{Fore.BLUE}▶ Calculating mAP metrics...{Style.RESET_ALL}")
    
    # Evaluate model on validation set
    map_metrics = evaluate_detections(model, dataloader, device, config, epoch)
    
    # Log to TensorBoard (per epoch)
    if logger:
        logger.log_val_epoch({**loss_stats, **map_metrics}, epoch)
    
    # Calculate validation time
    val_time = time.time() - start_time
    
    # Print validation summary
    print(f"\n{Fore.GREEN}█ Epoch {epoch+1} Validation Summary {Style.RESET_ALL}")
    print(f"  Time: {val_time:.2f}s ({val_time / num_batches:.2f}s/batch)")
    print(f"  Loss: {loss_stats['loss']:.4f}, Class Loss: {loss_stats['class_loss']:.4f}, Box Loss: {loss_stats['box_loss']:.4f}")
    print(f"  GIoU Loss: {loss_stats['giou_loss']:.4f}, Objectness Loss: {loss_stats['objectness_loss']:.4f}, Diversity Loss: {loss_stats['diversity_loss']:.4f}")
    print(f"  Precision: {map_metrics['precision']:.4f}, Recall: {map_metrics['recall']:.4f}")
    print(f"  mAP@0.5: {map_metrics['mAP50']:.4f}, mAP@0.5:0.95: {map_metrics['mAP50-95']:.4f}")
    
    # Merge loss stats and mAP metrics
    combined_stats = {**loss_stats, **map_metrics}
    combined_stats['time'] = val_time
    
    return combined_stats


def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    """
    Save checkpoint
    
    Args:
        model: Model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        filename: Checkpoint filename
    """
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
    }
    torch.save(state, filename)
    print(f"{Fore.GREEN}▶ Checkpoint saved to {filename}{Style.RESET_ALL}")


def load_checkpoint(model, optimizer, scheduler, filename):
    """
    Load checkpoint
    
    Args:
        model: Model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        filename: Checkpoint filename
        
    Returns:
        int: Last epoch
    """
    if not os.path.isfile(filename):
        print(f"{Fore.YELLOW}⚠ Checkpoint {filename} not found. Starting from scratch.{Style.RESET_ALL}")
        return 0
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler and checkpoint['scheduler']:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    print(f"{Fore.GREEN}▶ Loaded checkpoint from epoch {checkpoint['epoch']}{Style.RESET_ALL}")
    return checkpoint['epoch']


def train(config):
    """
    Train SAD model
    
    Args:
        config: Configuration
    """
    # Create output directories
    for path in config.paths.values():
        os.makedirs(path, exist_ok=True)
    
    # Initialize TensorBoard and experiment directory
    tb_dir = get_experiment_dir()
    logger = TensorboardLogger(tb_dir)
    print(f"{Fore.GREEN}▶ TensorBoard logging to {tb_dir}{Style.RESET_ALL}")
    
    # Create CSV file for logging epoch results
    csv_path = create_results_csv(tb_dir)
    print(f"{Fore.GREEN}▶ Results will be logged to {csv_path}{Style.RESET_ALL}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{Fore.GREEN}█ Using device: {device}{Style.RESET_ALL}")
    
    # Print system information
    print(f"{Fore.GREEN}█ System: {get_system_info()}{Style.RESET_ALL}")
    
    # Create model
    num_classes = len(config.dataset['class_names'])
    print(f"{Fore.GREEN}█ Training with {num_classes} classes: {config.dataset['class_names']}{Style.RESET_ALL}")
    
    model = SADDetector(
        num_classes=num_classes,
        num_spotlights=config.model['num_spotlights']
    )
    model = model.to(device)
    
    # Log model to TensorBoard
    logger.log_model(model)
    
    # Create criterion
    criterion = SADLoss(
        num_classes=num_classes,
        **config.train['loss_weights']
    )
    
    # Create optimizer
    if config.train['optimizer']['name'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.train['optimizer']['lr'],
            weight_decay=config.train['optimizer']['weight_decay']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.train['optimizer']['lr'],
            weight_decay=config.train['optimizer']['weight_decay']
        )
    
    # Create learning rate scheduler
    scheduler = None
    if config.train['scheduler']['name'].lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train['epochs'],
            eta_min=config.train['scheduler']['warmup_lr_start']
        )
    
    # Log hyperparameters to TensorBoard
    hparams = {
        'model/num_classes': num_classes,
        'model/num_spotlights': config.model['num_spotlights'],
        'optimizer/name': config.train['optimizer']['name'],
        'optimizer/lr': config.train['optimizer']['lr'],
        'optimizer/weight_decay': config.train['optimizer']['weight_decay'],
        'scheduler/name': config.train['scheduler']['name'] if config.train['scheduler']['name'] else 'none',
        'batch_size/train': config.dataset['train']['batch_size'],
        'batch_size/val': config.dataset['val']['batch_size'],
    }
    # Add loss weights to hyperparameters
    for k, v in config.train['loss_weights'].items():
        hparams[f'loss_weights/{k}'] = v
    
    logger.log_hyperparameters(hparams)
    
    # Create dataloaders based on dataset format
    dataset_format = config.dataset.get('format', 'coco')
    print(f"{Fore.GREEN}█ Using dataset format: {dataset_format}{Style.RESET_ALL}")
    
    if dataset_format == 'yolo':
        # YOLO format dataset
        print(f"{Fore.BLUE}▶ Loading YOLO format dataset from: {config.dataset['train']['img_dir']}{Style.RESET_ALL}")
        train_dataset = build_yolo_dataset(
            img_dir=config.dataset['train']['img_dir'],
            label_dir=config.dataset['train']['label_dir'],
            class_names=config.dataset['class_names'],
            train=True
        )
        
        val_dataset = build_yolo_dataset(
            img_dir=config.dataset['val']['img_dir'],
            label_dir=config.dataset['val']['label_dir'],
            class_names=config.dataset['class_names'],
            train=False
        )
        
        train_dataloader = build_yolo_dataloader(
            dataset=train_dataset,
            batch_size=config.dataset['train']['batch_size'],
            num_workers=config.dataset['train']['num_workers'],
            shuffle=True
        )
        
        val_dataloader = build_yolo_dataloader(
            dataset=val_dataset,
            batch_size=config.dataset['val']['batch_size'],
            num_workers=config.dataset['val']['num_workers'],
            shuffle=False
        )
    else:
        # COCO format dataset
        print(f"{Fore.BLUE}▶ Loading COCO format dataset from: {config.dataset['train']['img_dir']}{Style.RESET_ALL}")
        train_dataset = build_dataset(
            img_dir=config.dataset['train']['img_dir'],
            ann_file=config.dataset['train']['ann_file'],
            train=True
        )
        
        val_dataset = build_dataset(
            img_dir=config.dataset['val']['img_dir'],
            ann_file=config.dataset['val']['ann_file'],
            train=False
        )
        
        train_dataloader = build_dataloader(
            dataset=train_dataset,
            batch_size=config.dataset['train']['batch_size'],
            num_workers=config.dataset['train']['num_workers'],
            shuffle=True
        )
        
        val_dataloader = build_dataloader(
            dataset=val_dataset,
            batch_size=config.dataset['val']['batch_size'],
            num_workers=config.dataset['val']['num_workers'],
            shuffle=False
        )
    
    # Analyze dataset and save statistics
    train_stats = analyze_dataset(train_dataset, config.dataset['class_names'], tb_dir)
    
    # Log dataset information
    print(f"{Fore.BLUE}▶ Train dataset size: {len(train_dataset)} images, {train_stats['total_boxes']} labels{Style.RESET_ALL}")
    print(f"{Fore.BLUE}▶ Val dataset size: {len(val_dataset)} images{Style.RESET_ALL}")
    
    # Get latest checkpoint
    checkpoint_dir = config.paths['checkpoint_dir']
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest.pth')
    best_checkpoint = os.path.join(checkpoint_dir, 'best.pth')
    
    # Resume training from checkpoint if exists
    start_epoch = 0
    # if os.path.exists(latest_checkpoint):
    #     start_epoch = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)
    
    # Print total training information
    print(f"\n{Fore.GREEN}█ Training Configuration {Style.RESET_ALL}")
    print(f"  Start Epoch: {start_epoch + 1}")
    print(f"  Total Epochs: {config.train['epochs']}")
    print(f"  Checkpoint Interval: {config.train['checkpoint_interval']}")
    print(f"  Train Batch Size: {config.dataset['train']['batch_size']}")
    print(f"  Val Batch Size: {config.dataset['val']['batch_size']}")
    print(f"  Train Batches per Epoch: {len(train_dataloader)}")
    print(f"  Val Batches per Epoch: {len(val_dataloader)}")
    
    # Copy model weights to experiment directory
    weights_dir = os.path.join(tb_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_start_time = time.time()
    
    for epoch in range(start_epoch, config.train['epochs']):
        print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}█ Epoch {epoch+1}/{config.train['epochs']} {Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
        
        # Train for one epoch
        train_stats = train_one_epoch(
            model, train_dataloader, optimizer, criterion, device, epoch, config, logger
        )
        
        # Validate
        val_stats = validate(
            model, val_dataloader, criterion, device, config, epoch, logger
        )
        
        # Update learning rate
        if scheduler:
            scheduler.step()
            # Log learning rate
            logger.log_learning_rate(optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if (epoch + 1) % config.train['checkpoint_interval'] == 0:
            epoch_checkpoint = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, epoch_checkpoint)
            
            # Also save to experiment weights directory
            exp_checkpoint = os.path.join(weights_dir, f"epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, exp_checkpoint)
        
        # Save latest checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, latest_checkpoint)
        
        # Save best checkpoint
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            save_checkpoint(model, optimizer, scheduler, epoch, best_checkpoint)
            
            # Also save to experiment weights directory
            best_exp_checkpoint = os.path.join(weights_dir, "best.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, best_exp_checkpoint)
            
            print(f"{Fore.GREEN}✓ New best model with validation loss: {best_val_loss:.4f}{Style.RESET_ALL}")
        
        # Log epoch results to CSV
        memory_usage = f"{torch.cuda.memory_reserved(0)/1E9:.1f}G" if torch.cuda.is_available() else "N/A"
        log_epoch_results(
            csv_path, 
            epoch, 
            train_stats, 
            val_stats, 
            optimizer.param_groups[0]['lr'],
            memory_usage
        )
        
        # Calculate ETA for remaining epochs
        elapsed_time = time.time() - train_start_time
        epochs_done = epoch - start_epoch + 1
        epochs_remaining = config.train['epochs'] - epoch - 1
        
        if epochs_done > 0:
            time_per_epoch = elapsed_time / epochs_done
            eta = time_per_epoch * epochs_remaining
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            
            print(f"\n{Fore.GREEN}█ Progress {Style.RESET_ALL}")
            print(f"  Elapsed: {str(datetime.timedelta(seconds=int(elapsed_time)))}")
            print(f"  ETA: {eta_str}")
            print(f"  Progress: {100 * (epoch + 1) / config.train['epochs']:.1f}%")
        
        # Generate result plots every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config.train['epochs']:
            visualize_results(csv_path, tb_dir)
    
    # Training finished
    total_time = time.time() - train_start_time
    
    print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}█ Training Completed {Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    print(f"  Total Time: {str(datetime.timedelta(seconds=int(total_time)))}")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    print(f"  Results saved to: {csv_path}")
    print(f"  TensorBoard logs saved to: {tb_dir}")
    print(f"  Use 'tensorboard --logdir={os.path.dirname(tb_dir)}' to view training logs")
    
    # Generate final result plots
    visualize_results(csv_path, tb_dir)
    
    # Close TensorBoard writer
    logger.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train SAD model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--yaml', type=str, default='coco.yaml', help='Path to YAML dataset config')
    return parser.parse_args()


if __name__ == '__main__':
    # Start timing total script execution
    script_start_time = time.time()
    
    args = parse_args()
    
    # Print start banner
    print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}█ SAD Detector Training {Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    
    # Print version and environment info
    print(f"{Fore.BLUE}▶ PyTorch: {torch.__version__}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}▶ Python: {sys.version.split()[0]}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}▶ CUDA available: {torch.cuda.is_available()}{Style.RESET_ALL}")
    if torch.cuda.is_available():
        print(f"{Fore.BLUE}▶ CUDA device: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")
    
    # Load configuration
    config = DefaultConfig()
    
    # Update with YAML dataset config if provided
    if args.yaml:
        yaml_path = args.yaml
        if not os.path.isabs(yaml_path):
            yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_path)
        
        if os.path.exists(yaml_path):
            print(f"{Fore.GREEN}▶ Loading dataset config from {yaml_path}{Style.RESET_ALL}")
            config = update_config_with_yaml(config, yaml_path)
        else:
            print(f"{Fore.YELLOW}⚠ YAML config file {yaml_path} not found, using default config{Style.RESET_ALL}")
    
    # Train model
    try:
        train(config)
        
        # Print total execution time
        total_execution_time = time.time() - script_start_time
        print(f"\n{Fore.GREEN}▶ Total execution time: {str(datetime.timedelta(seconds=int(total_execution_time)))}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}▶ Training completed successfully!{Style.RESET_ALL}")
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠ Training interrupted by user.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.YELLOW}⚠ Training failed with error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()