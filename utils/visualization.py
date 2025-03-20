import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
import cv2
import torchvision.transforms as T

def plot_results(img, boxes, scores, labels, class_names=None, threshold=0.5, figsize=(12, 12)):
    """
    Plot detection results on an image
    
    Args:
        img: PIL Image or torch.Tensor
        boxes (torch.Tensor): Bounding boxes in (cx, cy, w, h) format, values in [0, 1]
        scores (torch.Tensor): Confidence scores
        labels (torch.Tensor): Class indices
        class_names (list, optional): List of class names
        threshold (float): Confidence threshold for displaying detections
        figsize (tuple): Figure size
    """
    # Convert torch tensor to PIL Image if needed
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:  # Remove batch dimension
            img = img.squeeze(0)
        # Denormalize if needed
        if img.max() <= 1.0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
        # Convert tensor to PIL Image
        img = T.ToPILImage()(img)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    
    # Filter detections by confidence threshold
    mask = scores > threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
    img_width, img_height = img.size
    
    # Plot each box
    for box, score, label in zip(boxes, scores, labels):
        # Convert from [0,1] to pixel coordinates
        cx, cy, w, h = box.tolist()
        cx *= img_width
        cy *= img_height
        w *= img_width
        h *= img_height
        
        # Convert to (x1, y1, w, h) format for Rectangle patch
        x1 = cx - w/2
        y1 = cy - h/2
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        class_name = class_names[label] if class_names else f"Class {label}"
        plt.text(
            x1, y1, f"{class_name} {score:.2f}", 
            bbox=dict(facecolor='yellow', alpha=0.5),
            fontsize=12
        )
    
    plt.axis('off')
    return fig


def visualize_attention_maps(img, attention_maps, num_maps=9, figsize=(15, 10)):
    """
    Visualize attention maps from the SAD model
    
    Args:
        img: PIL Image or torch.Tensor
        attention_maps (torch.Tensor): Attention maps of shape (num_spotlights, H, W)
        num_maps (int): Number of maps to visualize
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: Figure with attention maps
    """
    # Convert torch tensor to PIL Image if needed
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:  # Remove batch dimension
            img = img.squeeze(0)
        # Denormalize if needed
        if img.max() <= 1.0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
        # Convert tensor to PIL Image
        img = T.ToPILImage()(img)
    
    # Create figure with subplots
    n_cols = min(3, num_maps)
    n_rows = (num_maps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes if needed
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    if n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    
    # Plot original image in the first subplot
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Plot attention maps
    for i in range(1, min(num_maps, len(attention_maps))):
        ax = axes[i]
        attention_map = attention_maps[i].cpu().numpy()
        
        # Display the original image
        ax.imshow(np.array(img))
        
        # Overlay attention map with transparency
        ax.imshow(attention_map, alpha=0.7, cmap='hot')
        ax.set_title(f"Spotlight {i+1}")
        ax.axis('off')
    
    # Hide any unused subplots
    for i in range(min(num_maps, len(attention_maps)), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_detections_with_attention(img, outputs, class_names=None, threshold=0.5, 
                                       max_attention_maps=5, figsize=(18, 10)):
    """
    Visualize detection results with attention maps
    
    Args:
        img: PIL Image or torch.Tensor
        outputs (dict): Model outputs with keys:
            - pred_boxes (torch.Tensor): Predicted boxes
            - pred_logits (torch.Tensor): Class logits
            - attention_maps (torch.Tensor): Attention maps
        class_names (list, optional): List of class names
        threshold (float): Confidence threshold
        max_attention_maps (int): Maximum number of attention maps to show
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: Figure with detections and attention maps
    """
    # Convert torch tensor to PIL Image if needed
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:  # Remove batch dimension
            img = img.squeeze(0)
        # Denormalize if needed
        if img.max() <= 1.0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
        # Convert tensor to PIL Image
        img_pil = T.ToPILImage()(img)
    else:
        img_pil = img
    
    # Get predictions
    # Handle different tensor shapes
    pred_boxes = outputs.get('pred_boxes', None)
    pred_logits = outputs.get('pred_logits', None)
    attention_maps = outputs.get('attention_maps', None)
    
    if pred_boxes is None or pred_logits is None:
        # If necessary outputs are missing, just return the image
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(img_pil)
        ax.set_title("Image (No detections found)")
        ax.axis('off')
        return fig
    
    # Ensure we have batch dimension
    if pred_boxes.dim() == 2:
        pred_boxes = pred_boxes.unsqueeze(0)
    
    if pred_logits.dim() == 2:
        pred_logits = pred_logits.unsqueeze(0)
        
    # Extract predictions from the first batch
    pred_boxes = pred_boxes[0]
    pred_logits = pred_logits[0]
    
    # Get scores and labels
    if pred_logits.size(1) == 1:  # Binary case (objectness only)
        pred_scores = pred_logits.sigmoid().squeeze(-1)
        pred_labels = torch.zeros_like(pred_scores, dtype=torch.long)
    else:
        # Multi-class case
        pred_scores, pred_labels = pred_logits.softmax(-1).max(-1)
    
    # Get attention maps if available
    if attention_maps is not None and attention_maps.dim() > 1:
        attention_maps = attention_maps[0] if attention_maps.dim() > 3 else attention_maps
    
    # Filter by confidence threshold
    mask = pred_scores > threshold
    filtered_boxes = pred_boxes[mask]
    filtered_scores = pred_scores[mask]
    filtered_labels = pred_labels[mask]
    
    # Create figure with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot detections on the left
    ax1.imshow(img_pil)
    for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
        cx, cy, w, h = box.tolist()
        img_width, img_height = img_pil.size
        
        # Convert normalized coordinates to pixel coordinates
        cx *= img_width
        cy *= img_height
        w *= img_width
        h *= img_height
        
        # Convert to top-left corner for Rectangle
        x1 = cx - w/2
        y1 = cy - h/2
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
        
        class_name = class_names[label] if class_names else f"Class {label}"
        ax1.text(
            x1, y1, f"{class_name} {score:.2f}", 
            bbox=dict(facecolor='yellow', alpha=0.5),
            fontsize=10
        )
    
    ax1.set_title("Detections")
    ax1.axis('off')
    
    # Plot attention maps on the right
    # FIX: Modified to avoid using subplot_mosaic which is not available in all matplotlib versions
    if attention_maps is not None and mask.sum() > 0:
        # If we have detections and attention maps, show them
        filtered_attention_idx = mask.nonzero(as_tuple=True)[0]
        top_attention_idx = filtered_attention_idx[:min(max_attention_maps, len(filtered_attention_idx))]
        
        if len(top_attention_idx) == 1:
            # Just one attention map to show
            attention_map = attention_maps[top_attention_idx[0]].cpu().numpy()
            ax2.imshow(img_pil)
            ax2.imshow(attention_map, alpha=0.7, cmap='hot')
            ax2.set_title(f"Attention Map (Top Detection)")
            ax2.axis('off')
        elif len(top_attention_idx) > 1:
            # Multiple attention maps to show - use nested gridspec
            # First clear the existing axis
            ax2.clear()
            ax2.axis('off')
            # Set overall title
            ax2.set_title("Top Attention Maps")
            
            # Create a grid of attention maps as subplots directly in the figure
            n_cols = min(2, len(top_attention_idx))
            n_rows = (len(top_attention_idx) + n_cols - 1) // n_cols
            
            # Calculate position for grid in ax2
            pos = ax2.get_position()
            
            # Create a gridspec inside ax2's position
            gs = fig.add_gridspec(n_rows, n_cols, 
                                 left=pos.x0, right=pos.x1,
                                 bottom=pos.y0, top=pos.y1)
            
            # Create each attention map in its own subplot
            for i, idx in enumerate(top_attention_idx):
                row = i // n_cols
                col = i % n_cols
                sub_ax = fig.add_subplot(gs[row, col])
                
                # Show attention map
                attention_map = attention_maps[idx].cpu().numpy()
                sub_ax.imshow(img_pil)
                sub_ax.imshow(attention_map, alpha=0.7, cmap='hot')
                sub_ax.set_title(f"Attn {i+1}")
                sub_ax.axis('off')
    else:
        # No detections or attention maps - show a message
        ax2.text(0.5, 0.5, "No attention maps available\nor no detections above threshold", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes)
        ax2.axis('off')
        ax2.set_title("Attention Maps")
    
    plt.tight_layout()
    return fig


def save_visualization(fig, output_path):
    """
    Save visualization figure to file
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save
        output_path (str): Path to save the figure
    """
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Visualization saved to {output_path}")
