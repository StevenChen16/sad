import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np

class YOLODataset(Dataset):
    """
    Dataset class for YOLO format data
    
    YOLO format:
    - Images in 'images/train/' and 'images/val/'
    - Labels in 'labels/train/' and 'labels/val/' as .txt files
    - Each label file has one line per object: class_id x_center y_center width height
    - All values normalized to [0,1]
    
    Args:
        img_dir (str): Directory containing images (e.g., 'data/coco8/images/train')
        label_dir (str): Directory containing labels (e.g., 'data/coco8/labels/train')
        class_names (list): List of class names
        transform (callable, optional): Transform to apply to images
    """
    def __init__(self, img_dir, label_dir, class_names, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.class_names = class_names
        
        # Get all image files
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + 
                               glob.glob(os.path.join(img_dir, '*.jpeg')) + 
                               glob.glob(os.path.join(img_dir, '*.png')))
        
        # Check if we found any images
        if len(self.img_files) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")
        
        print(f"Found {len(self.img_files)} images in {img_dir}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Get dataset item
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, target) where target is a dictionary containing bounding boxes and labels
        """
        # Get image path
        img_path = self.img_files[idx]
        img_name = os.path.basename(img_path)
        img_id = os.path.splitext(img_name)[0]  # Remove extension
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Get original dimensions for normalization reference
        orig_width, orig_height = img.size
        
        # Get label path
        label_path = os.path.join(self.label_dir, f"{img_id}.txt")
        
        # Initialize empty lists for boxes and labels
        boxes = []
        labels = []
        
        # Load labels if file exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        # YOLO format: class_id x_center y_center width height
                        values = line.strip().split()
                        if len(values) == 5:
                            class_id = int(values[0])
                            x_center = float(values[1])  # Already normalized
                            y_center = float(values[2])  # Already normalized
                            width = float(values[3])     # Already normalized
                            height = float(values[4])    # Already normalized
                            
                            # Basic validation checks
                            if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                0 < width <= 1 and 0 < height <= 1 and 
                                0 <= class_id < len(self.class_names)):
                                # YOLO format is already normalized and in center format
                                boxes.append([x_center, y_center, width, height])
                                labels.append(class_id)
        
        # Apply transform if provided
        if self.transform is not None:
            img = self.transform(img)
        
        # Create target dict
        num_objects = len(boxes)
        target = {
            'boxes': torch.zeros((num_objects, 4), dtype=torch.float32),
            'labels': torch.zeros(num_objects, dtype=torch.int64),
            'object_masks': torch.ones(num_objects, dtype=torch.bool),
            'image_id': torch.tensor([int(img_id) if img_id.isdigit() else hash(img_id) % 10000])
        }
        
        if num_objects > 0:
            target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.tensor(labels, dtype=torch.int64)
        
        return img, target


def build_yolo_dataset(img_dir, label_dir, class_names, train=True):
    """
    Build YOLO format dataset for training or evaluation
    
    Args:
        img_dir (str): Directory containing images
        label_dir (str): Directory containing labels
        class_names (list): List of class names
        train (bool): Whether to build dataset for training
        
    Returns:
        Dataset: Dataset for training or evaluation
    """
    # Check if directories exist
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    if not os.path.exists(label_dir):
        print(f"WARNING: Label directory not found: {label_dir}")
        print(f"Creating missing label directory: {label_dir}")
        os.makedirs(label_dir, exist_ok=True)
    
    # Create transforms
    if train:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Fixed resize first to maintain aspect ratio
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create dataset
    dataset = YOLODataset(img_dir, label_dir, class_names, transform=transform)
    return dataset


def collate_fn(batch):
    """
    Collate function for dataloader
    
    Args:
        batch (list): List of (image, target) tuples
        
    Returns:
        tuple: Batched images and targets
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    
    # Process targets
    batch_targets = {
        'boxes': [],
        'labels': [],
        'object_masks': [],
        'image_id': []
    }
    
    max_objects = max(target['boxes'].shape[0] for target in targets)
    
    for target in targets:
        num_objects = target['boxes'].shape[0]
        
        # Pad boxes
        padded_boxes = torch.zeros((max_objects, 4), dtype=torch.float32)
        padded_boxes[:num_objects] = target['boxes']
        batch_targets['boxes'].append(padded_boxes)
        
        # Pad labels
        padded_labels = torch.zeros(max_objects, dtype=torch.int64)
        padded_labels[:num_objects] = target['labels']
        batch_targets['labels'].append(padded_labels)
        
        # Pad masks
        padded_masks = torch.zeros(max_objects, dtype=torch.bool)
        padded_masks[:num_objects] = target['object_masks']
        batch_targets['object_masks'].append(padded_masks)
        
        # Image ids
        batch_targets['image_id'].append(target['image_id'])
    
    # Stack batched targets
    batch_targets['boxes'] = torch.stack(batch_targets['boxes'], dim=0)
    batch_targets['labels'] = torch.stack(batch_targets['labels'], dim=0)
    batch_targets['object_masks'] = torch.stack(batch_targets['object_masks'], dim=0)
    batch_targets['image_id'] = torch.cat(batch_targets['image_id'], dim=0)
    
    return images, batch_targets


def build_yolo_dataloader(dataset, batch_size, num_workers=4, shuffle=True):
    """
    Build dataloader for YOLO format dataset
    
    Args:
        dataset (Dataset): Dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle data
        
    Returns:
        DataLoader: DataLoader for training or evaluation
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
