import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import numpy as np
from pycocotools.coco import COCO

class SADDataset(Dataset):
    """
    Dataset class for Spotlight Attention Detection
    
    Support standard detection datasets in COCO format.
    
    Args:
        img_dir (str): Directory containing images
        ann_file (str): Path to annotation file in COCO format
        transform (callable, optional): Transform to apply to images
    """
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Check if annotation file exists
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        
        # Load annotations using COCO API
        self.coco = COCO(ann_file)
        
        # Get image ids with annotations
        self.img_ids = list(sorted(self.coco.getImgIds()))
        
        # Filter out images without annotations if necessary
        if len(self.img_ids) == 0:
            raise ValueError(f"No valid images found in {ann_file}")
        
        print(f"Loaded {len(self.img_ids)} images from {ann_file}")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """
        Get dataset item
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Contains image tensor and target annotations
        """
        # Get image info
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            # Try alternate path formats
            # Sometimes COCO images are in subdirectories
            base_name = os.path.basename(img_info['file_name'])
            alternate_path = os.path.join(self.img_dir, base_name)
            
            if os.path.exists(alternate_path):
                img_path = alternate_path
            else:
                # Try with COCO2017 format (COCO uses either file_name or uses id as filename)
                coco_style_name = f"{img_id:012d}.jpg"
                coco_style_path = os.path.join(self.img_dir, coco_style_name)
                
                if os.path.exists(coco_style_path):
                    img_path = coco_style_path
                else:
                    raise FileNotFoundError(f"Image not found: {img_path} or {alternate_path} or {coco_style_path}")
        
        img = Image.open(img_path).convert('RGB')
        
        # Get original dimensions
        orig_width, orig_height = img.size
        
        # Apply transform if provided
        if self.transform is not None:
            img = self.transform(img)
        
        # Get image dimensions after transform
        if isinstance(img, torch.Tensor):
            _, height, width = img.shape
        else:
            width, height = img.size
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Process annotations
        boxes = []
        labels = []
        
        for ann in anns:
            # Skip annotations without bounding boxes
            if 'bbox' not in ann or len(ann['bbox']) != 4:
                continue
            
            # Get bounding box
            x, y, w, h = ann['bbox']  # COCO format: [x, y, w, h]
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            # Convert to [cx, cy, w, h] and normalize
            cx = (x + w / 2) / orig_width
            cy = (y + h / 2) / orig_height
            nw = w / orig_width
            nh = h / orig_height
            
            boxes.append([cx, cy, nw, nh])
            labels.append(ann['category_id'] - 1)  # COCO categories are 1-indexed, convert to 0-indexed
        
        # Create target dict
        num_objects = len(boxes)
        target = {
            'boxes': torch.zeros((num_objects, 4), dtype=torch.float32),
            'labels': torch.zeros(num_objects, dtype=torch.int64),
            'object_masks': torch.ones(num_objects, dtype=torch.bool),
            'image_id': torch.tensor([img_id])
        }
        
        if num_objects > 0:
            target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.tensor(labels, dtype=torch.int64)
        
        return img, target


def build_dataset(img_dir, ann_file, train=True):
    """
    Build dataset for training or evaluation
    
    Args:
        img_dir (str): Directory containing images
        ann_file (str): Path to annotation file
        train (bool): Whether to build dataset for training
        
    Returns:
        Dataset: Dataset for training or evaluation
    """
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = SADDataset(img_dir, ann_file, transform=transform)
    return dataset


def build_dataloader(dataset, batch_size, num_workers=4, shuffle=True):
    """
    Build dataloader for training or evaluation
    
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
