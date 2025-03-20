# SAD: Spotlight Attention Detection

This project implements a novel object detection approach using Spotlight Attention to replace traditional anchor-based methods. The Spotlight Attention Detection (SAD) model uses multiple "spotlight" attention sources to focus on different regions of an image and directly predict object locations and classes.

## Core Concept

Traditional object detection methods like YOLO rely on predefined anchor boxes to detect objects. Instead, SAD uses a set of learnable "spotlight sources" that:

1. Dynamically focus attention on different regions of the feature map
2. Directly predict bounding boxes and class probabilities for objects in those regions
3. Eliminates the need for anchor-based detection and post-processing steps like NMS

This approach is inspired by how humans perceive objects - by focusing attention on different parts of a scene rather than scanning through predefined grids.

## Features

- **Anchor-free detection**: No need for predefined anchor boxes
- **Attention-based**: Uses attention mechanism to focus on important regions
- **End-to-end**: Directly predicts objects without complex post-processing
- **Interpretable**: Attention maps can be visualized to understand model decisions
- **Multiple Dataset Formats**: Supports both YOLO and COCO format datasets

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- torchvision
- matplotlib (for visualization)
- PIL (for image processing)
- PyYAML (for config files)

### Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/sad-detector.git
cd sad-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Setup

The project supports two dataset formats:

#### YOLO Format
```
dataset_root/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── ...
    └── val/
        ├── image1.txt
        └── ...
```

Each label file follows the YOLO format: `class_id x_center y_center width height` with normalized coordinates.

#### COCO Format
```
dataset_root/
├── train2017/
│   ├── image1.jpg
│   └── ...
├── val2017/
│   ├── image1.jpg
│   └── ...
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Configuration

Create a YAML configuration file (e.g., `coco.yaml`) with the following structure:

```yaml
# Dataset path
path: path/to/dataset_root

# Relative paths to train and validation sets
train: train
val: val

# Class names
names:
  0: person
  1: bicycle
  2: car
  # ...
```

The project will automatically detect whether your dataset is in YOLO or COCO format.

## Usage

### Demo

Run the attention mechanism demo to visualize how spotlights focus on different parts of an image:

```bash
python demo.py --image path/to/your/image.jpg --output visualization.png
```

### Training

Train the model using your dataset:

```bash
python train.py --yaml path/to/your/dataset.yaml
```

Additional training options:
```bash
python train.py --yaml path/to/your/dataset.yaml --epochs 100 --batch_size 16
```

### Evaluation

Evaluate the trained model:

```bash
python evaluate.py --yaml path/to/your/dataset.yaml
```

### Inference

Run inference on a single image or directory:

```bash
python inference.py --input path/to/your/image.jpg --yaml path/to/your/dataset.yaml
python inference.py --input path/to/your/images_dir --output results_dir --yaml path/to/your/dataset.yaml
```

## Project Structure

```
SAD/
├── models/
│   ├── __init__.py
│   ├── spotlight_attention.py  # Core attention mechanism
│   └── sad_detector.py         # Full detector model
├── utils/
│   ├── __init__.py
│   ├── config.py               # Configuration loading
│   ├── dataset.py              # COCO dataset handling
│   ├── dataset_yolo.py         # YOLO dataset handling  
│   ├── sad_loss.py             # Loss function
│   ├── post_processing.py      # Post-processing utilities
│   └── visualization.py        # Visualization utilities
├── configs/
│   └── default.py              # Default configuration
├── demo.py                     # Attention visualization demo
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── inference.py                # Inference script
└── README.md
```

## Example: Using with COCO8 Tiny Dataset

The COCO8 tiny dataset is a small subset of COCO dataset with YOLO format labels that's perfect for testing:

```bash
# Download the dataset (if needed)
# git clone https://github.com/ultralytics/ultralytics
# cp -r ultralytics/datasets/coco8 ./datasets/

# Train model on coco8
python train.py --yaml coco.yaml
```

When using the coco8 dataset, make sure your coco.yaml file looks like:

```yaml
path: ./datasets/coco8  # dataset root directory
train: train  # relative path to train images
val: val  # relative path to val images

# Class names
names:
  0: person
  1: bicycle
  # ...
```

## Customization

You can customize the model's hyperparameters by modifying the configuration in `configs/default.py` or by providing a custom config file with the `--config` argument.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by DETR (DEtection TRansformer) from Facebook Research
- Builds upon ideas from attention mechanisms in computer vision
