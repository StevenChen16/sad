import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import argparse

class SpotlightAttentionDemo(nn.Module):
    """
    Simplified demo version of Spotlight Attention to visualize how it works
    """
    def __init__(self, feature_size=32, num_spotlights=9, hidden_dim=64, spotlight_dim=32):
        super(SpotlightAttentionDemo, self).__init__()
        self.num_spotlights = num_spotlights
        self.hidden_dim = hidden_dim
        
        # Learnable spotlight sources (query points)
        self.spotlight_sources = nn.Parameter(torch.randn(num_spotlights, spotlight_dim))
        
        # Feature transformation layers (simplified for demo)
        self.key_proj = nn.Conv2d(3, spotlight_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(3, hidden_dim, kernel_size=1)
        
        # Initialize spotlight sources to focus on different regions
        self._initialize_spotlight_sources()
    
    def _initialize_spotlight_sources(self):
        """Initialize spotlight sources to have some diversity"""
        if self.num_spotlights == 9:
            # Initialize to roughly correspond to a 3x3 grid for visualization
            grid = torch.tensor([
                [-1, -1], [-1, 0], [-1, 1],
                [0, -1], [0, 0], [0, 1],
                [1, -1], [1, 0], [1, 1]
            ], dtype=torch.float32)
            
            # Use these positions to influence the initial spotlight sources
            for i in range(min(9, self.num_spotlights)):
                direction = grid[i].view(1, 2)
                direction = F.normalize(direction, dim=1)
                # Project this direction into the spotlight space
                self.spotlight_sources.data[i, :2] = direction
    
    def forward(self, x):
        """
        Forward pass to generate attention maps
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W)
            
        Returns:
            dict: Contains attention maps and value features
        """
        batch_size, _, height, width = x.shape
        
        # Project features to get keys and values
        keys = self.key_proj(x)  # (B, spotlight_dim, H, W)
        values = self.value_proj(x)  # (B, hidden_dim, H, W)
        
        # Reshape for attention computation
        keys = keys.flatten(2).permute(0, 2, 1)  # (B, H*W, spotlight_dim)
        values = values.flatten(2).permute(0, 2, 1)  # (B, H*W, hidden_dim)
        
        # Expand spotlight sources for batch dimension
        spotlight_queries = self.spotlight_sources.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Calculate similarity between spotlight sources and keys
        attention_logits = torch.bmm(spotlight_queries, keys.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_logits, dim=2)  # (B, num_spotlights, H*W)
        
        # Reshape attention maps back to 2D for visualization
        attention_maps = attention_weights.view(batch_size, self.num_spotlights, height, width)
        
        # Apply attention to values (optional for demo)
        spotlight_features = torch.bmm(attention_weights, values)  # (B, num_spotlights, hidden_dim)
        
        return {
            'attention_maps': attention_maps,
            'spotlight_features': spotlight_features
        }


def visualize_spotlight_attention(image_path, model, output_path=None):
    """
    Visualize the spotlight attention maps on an image
    
    Args:
        image_path (str): Path to the input image
        model (nn.Module): Spotlight attention model
        output_path (str, optional): Path to save visualization. If None, just displays.
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Get attention maps
    with torch.no_grad():
        outputs = model(input_tensor)
    
    attention_maps = outputs['attention_maps'][0]  # Remove batch dimension
    
    # Visualize
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # Show original image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Show attention maps
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < attention_maps.size(0):
            # Get attention map
            attention_map = attention_maps[i].cpu().numpy()
            
            # Display the original image
            ax.imshow(np.array(image))
            
            # Overlay attention map with some transparency
            ax.imshow(attention_map, alpha=0.6, cmap='hot')
            ax.set_title(f"Spotlight {i+1}")
            ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Spotlight Attention Demo")
    parser.add_argument('--image', type=str, default=None, 
                        help='Path to input image. If not provided, you need to modify the code to use a test image.')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save visualization. If not provided, it will display the results.')
    parser.add_argument('--num_spotlights', type=int, default=9,
                        help='Number of spotlight sources to generate (default: 9)')
    
    args = parser.parse_args()
    
    # Create model
    model = SpotlightAttentionDemo(num_spotlights=args.num_spotlights)
    
    # Path to your image
    image_path = args.image
    if image_path is None:
        print("Please provide an image path with --image argument")
        return
    
    # Visualize attention maps
    visualize_spotlight_attention(image_path, model, args.output)
    print(f"{'Saved' if args.output else 'Displayed'} attention visualization for {image_path}")


if __name__ == "__main__":
    main()