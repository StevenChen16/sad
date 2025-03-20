import os
import yaml

def create_config_from_yaml(yaml_path):
    """
    Create configuration from YAML file
    
    Args:
        yaml_path (str): Path to YAML file
        
    Returns:
        object: Configuration object
    """
    # Load YAML file
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Create configuration object
    from configs.default import Config
    config = Config()
    
    # Update configuration
    return update_config_with_yaml(config, yaml_path)
    

def update_config_with_yaml(config, yaml_path):
    """
    Update configuration with YAML file
    
    Args:
        config (object): Configuration object
        yaml_path (str): Path to YAML file
        
    Returns:
        object: Updated configuration object
    """
    # Load YAML file
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    print(f"Loaded YAML config: {yaml_config}")
    
    # Get absolute path of YAML directory
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    
    # Get path to dataset
    dataset_path = yaml_config.get('path', '')
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(yaml_dir, dataset_path)
    
    print(f"Dataset path: {dataset_path}")
    
    # Get train and val paths
    train_path = os.path.join(dataset_path, 'images', yaml_config.get('train', 'train'))
    val_path = os.path.join(dataset_path, 'images', yaml_config.get('val', 'val'))
    
    # Get train and val label paths
    train_label_path = os.path.join(dataset_path, 'labels', yaml_config.get('train', 'train'))
    val_label_path = os.path.join(dataset_path, 'labels', yaml_config.get('val', 'val'))
    
    print(f"Train image path: {train_path}")
    print(f"Train label path: {train_label_path}")
    print(f"Val image path: {val_path}")
    print(f"Val label path: {val_label_path}")
    
    # Check if paths exist
    if not os.path.exists(train_path):
        print(f"Warning: Train image path {train_path} does not exist")
    if not os.path.exists(train_label_path):
        print(f"Warning: Train label path {train_label_path} does not exist")
    if not os.path.exists(val_path):
        print(f"Warning: Val image path {val_path} does not exist")
    if not os.path.exists(val_label_path):
        print(f"Warning: Val label path {val_label_path} does not exist")
    
    # Update dataset configuration
    config.dataset['train']['img_dir'] = train_path
    config.dataset['train']['label_dir'] = train_label_path
    config.dataset['val']['img_dir'] = val_path
    config.dataset['val']['label_dir'] = val_label_path
    
    # Update class names
    if 'names' in yaml_config:
        class_names = []
        # YAML may have class names as a dict (index: name) or a list
        if isinstance(yaml_config['names'], dict):
            # Sort by index to ensure correct order
            max_idx = max([int(k) for k in yaml_config['names'].keys()])
            class_names = [''] * (max_idx + 1)
            for idx, name in yaml_config['names'].items():
                class_names[int(idx)] = name
        elif isinstance(yaml_config['names'], list):
            class_names = yaml_config['names']
        
        # Remove any empty class names
        config.dataset['class_names'] = [name for name in class_names if name]
        print(f"Class names: {config.dataset['class_names']}")
        
        # Update model config with number of classes
        config.model['num_classes'] = len(config.dataset['class_names'])
    
    # Update dataset format
    config.dataset['format'] = 'yolo'
    
    # Update batch size if provided
    if 'batch_size' in yaml_config:
        config.dataset['train']['batch_size'] = yaml_config['batch_size']
        config.dataset['val']['batch_size'] = yaml_config['batch_size']
    
    return config
