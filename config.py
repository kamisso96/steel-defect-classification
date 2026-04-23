# config.py

import tensorflow as tf

class Config:
    # Paths
    train_dir = 'data/NEU_prepared/train'
    val_dir = 'data/NEU_prepared/val'
    test_dir = 'data/NEU_prepared/test'
    
    # Data parameters
    img_size = (224, 224)
    num_channels = 3
    batch_size = 32
    num_classes = 6  # NEU has 6 classes
    
    # Training parameters
    epochs = 50
    learning_rate = 0.001
    early_stopping_patience = 10
    
    # Model saving
    results_dir = 'results'
    
    # Class names (verify with your dataset)
    class_names = ['crazing', 'inclusion', 'patches', 
                   'pitted_surface', 'rolled-in_scale', 'scratches']
    
    # GPU settings (if needed)
    gpu_memory_limit = None  # Set to e.g., 4096 for 4GB limit if needed

    # config.py (add at the end)

        # Unified configuration for fair comparison (all models at same settings)
    UNIFIED_CONFIG = {
        'img_size': (160, 160),               # 160x160 resolution
        'batch_size': 16,                      # consistent batch size
        'epochs': 30,                           # enough for convergence
        'learning_rate_attention': 0.001,       # for baseline, SE, CBAM
        'learning_rate_transfer': 0.0001,       # for MobileNetV2, ViT, hybrid
    }

# Optional: Limit GPU memory (prevents OOM errors)
if Config.gpu_memory_limit:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=Config.gpu_memory_limit)]
            )
        except RuntimeError as e:
            print(e)