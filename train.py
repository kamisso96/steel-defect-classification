# train.py
import tensorflow as tf
import argparse
import os
from config import Config
from models.baseline import build_baseline_cnn
from models.attention_models import build_cnn_se, build_cnn_cbam
from models.transfer_models import build_mobilenetv2_finetune
from models.transformer_models import build_tiny_vit, build_hybrid_cnn_transformer
from models.attention_transfer import build_mobilenetv2_with_se
from models.utils import get_callbacks, plot_training_history

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='baseline',
                    choices=['baseline', 'se', 'cbam', 'mobilenetv2', 'vit', 'hybrid', 'mobilenetv2_se'],
                    help='Model to train: baseline, se, cbam, mobilenetv2, vit, hybrid, mobilenetv2_se')
args = parser.parse_args()

# Load config
config = Config()

# Use unified configuration for all models (160x160, batch size 16, epochs 30)
current_img_size = config.UNIFIED_CONFIG['img_size']
current_batch_size = config.UNIFIED_CONFIG['batch_size']
current_epochs = config.UNIFIED_CONFIG['epochs']

# Choose learning rate based on model type
if args.model in ['baseline', 'se', 'cbam']:
    current_lr = config.UNIFIED_CONFIG['learning_rate_attention']
else:  # mobilenetv2, vit, hybrid, mobilenetv2_se
    current_lr = config.UNIFIED_CONFIG['learning_rate_transfer']

print(f"\n=== Training Configuration ===")
print(f"Model: {args.model}")
print(f"Image size: {current_img_size}")
print(f"Batch size: {current_batch_size}")
print(f"Epochs: {current_epochs}")
print(f"Learning rate: {current_lr}")
print("==============================\n")

# Prepare data pipelines
def prepare_dataset(data_dir, batch_size, img_size, augment=False):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    # Normalize to [0,1]
    normalization = tf.keras.layers.Rescaling(1./255)
    ds = ds.map(lambda x, y: (normalization(x), y))
    if augment:
        # Add simple augmentation for training only
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y))
    ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

print("Loading datasets...")

# First, load a temporary dataset to get class names (before any transformations)
temp_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    config.train_dir,
    image_size=current_img_size,
    batch_size=current_batch_size,
    shuffle=True
)
class_names = temp_train_ds.class_names
print("Classes:", class_names)

# Now load the actual datasets with preprocessing
train_ds = prepare_dataset(config.train_dir, current_batch_size, current_img_size, augment=True)
val_ds = prepare_dataset(config.val_dir, current_batch_size, current_img_size, augment=False)
test_ds = prepare_dataset(config.test_dir, current_batch_size, current_img_size, augment=False)

# Build model
if args.model == 'baseline':
    model = build_baseline_cnn(
        input_shape=(*current_img_size, config.num_channels),
        num_classes=config.num_classes
    )
    model_name = 'baseline'

elif args.model == 'se':
    model = build_cnn_se(
        input_shape=(*current_img_size, config.num_channels),
        num_classes=config.num_classes
    )
    model_name = 'se'

elif args.model == 'cbam':
    model = build_cnn_cbam(
        input_shape=(*current_img_size, config.num_channels),
        num_classes=config.num_classes
    )
    model_name = 'cbam'

elif args.model == 'mobilenetv2':
    model = build_mobilenetv2_finetune(
        input_shape=(*current_img_size, config.num_channels),
        num_classes=config.num_classes,
        freeze_backbone=True  # freeze for faster training; can unfreeze later if desired
    )
    model_name = 'mobilenetv2'

elif args.model == 'vit':
    model = build_tiny_vit(
        input_shape=(*current_img_size, config.num_channels),
        num_classes=config.num_classes
    )
    model_name = 'vit'

elif args.model == 'hybrid':
    model = build_hybrid_cnn_transformer(
        input_shape=(*current_img_size, config.num_channels),
        num_classes=config.num_classes
    )
    model_name = 'hybrid'

elif args.model == 'mobilenetv2_se':
    model = build_mobilenetv2_with_se(
        input_shape=(*current_img_size, config.num_channels),
        num_classes=config.num_classes
    )
    model_name = 'mobilenetv2_se'

else:
    raise ValueError(f"Unknown model: {args.model}")

model.summary()

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=current_lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = get_callbacks(model_name, config)

# Train
print(f"Starting training for {model_name}...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=current_epochs,
    callbacks=callbacks,
    verbose=1
)

# Save final model
final_model_path = os.path.join(config.results_dir, f'{model_name}_final.h5')
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")

# Plot history
plot_training_history(history, model_name, config)

# Evaluate on test set
print("Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Save test result to a file
with open(os.path.join(config.results_dir, f'{model_name}_test_result.txt'), 'w') as f:
    f.write(f"Test accuracy: {test_acc:.4f}\n")
    f.write(f"Test loss: {test_loss:.4f}\n")