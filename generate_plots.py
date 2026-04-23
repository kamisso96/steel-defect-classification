# generate_plots.py
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from config import Config
from models.baseline import build_baseline_cnn
from models.attention_models import build_cnn_se, build_cnn_cbam
from models.transfer_models import build_mobilenetv2_finetune
from models.attention_transfer import build_mobilenetv2_with_se
from models.transformer_models import build_tiny_vit, build_hybrid_cnn_transformer

# ==================== CONFIGURATION ====================
config = Config()
results_dir = config.results_dir
img_size = (160, 160)  # must match training
batch_size = 16

# Model names and display names
models = {
    'baseline': 'Baseline CNN',
    'se': 'CNN + SE',
    'cbam': 'CNN + CBAM',
    'mobilenetv2': 'MobileNetV2',
    'vit': 'TinyViT',
    'hybrid': 'Hybrid CNN-Transformer',
    'mobilenetv2_se': 'MobileNetV2 + SE'
}

# Mapping from model name to builder function (for loading)
builders = {
    'baseline': lambda: build_baseline_cnn(input_shape=(*img_size, 3), num_classes=6),
    'se': lambda: build_cnn_se(input_shape=(*img_size, 3), num_classes=6),
    'cbam': lambda: build_cnn_cbam(input_shape=(*img_size, 3), num_classes=6),
    'mobilenetv2': lambda: build_mobilenetv2_finetune(input_shape=(*img_size, 3), num_classes=6, freeze_backbone=True),
    'vit': lambda: build_tiny_vit(input_shape=(*img_size, 3), num_classes=6),
    'hybrid': lambda: build_hybrid_cnn_transformer(input_shape=(*img_size, 3), num_classes=6),
    'mobilenetv2_se': lambda: build_mobilenetv2_with_se(input_shape=(*img_size, 3), num_classes=6)
}

# ==================== LOAD TEST DATA ====================
def prepare_test_dataset():
    # First load raw dataset to get class names
    raw_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.test_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    class_names = raw_test_ds.class_names

    # Now apply transformations
    normalization = tf.keras.layers.Rescaling(1./255)
    test_ds = raw_test_ds.map(lambda x, y: (normalization(x), y))
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return test_ds, class_names

test_ds, class_names = prepare_test_dataset()
print("Classes:", class_names)

# ==================== 1. BAR CHART OF TEST ACCURACIES ====================
def plot_accuracy_bar_chart():
    # Read accuracies from result files
    accuracies = {}
    for model_name in models.keys():
        result_file = os.path.join(results_dir, f'{model_name}_test_result.txt')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('Test accuracy:'):
                        acc = float(line.split(':')[1].strip())
                        accuracies[models[model_name]] = acc * 100  # convert to percent
                        break
        else:
            print(f"Warning: {result_file} not found. Skipping {model_name}.")

    if not accuracies:
        print("No accuracy files found. Skipping bar chart.")
        return

    plt.figure(figsize=(10,6))
    bars = plt.bar(accuracies.keys(), accuracies.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    plt.ylabel('Test Accuracy (%)')
    plt.title('Model Comparison on NEU Dataset (160×160)')
    plt.ylim(0, 105)
    for bar, acc in zip(bars, accuracies.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'), dpi=300)
    plt.show()
    print("Bar chart saved to", os.path.join(results_dir, 'accuracy_comparison.png'))

# ==================== 2. CONFUSION MATRICES ====================
def plot_confusion_matrix(model_name, model, test_ds, class_names):
    # Get predictions
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {models[model_name]}')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{model_name}.png'), dpi=300)
    plt.show()
    print(f"Confusion matrix for {model_name} saved.")

def generate_confusion_matrices():
    # For the two models we want: baseline and mobilenetv2
    target_models = ['baseline', 'mobilenetv2']
    for model_name in target_models:
        model_path = os.path.join(results_dir, f'best_{model_name}.h5')
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found. Skipping confusion matrix for {model_name}.")
            continue
        # Recreate model architecture
        model = builders[model_name]()
        model.load_weights(model_path)
        plot_confusion_matrix(model_name, model, test_ds, class_names)

# ==================== 3. TRAINING HISTORY PLOTS (from saved PNGs) ====================
def display_training_history_pngs():
    # Use lower DPI (100) and slightly smaller figure size
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))  # reduced from (15,12)
    axes = axes.flatten()
    for idx, model_name in enumerate(models.keys()):
        history_png = os.path.join(results_dir, f'{model_name}_history.png')
        if os.path.exists(history_png):
            img = plt.imread(history_png)
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(models[model_name], fontsize=8)  # smaller font
        else:
            axes[idx].text(0.5, 0.5, 'No image available', ha='center', va='center', fontsize=8)
            axes[idx].axis('off')
    # Turn off any remaining unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    plt.suptitle('Figures 4.4-4.10: Training Histories of All Models',
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    # Save with 100 DPI instead of 300
    save_path = os.path.join(results_dir, 'all_training_histories.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print(f"Combined training histories saved to: {save_path} (100 DPI)")

# ==================== 3b. ALTERNATIVE: Generate from saved history objects ====================
def plot_history_from_pickle(model_name):
    pkl_path = os.path.join(results_dir, f'{model_name}_history.pkl')
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        history = pickle.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    ax1.plot(history['accuracy'], label='Train')
    ax1.plot(history['val_accuracy'], label='Validation')
    ax1.set_title(f'{models[model_name]} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(history['loss'], label='Train')
    ax2.plot(history['val_loss'], label='Validation')
    ax2.set_title(f'{models[model_name]} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_history_from_pkl.png'), dpi=300)
    plt.show()
    return fig

def generate_all_history_from_pickle():
    for model_name in models.keys():
        plot_history_from_pickle(model_name)

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("Generating plots for Chapter 4...")

    # 1. Bar chart
    plot_accuracy_bar_chart()

    # 2. Confusion matrices for baseline and mobilenetv2
    generate_confusion_matrices()

    # 3. Training histories (choose one method)
    # If you have the PNG files:
    display_training_history_pngs()

    # If you have saved pickle histories, uncomment the next line:
    # generate_all_history_from_pickle()

    print("All plots generated successfully!")