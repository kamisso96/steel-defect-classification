import os
import matplotlib.pyplot as plt
import numpy as np

# Path to your results folder
results_dir = 'results'

# Model names (as they appear in result files)
model_keys = ['baseline', 'se', 'cbam', 'mobilenetv2', 'mobilenetv2_se', 'vit', 'hybrid']
model_display_names = {
    'baseline': 'Baseline CNN',
    'se': 'CNN + SE',
    'cbam': 'CNN + CBAM',
    'mobilenetv2': 'MobileNetV2',
    'mobilenetv2_se': 'MobileNetV2 + SE (Proposed)',
    'vit': 'TinyViT',
    'hybrid': 'Hybrid CNN-Transformer'
}

# Inference times measured on CPU (ms per sample) – from Table 4.9
inference_times = {
    'baseline': 4.2,
    'se': 4.5,
    'cbam': 4.7,
    'mobilenetv2': 5.1,
    'mobilenetv2_se': 5.4,
    'vit': 18.3,
    'hybrid': 12.6
}

# Read test accuracies from result files
accuracies = {}
for key in model_keys:
    result_file = os.path.join(results_dir, f'{key}_test_result.txt')
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            for line in f:
                if line.startswith('Test accuracy:'):
                    acc = float(line.split(':')[1].strip()) * 100  # as percentage
                    accuracies[key] = acc
                    break
    else:
        print(f"Warning: {result_file} not found")

# Prepare data for scatter plot
x = [inference_times[k] for k in model_keys if k in accuracies]
y = [accuracies[k] for k in model_keys if k in accuracies]
labels = [model_display_names[k] for k in model_keys if k in accuracies]

# Create plot
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(x)))
plt.scatter(x, y, s=120, c=colors, alpha=0.8)

# Add labels to each point
for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.xlabel('Inference Time (ms per sample)', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.title('Figure 4.11: Test Accuracy vs. Inference Time (CPU)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Optional: draw a dashed line showing the Pareto frontier (best trade‑off)
# You can manually identify the best points or leave it out.

plt.tight_layout()
save_path = os.path.join(results_dir, 'accuracy_vs_inference.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Scatter plot saved to {save_path}")