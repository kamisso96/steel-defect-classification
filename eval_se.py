# eval_se.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from config import Config
from models.attention_transfer import build_mobilenetv2_with_se

config = Config()
img_size = config.UNIFIED_CONFIG['img_size']
batch_size = config.UNIFIED_CONFIG['batch_size']

# Load test dataset (first without mapping to get class names)
raw_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    config.test_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)
class_names = raw_test_ds.class_names
print("Classes:", class_names)

# Now apply normalization
test_ds = raw_test_ds.map(lambda x, y: (x/255., y))
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Load model
model = build_mobilenetv2_with_se(input_shape=(*img_size, 3), num_classes=6)
model.load_weights('results/best_mobilenetv2_se.h5')
print("Model loaded successfully.")

# Predict
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - MobileNetV2 + SE')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('results/confusion_matrix_mobilenetv2_se.png', dpi=300)
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))