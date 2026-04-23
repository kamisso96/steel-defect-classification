# data/augmentation.py

import tensorflow as tf

def get_data_augmentation():
    """Return a sequential model with data augmentation layers"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

# Example usage in training pipeline:
# augmentation = get_data_augmentation()
# train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))
