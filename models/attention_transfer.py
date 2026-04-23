# models/attention_transfer.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from models.attention_blocks import se_block   # reuse your existing SE block

def build_mobilenetv2_with_se(input_shape=(160, 160, 3), num_classes=6, reduction=16):
    """
    MobileNetV2 with a Squeeze-and-Excitation block inserted after global pooling.
    Backbone is frozen; only SE and the classification head are trainable.
    """
    # Load pre-trained MobileNetV2 without top
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # freeze backbone

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    
    # Global average pooling to get feature vector
    x = layers.GlobalAveragePooling2D()(x)   # shape: (batch, 1280)
    
    # Reshape to (batch, 1, 1, 1280) for SE block (SE expects 4D)
    x = layers.Reshape((1, 1, 1280))(x)
    
    # Apply SE block (channel attention)
    x = se_block(x, reduction=reduction)   # output shape: (batch, 1, 1, 1280)
    
    # Flatten back
    x = layers.Flatten()(x)
    
    # Classification head
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='MobileNetV2_SE')
    return model
    