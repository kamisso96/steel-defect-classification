# models/transfer_models.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

def build_mobilenetv2_finetune(input_shape=(128, 128, 3), num_classes=6, freeze_backbone=True):
    """
    MobileNetV2 pre-trained on ImageNet, fine-tuned for steel defects.
    Smaller and faster than VGG16/EfficientNet.
    """
    # Load pre-trained MobileNetV2 without top
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze backbone if requested
    base_model.trainable = not freeze_backbone
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='MobileNetV2_FT')
    return model