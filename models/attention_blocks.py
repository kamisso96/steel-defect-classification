# models/attention_blocks.py
import tensorflow as tf
from tensorflow.keras import layers, backend as K

def se_block(input_feature, reduction=16):
    """
    Squeeze-and-Excitation block (channel attention)
    """
    channels = input_feature.shape[-1]
    
    # Squeeze: Global Average Pooling
    squeeze = layers.GlobalAveragePooling2D()(input_feature)
    
    # Excitation: two FC layers with sigmoid
    excitation = layers.Dense(channels // reduction, activation='relu')(squeeze)
    excitation = layers.Dense(channels, activation='sigmoid')(excitation)
    excitation = layers.Reshape((1, 1, channels))(excitation)
    
    # Scale
    scaled = layers.multiply([input_feature, excitation])
    return scaled

def cbam_block(input_feature, reduction=16):
    """
    Convolutional Block Attention Module (channel + spatial)
    """
    channels = input_feature.shape[-1]
    
    # ===== Channel Attention =====
    # Using Lambda layers to wrap TensorFlow ops
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    
    # Shared MLP
    dense1 = layers.Dense(channels // reduction, activation='relu')
    dense2 = layers.Dense(channels, activation='sigmoid')
    
    avg_out = dense2(dense1(avg_pool))
    max_out = dense2(dense1(max_pool))
    
    channel_attention = layers.Add()([avg_out, max_out])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    channel_attention = layers.Reshape((1, 1, channels))(channel_attention)
    
    x = layers.multiply([input_feature, channel_attention])
    
    # ===== Spatial Attention =====
    # Use Lambda layers to wrap tf.reduce_mean and tf.reduce_max
    avg_pool_spatial = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
    max_pool_spatial = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)
    
    concat = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    
    spatial_attention = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    output = layers.multiply([x, spatial_attention])
    
    return output