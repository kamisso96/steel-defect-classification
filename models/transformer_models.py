# models/transformer_models.py
import tensorflow as tf
from tensorflow.keras import layers, Model

def mlp(x, hidden_units, dropout_rate):
    """Multi-layer perceptron block"""
    for units in hidden_units:
        x = layers.Dense(units, activation='gelu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def build_tiny_vit(input_shape=(160, 160, 3), num_classes=6, patch_size=8, projection_dim=64, transformer_layers=4, num_heads=4, mlp_head_units=[128]):
    """
    A tiny Vision Transformer suitable for CPU training.
    """
    inputs = layers.Input(shape=input_shape)

    # 1. Patch extraction + linear projection (patch embedding)
    patches = layers.Conv2D(
        projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid'
    )(inputs)  # Shape: (batch, n_patches_h, n_patches_w, projection_dim)
    
    # Reshape to sequence of patches
    patch_grid_h = input_shape[0] // patch_size
    patch_grid_w = input_shape[1] // patch_size
    num_patches = patch_grid_h * patch_grid_w
    x = layers.Reshape((num_patches, projection_dim))(patches)

    # 2. Add position embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    x = x + position_embedding

    # 3. Transformer encoder layers
    for _ in range(transformer_layers):
        # Layer norm 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=0.1
        )(x1, x1)
        # Residual connection
        x2 = layers.Add()([x, attention_output])
        # Layer norm 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.1)
        # Residual connection
        x = layers.Add()([x2, x3])

    # 4. Classification head
    # Take the output of the first token (like [CLS]) – but here we use global average pooling
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = mlp(x, hidden_units=mlp_head_units, dropout_rate=0.2)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='TinyViT')
    return model


def build_hybrid_cnn_transformer(input_shape=(160, 160, 3), num_classes=6):
    """
    Lightweight hybrid: a few CNN layers then a Transformer encoder.
    Dynamically computes the spatial dimensions after pooling.
    """
    inputs = layers.Input(shape=input_shape)

    # CNN front-end (small)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)               # -> height/2, width/2
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)               # -> height/4, width/4

    # Get spatial dimensions dynamically
    # We'll use the shape of x at runtime. For building the model, we need a fixed dimension.
    # Since input_shape is known at build time, we can compute it statically.
    h = input_shape[0] // 4
    w = input_shape[1] // 4
    x = layers.Reshape((h * w, 64))(x)

    # Transformer encoder (single layer)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='Hybrid_CNN_Transformer')
    return model