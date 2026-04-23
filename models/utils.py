# models/utils.py
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os

def get_callbacks(model_name, config):
    """Returns training callbacks: ModelCheckpoint, EarlyStopping, TensorBoard."""
    checkpoint_path = os.path.join(config.results_dir, f'best_{model_name}.h5')
    log_dir = os.path.join(config.results_dir, 'logs', model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
    ]
    return callbacks

def plot_training_history(history, model_name, config):
    """Plot training and validation accuracy/loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='train_accuracy')
    ax1.plot(history.history['val_accuracy'], label='val_accuracy')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='train_loss')
    ax2.plot(history.history['val_loss'], label='val_loss')
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(config.results_dir, f'{model_name}_history.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}")