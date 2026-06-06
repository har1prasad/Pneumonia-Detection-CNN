"""
Pneumonia Detection - Custom CNN Model Architecture
Description: This script defines the custom Convolutional Neural Network (CNN)
             architecture from scratch. It does not use transfer learning.
"""

import logging
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


def build_pneumonia_cnn(input_shape=(224, 224, 3)):
    """
    Builds and returns a custom deep CNN for Pneumonia detection.
    
    The architecture features:
    - 4 Convolutional blocks with increasing filters (32, 64, 128, 256)
    - Batch Normalization after each Conv2D to stabilize activations
    - Progressive Dropout rates (0.2, 0.2, 0.3, 0.4) to combat overfitting
    - Global Average Pooling (GAP) instead of Flattening to decrease weights
    - Fully Connected (Dense) reasoning layer with high Dropout (0.5)
    - Single-neuron output layer with Sigmoid activation for binary classification

    Args:
        input_shape (tuple): Dimensionality of the input image tensor.

    Returns:
        model (tf.keras.Model): Built and uncompiled Keras Model instance.
    """
    logger.info(f"Building custom CNN with input shape: {input_shape}")

    # Define the inputs
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Rescaling layer: normalizes pixels from [0, 255] to [0.0, 1.0]
    x = layers.Rescaling(scale=1.0 / 255.0, name="rescaling")(inputs)

    # --- Block 1 ---
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", name="conv2d_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_1")(x)
    x = layers.Dropout(0.2, name="dropout_1")(x)

    # --- Block 2 ---
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", name="conv2d_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)

    # --- Block 3 ---
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", name="conv2d_3")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Activation("relu", name="relu_3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_3")(x)
    x = layers.Dropout(0.3, name="dropout_3")(x)

    # --- Block 4 ---
    x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", name="conv2d_4")(x)
    x = layers.BatchNormalization(name="bn_4")(x)
    x = layers.Activation("relu", name="relu_4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_4")(x)
    x = layers.Dropout(0.4, name="dropout_4")(x)

    # --- Global Pooling & Dense Classification Head ---
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    # Dense FC Layer
    x = layers.Dense(128, name="dense_fc")(x)
    x = layers.BatchNormalization(name="bn_fc")(x)
    x = layers.Activation("relu", name="relu_fc")(x)
    x = layers.Dropout(0.5, name="dropout_fc")(x)

    # Output node (binary classification)
    outputs = layers.Dense(1, activation="sigmoid", name="output_classification")(x)

    # Construct the functional Keras model
    model = Model(inputs=inputs, outputs=outputs, name="Pneumonia_Custom_CNN")
    
    logger.info("Custom CNN built successfully.")
    return model


if __name__ == "__main__":
    # Test building the model and display architecture summary
    logging.basicConfig(level=logging.INFO)
    model = build_pneumonia_cnn()
    model.summary()
