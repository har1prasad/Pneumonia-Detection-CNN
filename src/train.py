"""
Pneumonia Detection - Model Training Pipeline
Description: This script compiles, configures class weights, trains the custom CNN,
             and exports the model in Keras format alongside metric plots.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)

# Import our custom dataset loading and model architecture functions
from data_preprocessing import check_gpu, load_datasets
from model import build_pneumonia_cnn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)

# Global Configuration Constants
DATASET_DIR = r"C:\Users\Hari\Downloads\anti-ggravity\CNN_learning\dataset\chest_xray"
MODEL_DIR = r"C:\Users\Hari\Downloads\anti-ggravity\CNN_learning\models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "pneumonia_model.keras")
BATCH_SIZE = 32
EPOCHS = 30
IMG_SIZE = (224, 224)


def get_class_weights(train_dir):
    """
    Computes class weights dynamically from file counts to address class imbalance.
    Normal scans are class 0; Pneumonia scans are class 1.
    Formula: weight = total_samples / (num_classes * class_samples)
    """
    normal_dir = os.path.join(train_dir, "NORMAL")
    pneumonia_dir = os.path.join(train_dir, "PNEUMONIA")
    
    if not os.path.exists(normal_dir) or not os.path.exists(pneumonia_dir):
        logger.warning(
            f"Directory structure check failed. Check {normal_dir} or {pneumonia_dir}. "
            "Falling back to default class weights."
        )
        return {0: 1.0, 1: 1.0}

    normal_count = len(os.listdir(normal_dir))
    pneumonia_count = len(os.listdir(pneumonia_dir))
    total_samples = normal_count + pneumonia_count

    logger.info(f"Class counts in training data -> NORMAL: {normal_count}, PNEUMONIA: {pneumonia_count}")

    # Compute weights
    weight_for_normal = total_samples / (2.0 * normal_count)
    weight_for_pneumonia = total_samples / (2.0 * pneumonia_count)

    class_weights = {0: weight_for_normal, 1: weight_for_pneumonia}
    logger.info(f"Calculated class weights -> NORMAL (0): {weight_for_normal:.4f}, PNEUMONIA (1): {weight_for_pneumonia:.4f}")
    return class_weights


def plot_training_history(history, output_dir):
    """
    Saves a visualization of the training loss and accuracy history.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs_range = range(1, len(history.history["loss"]) + 1)
    
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history["accuracy"], label="Training Accuracy")
    plt.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history["loss"], label="Training Loss")
    plt.plot(epochs_range, history.history["val_loss"], label="Validation Loss")
    plt.title("Model Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plot_path = os.path.join(output_dir, "training_history.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Training history plot saved to: {plot_path}")


def run_training():
    """
    Executes the complete training workflow.
    """
    # 1. Check GPU environment
    check_gpu()

    # 2. Load Datasets
    try:
        train_ds, val_ds, _ = load_datasets(
            DATASET_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 3. Compute Class Weights
    train_dir = os.path.join(DATASET_DIR, "train")
    class_weights = get_class_weights(train_dir)

    # 4. Build Model
    model = build_pneumonia_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # 5. Compile Model
    # Using BinaryCrossentropy loss and Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    logger.info("Model compiled successfully.")

    # 6. Setup Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save the best model strictly in Keras native format
    checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    # Early stopping if validation loss plateaus for 10 epochs
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Learning rate scheduler (reduces LR when validation loss plateaus)
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # CSV logger to track training epoch metrics
    csv_logger = CSVLogger(os.path.join(MODEL_DIR, "training_log.csv"), append=False)

    callbacks_list = [checkpoint, early_stop, lr_scheduler, csv_logger]

    # 7. Start Training
    logger.info("Starting model training...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )
    logger.info("Model training completed.")

    # 8. Plot and Save Learning Curves
    plot_training_history(history, MODEL_DIR)


if __name__ == "__main__":
    run_training()
