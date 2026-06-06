"""
Pneumonia Detection - Data Preprocessing Pipeline
Description: This script sets up GPU devices, configures logging, loads the chest
             X-ray datasets, splits the training data to create a robust
             validation set, and configures data augmentation.
"""

import os
import logging
import tensorflow as tf
from tensorflow.keras import layers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocessing.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)


def check_gpu():
    """
    Detects and logs available GPU devices for TensorFlow.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"GPU(s) detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu.name}")
            # Enable memory growth to prevent TensorFlow from allocating all memory at once
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"  Enabled memory growth for GPU {i}")
            except Exception as e:
                logger.warning(f"  Could not set memory growth: {e}")
    else:
        logger.info("No GPU detected. Running on CPU.")


def get_data_augmentation():
    """
    Creates a data augmentation pipeline using Keras preprocessing layers.
    Only realistic chest X-ray perturbations are applied:
    - Random rotation (max 8 degrees/0.022 turns)
    - Random zoom (max 10%)
    - Random translation/shift (max 8%)
    - Random horizontal flip (lung symmetry generalization)
    - NO vertical flip (X-rays are never upside down)
    """
    augmentation_model = tf.keras.Sequential([
        layers.RandomRotation(factor=0.022, name="random_rotation"),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1, name="random_zoom"),
        layers.RandomTranslation(
            height_factor=0.08, width_factor=0.08, name="random_translation"
        ),
        layers.RandomFlip(mode="horizontal", name="random_flip")
    ], name="data_augmentation")
    
    return augmentation_model


def load_datasets(dataset_dir, img_size=(224, 224), batch_size=32):
    """
    Loads Chest X-Ray datasets from the specified directory.
    To solve the issue of a tiny validation set (16 images) in the raw dataset,
    this function uses the large train/ folder and splits it dynamically into
    80% Training and 20% Validation subsets. The test/ folder remains isolated.

    Args:
        dataset_dir (str): Base directory of the dataset containing 'train' and 'test'.
        img_size (tuple): Target size for image resizing.
        batch_size (int): Batch size for training.

    Returns:
        train_ds (tf.data.Dataset): Augmented training dataset.
        val_ds (tf.data.Dataset): Normalized validation dataset.
        test_ds (tf.data.Dataset): Normalized test dataset.
    """
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Expected 'train' and 'test' folders inside '{dataset_dir}'"
        )

    logger.info("Loading training and validation datasets from train folder...")
    # Load Training Set (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary"
    )

    # Load Validation Set (20%)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary"
    )

    logger.info("Loading test dataset from test folder...")
    # Load Test Set (Isolated)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False
    )

    # Configure dataset performance options and pipeline transformations
    # Prefetch overlapping CPU pre-processing with GPU execution
    AUTOTUNE = tf.data.AUTOTUNE

    # Data augmentation is applied on-the-fly during training only
    augmentation_model = get_data_augmentation()

    # Apply data augmentation only to training dataset
    train_ds = train_ds.map(
        lambda x, y: (augmentation_model(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

    # Cache datasets in memory or local storage to speed up training
    # Prefetch for speed optimization
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    logger.info(f"Loaded train classes: {train_ds.class_names if hasattr(train_ds, 'class_names') else 'binary'}")
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Test script run
    check_gpu()
    base_dir = r"C:\Users\Hari\Downloads\anti-ggravity\CNN_learning\dataset\chest_xray"
    try:
        train, val, test = load_datasets(base_dir)
        logger.info("Dataset loaded successfully!")
    except Exception as error:
        logger.error(f"Error during dataset loading: {error}")
