"""
Pneumonia Detection - Single Image Prediction / Inference Script
Description: This script processes a single chest X-ray image and predicts the
             probability of Pneumonia using the trained custom CNN.
"""

import os
import argparse
import logging
import numpy as np
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Global Configuration Constants
MODEL_PATH = r"C:\Users\Hari\Downloads\anti-ggravity\CNN_learning\models\pneumonia_model.keras"
IMG_SIZE = (224, 224)


def preprocess_single_image(image_path, target_size=IMG_SIZE):
    """
    Loads and preprocesses a single image.
    The custom CNN has an integrated Rescaling layer, so we only need to:
    1. Load the image.
    2. Ensure it is in 3-channel RGB.
    3. Resize it to target dimensions.
    4. Expand dimensions to shape (1, height, width, 3) to represent a batch size of 1.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Load image using Keras utilities
    img = tf.keras.utils.load_img(
        image_path,
        target_size=target_size,
        color_mode="rgb"  # Custom CNN expects 3 channels
    )
    
    # Convert image to numpy array
    img_array = tf.keras.utils.img_to_array(img)
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch


def predict_pneumonia(image_path, model_path=MODEL_PATH):
    """
    Loads the trained model, runs inference on the image, and outputs results.
    """
    # 1. Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # 2. Preprocess Image
    logger.info(f"Preprocessing image: {image_path}")
    img_batch = preprocess_single_image(image_path, target_size=IMG_SIZE)
    
    # 3. Perform Prediction
    logger.info("Running prediction...")
    prediction_prob = model.predict(img_batch, verbose=0)[0][0]
    
    # 4. Interpret Results
    # Since sigmoid output is in [0, 1], values close to 1 are Pneumonia, close to 0 are Normal
    is_pneumonia = prediction_prob >= 0.5
    class_label = "PNEUMONIA" if is_pneumonia else "NORMAL"
    confidence = prediction_prob if is_pneumonia else (1.0 - prediction_prob)
    
    logger.info("--- PREDICTION RESULT ---")
    logger.info(f"Class: {class_label}")
    logger.info(f"Confidence: {confidence * 100:.2f}%")
    logger.info(f"Raw Score (Pneumonia Probability): {prediction_prob:.4f}")
    logger.info("-------------------------")
    
    return class_label, confidence, prediction_prob


if __name__ == "__main__":
    # Command-line interface parser
    parser = argparse.ArgumentParser(
        description="Run Pneumonia prediction on a chest X-ray image."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Absolute or relative path to the JPEG/PNG chest X-ray image."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to the trained .keras model file."
    )
    
    args = parser.parse_args()
    
    try:
        predict_pneumonia(args.image_path, args.model_path)
    except Exception as error:
        logger.error(f"Inference failed: {error}")
