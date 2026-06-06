"""
Pneumonia Detection - Model Evaluation Pipeline
Description: This script evaluates the trained Keras model on the isolated test set,
             computes metrics, and saves the Confusion Matrix and ROC Curve plots.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# Import dataset loading utility
from data_preprocessing import load_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluation.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)

# Global Configuration Constants
DATASET_DIR = r"C:\Users\Hari\Downloads\anti-ggravity\CNN_learning\dataset\chest_xray"
MODEL_DIR = r"C:\Users\Hari\Downloads\anti-ggravity\CNN_learning\models"
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.keras")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def generate_plots(y_true, y_pred_probs, output_dir):
    """
    Generates and saves the Confusion Matrix and the ROC Curve.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate Binary Predictions using 0.5 default decision threshold
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NORMAL (0)", "PNEUMONIA (1)"],
        yticklabels=["NORMAL (0)", "PNEUMONIA (1)"]
    )
    plt.title("Confusion Matrix (Test Set)")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Confusion Matrix saved to: {cm_path}")

    # 3. ROC Curve & AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    logger.info(f"ROC Curve saved to: {roc_path}")
    
    return cm, roc_auc


def run_evaluation():
    """
    Loads the trained model, performs predictions on the isolated test set,
    displays performance summaries, and saves visualization plots.
    """
    # 1. Verify Model Existence
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Saved model not found at {MODEL_PATH}. Run training first.")
        return

    logger.info(f"Loading trained model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 2. Load Dataset (retrieve only test dataset)
    try:
        _, _, test_ds = load_datasets(
            DATASET_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE
        )
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return

    logger.info("Evaluating model on test dataset...")
    
    # 3. Run direct model evaluation to fetch compiled metrics
    eval_results = model.evaluate(test_ds, verbose=1)
    
    # Map metrics based on the index returned by model.evaluate
    # Order: loss, accuracy, precision, recall, auc
    metrics_str = "Test Metrics:\n"
    for name, value in zip(model.metrics_names, eval_results):
        metrics_str += f"  {name}: {value:.4f}\n"
    logger.info(metrics_str)

    # Write metrics to a text file for easy viewing
    metrics_txt_path = os.path.join(MODEL_DIR, "test_metrics.txt")
    with open(metrics_txt_path, "w") as f:
        f.write(metrics_str)
    logger.info(f"Test metrics text file written to: {metrics_txt_path}")

    # 4. Extract true labels and predictions
    logger.info("Generating predictions for detailed metrics...")
    
    # Gather true labels
    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy().flatten())
    y_true = np.array(y_true)

    # Predict probabilities
    y_pred_probs = model.predict(test_ds, verbose=1).flatten()

    # 5. Print and save Classification Report
    y_pred = (y_pred_probs >= 0.5).astype(int)
    report = classification_report(
        y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"], digits=4
    )
    logger.info(f"\nClassification Report:\n{report}")

    report_path = os.path.join(MODEL_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Classification report saved to: {report_path}")

    # 6. Generate and save Plots
    generate_plots(y_true, y_pred_probs, MODEL_DIR)


if __name__ == "__main__":
    run_evaluation()
