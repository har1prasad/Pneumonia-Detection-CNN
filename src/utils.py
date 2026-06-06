"""
Pneumonia Detection - Helper Utilities
Description: Centralized helper utilities for logging initialization, directory
             creation, GPU diagnostics, and visualization functions.
"""

import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc

logger = logging.getLogger(__name__)


def setup_logger(name, log_file="project.log", level=logging.INFO):
    """
    Sets up a logger that outputs to both console and a log file.
    """
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    # Setup handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    
    # Configure logger
    proj_logger = logging.getLogger(name)
    proj_logger.setLevel(level)
    
    # Prevent duplicate handlers
    if not proj_logger.handlers:
        proj_logger.addHandler(console_handler)
        proj_logger.addHandler(file_handler)
        
    return proj_logger


def verify_gpu_status():
    """
    Checks and returns GPU status details.
    """
    gpus = tf.config.list_physical_devices("GPU")
    status_info = {
        "gpu_available": len(gpus) > 0,
        "gpu_count": len(gpus),
        "devices": [gpu.name for gpu in gpus]
    }
    return status_info


def ensure_directories_exist(directories_list):
    """
    Ensures that a list of directories exists.
    """
    for directory in directories_list:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")


def plot_and_save_confusion_matrix(y_true, y_pred, output_path, class_names=None):
    """
    Utility to plot and save a confusion matrix.
    """
    if class_names is None:
        class_names = ["NORMAL (0)", "PNEUMONIA (1)"]
        
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix plot saved to: {output_path}")


def plot_and_save_roc_curve(y_true, y_pred_probs, output_path):
    """
    Utility to plot and save an ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"ROC curve plot saved to: {output_path}")


if __name__ == "__main__":
    # Test utilities run
    log = setup_logger("test_logger", "test_run.log")
    log.info("Testing utility helper module...")
    gpu_info = verify_gpu_status()
    log.info(f"GPU Info: {gpu_info}")
