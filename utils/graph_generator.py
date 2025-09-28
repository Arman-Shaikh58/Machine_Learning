# utils_plotting.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

def generate_plot_train_and_test_loss(train_losses, test_losses, num_epochs, folder_name=None):
    """Plot training vs test loss and optionally save, returns figure."""
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(range(1, num_epochs+1), train_losses, label="Training Loss")
    ax.plot(range(1, num_epochs+1), test_losses, label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Test Loss")
    ax.legend()
    if folder_name:
        plt.savefig(f"{folder_name}/train_vs_test_loss.png")
    return fig

def generate_plot_train_and_test_accuracy(train_accuracies, test_accuracies, num_epochs, folder_name=None):
    """Plot training vs test accuracy and optionally save, returns figure."""
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(range(1, num_epochs+1), train_accuracies, label="Training Accuracy")
    ax.plot(range(1, num_epochs+1), test_accuracies, label="Test Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training vs Test Accuracy")
    ax.legend()
    if folder_name:
        plt.savefig(f"{folder_name}/train_vs_test_accuracy.png")
    return fig

def generate_confusion_matrix(y_true, y_pred, folder_name=None, class_names=None):
    """Generate confusion matrix plot and optionally save, returns figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    if folder_name:
        plt.savefig(f"{folder_name}/confusion_matrix.png")
    return fig

def generate_roc_curve(y_true, y_scores, folder_name=None, num_classes=2):
    """Generate ROC curve plot and optionally save, returns figure."""
    fig, ax = plt.subplots(figsize=(8,6))
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_scores[:,i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'Class {i} ROC (area = {roc_auc:.2f})')
        ax.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Multi-class ROC Curve')
        ax.legend(loc='lower right')
    if folder_name:
        plt.savefig(f"{folder_name}/roc_curve.png")
    return fig
