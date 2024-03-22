import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)


def plot_confusion_matrix(y_true, y_pred_class, labels, normalize=False):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred_class, normalize=normalize)

    # Create a confusion matrix display
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Plot the confusion matrix
    cm_display.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def evaluate_metrics(y_test, y_pred, thres=None, labels=None, normalize="all"):

    # Compute and print the ROC AUC score
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    # Compute and print the AUC for the precision-recall curve
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc}")

    # Compute F1 scores for different thresholds
    f1_scores = 2 * precision * recall / (precision + recall)

    f1_scores = np.nan_to_num(f1_scores, nan=-np.inf)

    # Get the best threshold
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index] if thres is None else thres
    best_score = f1_scores[best_index]

    print("classification_report")
    print(classification_report(y_test, y_pred > best_threshold))
    print()

    print(f"Best F1-score: {best_score} at threshold {best_threshold}")

    # Plot the precision-recall curve
    plt.plot(recall, precision)
    plt.scatter(
        recall[best_index], precision[best_index], color="red"
    )  # mark best point
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.show()

    # Confusion matrix at the best threshold
    plot_confusion_matrix(y_test, y_pred > best_threshold, labels, normalize)
    return best_threshold
