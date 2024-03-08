import logging
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay, f1_score
import matplotlib.pyplot as plt


def plot_roc_curve(y_true, y_pred, model_name):
    """
    Plots ROC curve for model.

    Args:
        y_true: true labels. array-like of shape (n_samples,)
        y_pred: prediction. array-like of shape (n_samples,)
        model_name: name of the Model string

    Returns: none
    """
    y_pred = y_pred.reshape(-1)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name)
    display.plot()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plots confusion matrix for model and prints F1 score, precision, recall

    Args:
        y_true: true labels. array-like of shape (n_samples,)
        y_pred: prediction. array-like of shape (n_samples,)
        classes: estimator.classes_

    Returns: none
    """
    y_pred = y_pred.reshape(-1)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    logging.info(f"tp = {tp}, fp = {fp}, fn = {fn}, tn = {tn}")
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score_cal = 2 * (precision * recall) / (precision + recall)

    f1_ = f1_score(y_true, y_pred)
    logging.info(
        f"precision = {precision:.4f}, "
        f"recall = {recall:.4f}, "
        f"f1_score_cal = {f1_score_cal:.4f}, "
        f"f1 score = {f1_:.4f}"
    )


def get_score(y_true, y_pred):
    """
    Returns accuracy score for model.

    Args:
        y_true: true labels. array-like of shape (n_samples,)
        y_pred: prediction. array-like of shape (n_samples,)

    Returns:
        score float
    """
    y_pred = y_pred.reshape(-1)
    score = np.mean(y_pred == y_true)
    logging.info(f"Accuracy Score = {score}")

    return score
