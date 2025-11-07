from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from skimage.metrics import structural_similarity
import numpy as np


def calculate_confusion_matrix(y_true, y_pred, subjects):
    """
    Calculates a detailed confusion matrix and returns a dictionary with
    lists of subjects corresponding to each category (TP, FP, TN, FN).
    """
    confusion_matrix_dict = {
        "true_positive": [],
        "false_positive": [],
        "true_negative": [],
        "false_negative": [],
    }

    for subject, true_label, predicted_label in zip(subjects, y_true, y_pred):
        if true_label == 1 and predicted_label == 1:
            confusion_matrix_dict["true_positive"].append(subject)
        elif true_label == 1 and predicted_label == 0:
            confusion_matrix_dict["false_negative"].append(subject)
        elif true_label == 0 and predicted_label == 0:
            confusion_matrix_dict["true_negative"].append(subject)
        elif true_label == 0 and predicted_label == 1:
            confusion_matrix_dict["false_positive"].append(subject)

    return confusion_matrix_dict


def compute_classification_metrics(group_data, groups_subject_ids, invert=False):
    """
    Computes classification metrics for two groups of data.
    Assumes that group_data is a list of two lists, each containing the scores
    for the respective group.
    The function also assumes that the first group is the positive class and
    the second group is the negative class.
    If invert is True, the groups are swapped.
    """

    if len(group_data) != 2:
        raise ValueError("group_data must contain exactly two groups of data.")
    if len(groups_subject_ids) != 2:
        raise ValueError(
            "groups_subject_ids must contain exactly two groups of subject IDs."
        )

    if invert:
        # Swap the groups if invert is True
        group_data[0], group_data[1] = group_data[1], group_data[0]
        groups_subject_ids[0], groups_subject_ids[1] = (
            groups_subject_ids[1],
            groups_subject_ids[0],
        )
    y_true = [0] * len(group_data[0]) + [1] * len(group_data[1])

    y_scores = group_data[0] + group_data[1]
    subject_ids = groups_subject_ids[0] + groups_subject_ids[1]

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    cutoff = thresholds[np.argmax(tpr - fpr)]

    y_pred = [int(score > cutoff) for score in y_scores]
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    confusion_matrix = calculate_confusion_matrix(y_true, y_pred, subject_ids)

    return {
        "fpr": list(fpr),
        "tpr": list(tpr),
        "cutoff": cutoff,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "f1": f1,
        "y_true": y_true,
        "y_scores": y_scores,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix,
    }


def masked_ssim(image1, image2, mask, **kwargs):
    """
    Compute SSIM only inside a masked region.
    The mask should be a boolean array (True = include pixel).
    """
    _, ssim_map = structural_similarity(image1, image2, full=True, **kwargs)
    masked_ssim_map = ssim_map * mask

    valid_pixels = np.sum(mask)
    if valid_pixels == 0:
        return 0.0

    masked_score = np.sum(masked_ssim_map) / valid_pixels
    return masked_score
