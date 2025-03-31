"""
Evaluation metrics for PDF information extraction
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for token classification

    Args:
        eval_pred: Tuple of (predictions, labels)

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Get the model's label list (assuming it's part of trainer.model.config)
    try:
        label_list = [v for k, v in sorted(eval_pred.model.config.id2label.items())]
    except:
        # Fallback to a standard label list
        label_list = [
            "O",
            "B-HEADER",
            "I-HEADER",
            "B-QUESTION",
            "I-QUESTION",
            "B-ANSWER",
            "I-ANSWER",
        ]

    # Remove ignored index (special tokens with -100)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute metrics
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }


def accuracy_score(true_labels: List[List[str]], predictions: List[List[str]]) -> float:
    """
    Compute token-level accuracy

    Args:
        true_labels: List of lists of true labels
        predictions: List of lists of predicted labels

    Returns:
        Accuracy score
    """
    correct = 0
    total = 0

    for true_seq, pred_seq in zip(true_labels, predictions):
        for true_label, pred_label in zip(true_seq, pred_seq):
            total += 1
            if true_label == pred_label:
                correct += 1

    return correct / total if total > 0 else 0.0


def evaluate_extraction(
    true_info: Dict[str, List[str]], pred_info: Dict[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate structured information extraction

    Args:
        true_info: Dictionary mapping labels to lists of true text
        pred_info: Dictionary mapping labels to lists of predicted text

    Returns:
        Dictionary of metrics for each label
    """
    results = {}

    # Evaluate each label
    for label in true_info:
        if label not in pred_info:
            results[label] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue

        true_set = set(true_info[label])
        pred_set = set(pred_info[label])

        # Calculate metrics
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[label] = {"precision": precision, "recall": recall, "f1": f1}

    # Calculate overall metrics
    all_true = [item for items in true_info.values() for item in items]
    all_pred = [item for items in pred_info.values() for item in items]

    true_set = set(all_true)
    pred_set = set(all_pred)

    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    results["overall"] = {"precision": precision, "recall": recall, "f1": f1}

    return results
