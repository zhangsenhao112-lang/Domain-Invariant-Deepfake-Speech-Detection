#!/usr/bin/env python3
"""Compute EER and fixed-threshold binary metrics from CMSD scores."""

import argparse

import numpy as np

from cmsd.data import read_protocol


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels == 1)
    false_positive = np.cumsum(sorted_labels == 0)
    positives = (labels == 1).sum()
    negatives = (labels == 0).sum()
    false_negative_rate = 1.0 - true_positive / positives
    false_positive_rate = false_positive / negatives
    index = np.argmin(np.abs(false_negative_rate - false_positive_rate))
    return (
        float((false_negative_rate[index] + false_positive_rate[index]) / 2),
        float(scores[order[index]]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-format", choices=("simple", "asvspoof"), default="simple")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Bonafide threshold from source-domain development data",
    )
    args = parser.parse_args()

    labels = {
        trial.path: trial.label
        for trial in read_protocol(args.protocol, args.protocol_format)
    }
    scored = []
    with open(args.scores, encoding="utf-8") as handle:
        for line in handle:
            identifier, score = line.split()
            if identifier not in labels or labels[identifier] is None:
                raise ValueError(f"No label found for '{identifier}'")
            scored.append((labels[identifier], float(score)))
    y_true = np.asarray([item[0] for item in scored])
    y_score = np.asarray([item[1] for item in scored])
    if not (np.any(y_true == 0) and np.any(y_true == 1)):
        raise ValueError("Both bonafide and spoof trials are required")

    eer, eer_threshold = equal_error_rate(y_true, y_score)
    threshold = eer_threshold if args.threshold is None else args.threshold
    predicted = (y_score >= threshold).astype(int)
    tp = np.sum((predicted == 1) & (y_true == 1))
    fp = np.sum((predicted == 1) & (y_true == 0))
    fn = np.sum((predicted == 0) & (y_true == 1))
    accuracy = np.mean(predicted == y_true)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
    print(f"EER: {100 * eer:.4f}%")
    print(f"EER threshold: {eer_threshold:.10f}")
    print(f"Fixed threshold: {threshold:.10f}")
    print(f"Accuracy: {100 * accuracy:.4f}%")
    print(f"Precision: {100 * precision:.4f}%")
    print(f"Recall: {100 * recall:.4f}%")
    print(f"F1: {100 * f1:.4f}%")


if __name__ == "__main__":
    main()
