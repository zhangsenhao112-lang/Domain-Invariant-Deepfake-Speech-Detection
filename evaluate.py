#!/usr/bin/env python3
"""Paper-compatible cross-dataset evaluation for the released CMSD methods.

This is the portable counterpart of the experimental ``evaluate/evaluate.py``:

1. obtain an operating threshold from the complete ASVspoof 2021 LA eval set;
2. apply that unchanged threshold to five out-of-domain datasets;
3. report Accuracy, F1, Recall, and Macro-F1;
4. report per-generator SONAR and per-codec CodecFake results.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


OOD_DATASETS = ("In-the-Wild", "FoR", "SONAR", "CodecFake", "ADD2023")
LEGACY_MACRO_F1_DATASETS = {"FoR", "SONAR", "CodecFake"}


@dataclass
class Metrics:
    eer: float
    eer_threshold: float
    auc: float
    far: float
    frr: float
    accuracy: float
    f1: float
    recall: float
    precision: float
    macro_f1: float
    legacy_macro_f1: float
    bona_count: int
    spoof_count: int


def read_protocol(path: Path) -> tuple[list[str], np.ndarray]:
    identifiers, labels = [], []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            identifier, label = line.split()
            if identifier in seen:
                raise ValueError(f"{path}:{line_number}: duplicate trial {identifier}")
            if label not in {"bonafide", "spoof"}:
                raise ValueError(f"{path}:{line_number}: invalid label {label}")
            seen.add(identifier)
            identifiers.append(identifier)
            labels.append(label == "bonafide")
    return identifiers, np.asarray(labels, dtype=bool)


def read_score_map(path: Path) -> dict[str, float]:
    scores = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            identifier, value = line.split()
            if identifier in scores:
                raise ValueError(f"{path}:{line_number}: duplicate trial {identifier}")
            scores[identifier] = float(value)
    return scores


def aligned_scores(
    score_path: Path, protocol_path: Path
) -> tuple[list[str], np.ndarray, np.ndarray]:
    identifiers, labels = read_protocol(protocol_path)
    score_map = read_score_map(score_path)
    protocol_ids = set(identifiers)
    missing = protocol_ids - score_map.keys()
    extra = score_map.keys() - protocol_ids
    if missing or extra:
        raise ValueError(
            f"{score_path}: protocol mismatch "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    return identifiers, np.asarray([score_map[key] for key in identifiers]), labels


def compute_eer(
    bonafide_scores: np.ndarray, spoof_scores: np.ndarray
) -> tuple[float, float]:
    """Match ``evaluate/eval_metrics.py::compute_eer`` exactly."""
    score_count = bonafide_scores.size + spoof_scores.size
    scores = np.concatenate((bonafide_scores, spoof_scores))
    labels = np.concatenate(
        (np.ones(bonafide_scores.size), np.zeros(spoof_scores.size))
    )
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    target_sums = np.cumsum(labels)
    nontarget_sums = spoof_scores.size - (
        np.arange(1, score_count + 1) - target_sums
    )
    frr = np.r_[0.0, target_sums / bonafide_scores.size]
    far = np.r_[1.0, nontarget_sums / spoof_scores.size]
    thresholds = np.r_[scores[order[0]] - 0.001, scores[order]]
    index = int(np.argmin(np.abs(frr - far)))
    return float((frr[index] + far[index]) / 2), float(thresholds[index])


def compute_auc(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    scores = np.concatenate((bonafide_scores, spoof_scores))
    labels = np.concatenate(
        (np.ones(bonafide_scores.size), np.zeros(spoof_scores.size))
    )
    order = np.argsort(scores, kind="stable")[::-1]
    scores, labels = scores[order], labels[order]
    group_ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), scores.size - 1]
    tpr = np.r_[0.0, np.cumsum(labels)[group_ends] / bonafide_scores.size]
    fpr = np.r_[
        0.0,
        np.cumsum(labels == 0)[group_ends] / spoof_scores.size,
    ]
    return float(np.trapz(tpr, fpr))


def compute_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Metrics:
    bonafide = scores[labels]
    spoof = scores[~labels]
    eer, eer_threshold = compute_eer(bonafide, spoof)
    predicted_bonafide = scores > threshold
    tp = int(np.sum(predicted_bonafide & labels))
    fp = int(np.sum(predicted_bonafide & ~labels))
    fn = int(np.sum(~predicted_bonafide & labels))
    tn = int(np.sum(~predicted_bonafide & ~labels))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    precision_negative = tn / max(tn + fn, 1)
    recall_negative = tn / max(tn + fp, 1)
    f1_negative = (
        2
        * precision_negative
        * recall_negative
        / max(precision_negative + recall_negative, 1e-12)
    )
    macro_f1 = (f1 + f1_negative) / 2

    # Three legacy dataset scripts mixed positive-class F1 in percent with
    # negative-class F1 in [0, 1]. Keep this value only for exact paper-code
    # compatibility; ``macro_f1`` above is the mathematically correct metric.
    legacy_macro_f1 = ((100 * f1) + f1_negative) / 2
    return Metrics(
        eer=100 * eer,
        eer_threshold=eer_threshold,
        auc=100 * compute_auc(bonafide, spoof),
        far=100 * fp / max(fp + tn, 1),
        frr=100 * fn / max(fn + tp, 1),
        accuracy=100 * (tp + tn) / labels.size,
        f1=100 * f1,
        recall=100 * recall,
        precision=100 * precision,
        macro_f1=100 * macro_f1,
        legacy_macro_f1=legacy_macro_f1,
        bona_count=bonafide.size,
        spoof_count=spoof.size,
    )


def la_operating_point(
    score_path: Path,
    protocol_path: Path,
    metadata_path: Path,
    la_subset: str,
    threshold_precision: int | None,
) -> tuple[float, float, Metrics]:
    if la_subset == "all":
        _, scores, labels = aligned_scores(score_path, protocol_path)
    else:
        score_map = read_score_map(score_path)
        selected_ids, selected_labels = [], []
        with metadata_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.split()
                if len(fields) < 8:
                    raise ValueError(
                        f"{metadata_path}:{line_number}: expected 8 columns"
                    )
                identifier, attack, label, phase = (
                    fields[1],
                    fields[4],
                    fields[5],
                    fields[7],
                )
                if phase == "eval" and (attack == "A19" or label == "bonafide"):
                    selected_ids.append(identifier)
                    selected_labels.append(label == "bonafide")
        missing = set(selected_ids) - score_map.keys()
        if missing:
            raise ValueError(f"{score_path}: missing {len(missing)} LA trials")
        scores = np.asarray([score_map[key] for key in selected_ids])
        labels = np.asarray(selected_labels, dtype=bool)
    _, raw_threshold = compute_eer(scores[labels], scores[~labels])
    applied_threshold = (
        raw_threshold
        if threshold_precision is None
        else float(f"{raw_threshold:.{threshold_precision}f}")
    )
    metrics = compute_metrics(scores, labels, applied_threshold)
    return raw_threshold, applied_threshold, metrics


def subset_metrics(
    dataset: str,
    identifiers: list[str],
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> list[tuple[str, Metrics]]:
    groups = np.asarray([identifier.split("/", 1)[0] for identifier in identifiers])
    rows = []
    if dataset == "CodecFake":
        names = sorted(set(groups[labels]) & set(groups[~labels]))
        for name in names:
            selected = groups == name
            rows.append((name, compute_metrics(scores[selected], labels[selected], threshold)))
    elif dataset == "SONAR":
        spoof_groups = sorted(set(groups[~labels]))
        bona_indices = np.flatnonzero(labels)
        random_state = np.random.RandomState(42)
        for name in spoof_groups:
            spoof_indices = np.flatnonzero((groups == name) & ~labels)
            if spoof_indices.size <= bona_indices.size:
                sampled_bona = random_state.choice(
                    bona_indices, size=spoof_indices.size, replace=False
                )
            else:
                sampled_bona = bona_indices
            selected = np.r_[sampled_bona, spoof_indices]
            rows.append((name, compute_metrics(scores[selected], labels[selected], threshold)))
            # pandas ``sample(random_state=42)`` resets its RNG for every method.
            random_state = np.random.RandomState(42)
    return rows


def reported_macro_f1(dataset: str, metrics: Metrics, mode: str) -> float:
    if mode == "paper" and dataset in LEGACY_MACRO_F1_DATASETS:
        return metrics.legacy_macro_f1
    return metrics.macro_f1


def evaluate_method(
    method: str,
    filename: str,
    score_root: Path,
    protocol_root: Path,
    la_subset: str,
    threshold_precision: int | None,
    metric_mode: str,
) -> dict:
    raw_threshold, threshold, la_metrics = la_operating_point(
        score_root / "ASVspoof2021_LA" / filename,
        protocol_root / "ASVspoof2021_LA" / "eval.txt",
        protocol_root / "ASVspoof2021_LA" / "trial_metadata.txt",
        la_subset,
        threshold_precision,
    )
    print(f"\n{'=' * 106}\n{method}")
    print(
        f"LA operating set: {'full 2021 LA eval' if la_subset == 'all' else 'A19 + bonafide legacy subset'} "
        f"({la_metrics.bona_count + la_metrics.spoof_count} trials)\n"
        f"LA EER: {la_metrics.eer:.2f}% | raw threshold: {raw_threshold:.10f} "
        f"| applied threshold: {threshold:.10f}"
    )
    print(
        f"{'Dataset':<16} {'EER%':>8} {'AUC%':>8} {'FAR%':>8} {'FRR%':>8} "
        f"{'Acc%':>8} {'F1%':>8} {'Recall%':>9} {'MacroF1%':>10}"
    )

    dataset_results, subset_results = {}, {}
    for dataset in OOD_DATASETS:
        identifiers, scores, labels = aligned_scores(
            score_root / dataset / filename,
            protocol_root / dataset / "eval.txt",
        )
        metrics = compute_metrics(scores, labels, threshold)
        dataset_results[dataset] = metrics
        macro_f1 = reported_macro_f1(dataset, metrics, metric_mode)
        print(
            f"{dataset:<16} {metrics.eer:>8.2f} {metrics.auc:>8.2f} "
            f"{metrics.far:>8.2f} {metrics.frr:>8.2f} {metrics.accuracy:>8.2f} "
            f"{metrics.f1:>8.2f} {metrics.recall:>9.2f} {macro_f1:>10.2f}"
        )
        if dataset in {"SONAR", "CodecFake"}:
            subset_results[dataset] = subset_metrics(
                dataset, identifiers, scores, labels, threshold
            )

    averages = {
        key: float(
            np.mean(
                [
                    reported_macro_f1(dataset, metrics, metric_mode)
                    if key == "macro_f1"
                    else getattr(metrics, key)
                    for dataset, metrics in dataset_results.items()
                ]
            )
        )
        for key in ("accuracy", "f1", "recall", "macro_f1")
    }
    print(
        f"{'OOD AVERAGE':<16} {'--':>8} {'--':>8} {'--':>8} {'--':>8} "
        f"{averages['accuracy']:>8.2f} {averages['f1']:>8.2f} "
        f"{averages['recall']:>9.2f} {averages['macro_f1']:>10.2f}"
    )
    return {
        "method": method,
        "threshold": threshold,
        "metrics": dataset_results,
        "subsets": subset_results,
        "averages": averages,
    }


def print_summary(results: list[dict], metric_mode: str) -> None:
    print(f"\n\n{'=' * 94}")
    print(f"Cross-dataset summary (Macro-F1 mode: {metric_mode})")
    print(f"{'=' * 94}")
    print(
        f"{'Method':<24} {'Threshold':>12} {'Acc%':>10} {'F1%':>10} "
        f"{'Recall%':>10} {'MacroF1%':>12}"
    )
    for result in results:
        averages = result["averages"]
        print(
            f"{result['method']:<24} {result['threshold']:>12.4f} "
            f"{averages['accuracy']:>10.2f} {averages['f1']:>10.2f} "
            f"{averages['recall']:>10.2f} {averages['macro_f1']:>12.2f}"
        )

    for dataset in ("SONAR", "CodecFake"):
        names = sorted(
            {
                name
                for result in results
                for name, _ in result["subsets"].get(dataset, [])
            }
        )
        print(f"\n{dataset} per-subset balanced accuracy")
        print(f"{'Method':<24}" + "".join(f"{name:>16}" for name in names))
        for result in results:
            lookup = {
                name: metrics.accuracy
                for name, metrics in result["subsets"].get(dataset, [])
            }
            print(
                f"{result['method']:<24}"
                + "".join(f"{lookup.get(name, float('nan')):>15.2f}%" for name in names)
            )


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate the released CMSD methods using paper scoring logic"
    )
    parser.add_argument(
        "--score-name",
        action="append",
        required=True,
        help=(
            "Score filename present under every scores/<dataset>/ directory; "
            "repeat to compare multiple models"
        ),
    )
    parser.add_argument(
        "--la-subset",
        choices=("all", "a19"),
        default="all",
        help=(
            "LA trials used to derive the threshold. Default: complete 2021 LA; "
            "'a19' reproduces the legacy la_evaluate.py filter."
        ),
    )
    parser.add_argument(
        "--metric-mode",
        choices=("paper", "corrected"),
        default="paper",
        help=(
            "'paper' reproduces legacy Macro-F1 output; 'corrected' uses "
            "consistent class-wise units"
        ),
    )
    parser.add_argument(
        "--legacy-threshold-rounding",
        action="store_true",
        help="Round the LA threshold to 2 decimals as in the old subprocess parser",
    )
    parser.add_argument("--score-root", type=Path, default=root / "scores")
    parser.add_argument("--protocol-root", type=Path, default=root / "protocols")
    args = parser.parse_args()

    threshold_precision = 2 if args.legacy_threshold_rounding else None
    results = [
        evaluate_method(
            Path(filename).stem,
            filename,
            args.score_root,
            args.protocol_root,
            args.la_subset,
            threshold_precision,
            args.metric_mode,
        )
        for filename in args.score_name
    ]
    print_summary(results, args.metric_mode)


if __name__ == "__main__":
    main()
