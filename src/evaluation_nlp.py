"""Utilities to evaluate Vietnamese NER on a small hand-labeled benchmark."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.preprocessing.ner import VietnameseNER

TARGET_ENTITY_TYPES = ("PER", "LOC", "ORG")
DEFAULT_GROUND_TRUTH = Path(__file__).resolve().parents[1] / "data" / "ner_ground_truth.json"


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _token_len(text: str) -> int:
    return len(_normalize_text(text).split())


def _soft_entity_match(pred_text: str, truth_text: str) -> bool:
    pred_norm = _normalize_text(pred_text)
    truth_norm = _normalize_text(truth_text)

    if not pred_norm or not truth_norm:
        return False
    if pred_norm == truth_norm:
        return True

    shorter, longer = (
        (pred_norm, truth_norm)
        if _token_len(pred_norm) <= _token_len(truth_norm)
        else (truth_norm, pred_norm)
    )

    if shorter in longer and _token_len(shorter) >= 2:
        return True

    shorter_tokens = shorter.split()
    longer_tokens = longer.split()
    if len(shorter_tokens) == 1 and shorter_tokens[0] in longer_tokens:
        token = shorter_tokens[0]
        if len(token) <= 6:
            return True

    return False


def _unique_entities(entities: Iterable[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for entity in entities:
        ent_type = entity.get("type", "")
        ent_text = entity.get("text") or entity.get("entity_text") or ""
        key = (ent_type, _normalize_text(ent_text))
        if not ent_type or not ent_text or key in seen:
            continue
        seen.add(key)
        unique.append({"text": ent_text, "type": ent_type})
    return unique


def _match_entities(
    predicted: Sequence[Dict],
    truth: Sequence[Dict],
) -> Tuple[int, int, int]:
    matched_truth = set()
    tp = 0
    fp = 0

    for pred in predicted:
        found = None
        for idx, gold in enumerate(truth):
            if idx in matched_truth:
                continue
            if pred.get("type") != gold.get("type"):
                continue
            if _soft_entity_match(pred.get("text", ""), gold.get("text", "")):
                found = idx
                break

        if found is not None:
            matched_truth.add(found)
            tp += 1
        else:
            fp += 1

    fn = len(truth) - len(matched_truth)
    return tp, fp, fn


def load_ground_truth(path: str | Path = DEFAULT_GROUND_TRUTH) -> Dict:
    target = Path(path)
    with open(target, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return {"metadata": {}, "samples": payload}
    return payload


def evaluate_ner(
    ground_truth_path: str | Path = DEFAULT_GROUND_TRUTH,
    ner: VietnameseNER | None = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    payload = load_ground_truth(ground_truth_path)
    samples = payload.get("samples", [])
    ner = ner or VietnameseNER(use_model=False)

    counts = {
        ent_type: {"tp": 0, "fp": 0, "fn": 0}
        for ent_type in TARGET_ENTITY_TYPES
    }

    for sample in samples:
        sentence = sample.get("sentence", "")
        truth_entities = _unique_entities(sample.get("entities", []))
        predicted_entities = _unique_entities(
            [
                entity
                for entity in ner.extract(sentence)
                if entity.get("type") in TARGET_ENTITY_TYPES
            ]
        )

        for ent_type in TARGET_ENTITY_TYPES:
            truth_by_type = [entity for entity in truth_entities if entity["type"] == ent_type]
            pred_by_type = [entity for entity in predicted_entities if entity["type"] == ent_type]
            tp, fp, fn = _match_entities(pred_by_type, truth_by_type)
            counts[ent_type]["tp"] += tp
            counts[ent_type]["fp"] += fp
            counts[ent_type]["fn"] += fn

    metrics: Dict[str, Dict[str, float]] = {}
    micro_tp = micro_fp = micro_fn = 0

    for ent_type in TARGET_ENTITY_TYPES:
        tp = counts[ent_type]["tp"]
        fp = counts[ent_type]["fp"]
        fn = counts[ent_type]["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        metrics[ent_type] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    metrics["micro_avg"] = {
        "tp": micro_tp,
        "fp": micro_fp,
        "fn": micro_fn,
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
        "support": micro_tp + micro_fn,
    }

    if verbose:
        print(f"[NER Eval] Ground truth: {ground_truth_path}")
        print(f"[NER Eval] Samples: {len(samples)}")
        print()
        header = f"{'Type':<10}{'TP':>6}{'FP':>6}{'FN':>6}{'Prec':>10}{'Rec':>10}{'F1':>10}{'Support':>10}"
        print(header)
        print("-" * len(header))
        for ent_type in (*TARGET_ENTITY_TYPES, "micro_avg"):
            row = metrics[ent_type]
            print(
                f"{ent_type:<10}"
                f"{row['tp']:>6}"
                f"{row['fp']:>6}"
                f"{row['fn']:>6}"
                f"{row['precision']:>10.3f}"
                f"{row['recall']:>10.3f}"
                f"{row['f1']:>10.3f}"
                f"{row['support']:>10}"
            )

    return metrics


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese NER on a hand-labeled benchmark")
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=str(DEFAULT_GROUND_TRUTH),
        help="Path to the NER ground truth JSON file",
    )
    parser.add_argument(
        "--use-model",
        action="store_true",
        help="Use the transformer NER backend if available",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    ner = VietnameseNER(use_model=args.use_model)
    evaluate_ner(args.ground_truth, ner=ner, verbose=True)


if __name__ == "__main__":
    main()
