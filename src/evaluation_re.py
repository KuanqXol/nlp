"""
evaluation_re.py
────────────────
Đánh giá Relation Extraction trên benchmark annotate tay và chạy ablation:

- rule_based:   RelationExtractor
- phobert:      PhoBERTRelationExtractor (nếu có model)
- hybrid:       HybridRelationExtractor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.preprocessing.relation_extraction import R, RelationExtractor
from src.preprocessing.relation_extraction_phobert import HybridRelationExtractor

DEFAULT_GROUND_TRUTH = (
    Path(__file__).resolve().parents[1] / "data" / "relation_ground_truth.json"
)
SUPPORTED_RELATIONS = (
    R.LEADS,
    R.MEMBER_OF,
    R.APPOINTED,
    R.COOPERATES,
    R.SIGNS_DEAL,
    R.LOCATED_IN,
    R.INVESTS_IN,
    R.ACQUIRES,
    R.FOUNDED,
    R.PRODUCES,
    R.ATTACKS,
    R.SUPPORTS,
    R.SANCTIONS,
    R.MEETS,
    R.WARNS_ABOUT,
    R.FOUND_IN,
)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _normalize_triple_key(subject: str, relation: str, obj: str) -> Tuple[str, str, str]:
    subj_key = _normalize_text(subject)
    obj_key = _normalize_text(obj)
    rel_key = _normalize_text(relation)
    if rel_key in R.SYMMETRIC:
        ordered = tuple(sorted([subj_key, obj_key]))
        return ordered[0], rel_key, ordered[1]
    return subj_key, rel_key, obj_key


def _triple_set(triples: Iterable[Dict]) -> Set[Tuple[str, str, str]]:
    result: Set[Tuple[str, str, str]] = set()
    for triple in triples:
        subject = triple.get("subject", "")
        relation = triple.get("relation", "")
        obj = triple.get("object", "")
        if subject and relation and obj:
            result.add(_normalize_triple_key(subject, relation, obj))
    return result


def load_relation_ground_truth(path: str | Path = DEFAULT_GROUND_TRUTH) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("relation_ground_truth.json phải có key 'samples' là list.")
    return samples


def sample_to_document(sample: Dict) -> Dict:
    sentence = sample.get("sentence", "").strip()
    return {
        "id": sample.get("id", ""),
        "date": sample.get("date", ""),
        "category": sample.get("category", ""),
        "title": sample.get("title", sentence),
        "content": sentence,
        "full_text": sentence,
        "linked_entities": sample.get("linked_entities", []),
    }


def _run_extractor(extractor, document: Dict) -> List[Dict]:
    if hasattr(extractor, "process_document"):
        processed = extractor.process_document(document)
        return processed.get("triples", [])
    if hasattr(extractor, "extract_from_document"):
        return extractor.extract_from_document(document)
    if hasattr(extractor, "extract"):
        return extractor.extract(document)
    raise TypeError(f"Extractor không được hỗ trợ: {type(extractor)}")


def _safe_ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def _compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float | int]:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_relation_extractor(
    extractor,
    ground_truth: str | Path = DEFAULT_GROUND_TRUTH,
    verbose: bool = True,
) -> Dict[str, Dict[str, float | int]]:
    samples = load_relation_ground_truth(ground_truth)
    counts = {
        relation: {"tp": 0, "fp": 0, "fn": 0}
        for relation in SUPPORTED_RELATIONS
    }
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    for sample in samples:
        gold_set = _triple_set(sample.get("triples", []))
        pred_set = _triple_set(_run_extractor(extractor, sample_to_document(sample)))

        gold_by_relation = {relation: set() for relation in SUPPORTED_RELATIONS}
        pred_by_relation = {relation: set() for relation in SUPPORTED_RELATIONS}

        for triple in gold_set:
            if triple[1] in gold_by_relation:
                gold_by_relation[triple[1]].add(triple)
        for triple in pred_set:
            if triple[1] in pred_by_relation:
                pred_by_relation[triple[1]].add(triple)

        for relation in SUPPORTED_RELATIONS:
            gold_rel = gold_by_relation[relation]
            pred_rel = pred_by_relation[relation]
            tp = len(gold_rel & pred_rel)
            fp = len(pred_rel - gold_rel)
            fn = len(gold_rel - pred_rel)
            counts[relation]["tp"] += tp
            counts[relation]["fp"] += fp
            counts[relation]["fn"] += fn

        micro_tp += len(gold_set & pred_set)
        micro_fp += len(pred_set - gold_set)
        micro_fn += len(gold_set - pred_set)

    metrics = {
        relation: _compute_prf(values["tp"], values["fp"], values["fn"])
        for relation, values in counts.items()
    }
    metrics["micro_avg"] = _compute_prf(micro_tp, micro_fp, micro_fn)
    macro_relations = [metrics[relation]["f1"] for relation in SUPPORTED_RELATIONS]
    metrics["macro_avg"] = {
        "precision": sum(metrics[relation]["precision"] for relation in SUPPORTED_RELATIONS)
        / len(SUPPORTED_RELATIONS),
        "recall": sum(metrics[relation]["recall"] for relation in SUPPORTED_RELATIONS)
        / len(SUPPORTED_RELATIONS),
        "f1": sum(macro_relations) / len(macro_relations),
        "tp": micro_tp,
        "fp": micro_fp,
        "fn": micro_fn,
    }

    if verbose:
        print("[RE] Precision / Recall / F1")
        for relation in [*SUPPORTED_RELATIONS, "micro_avg", "macro_avg"]:
            score = metrics[relation]
            print(
                f"  {relation:16s} "
                f"P={score['precision']:.3f} "
                f"R={score['recall']:.3f} "
                f"F1={score['f1']:.3f} "
                f"(tp={score['tp']}, fp={score['fp']}, fn={score['fn']})"
            )

    return metrics


def run_re_ablation(
    ground_truth: str | Path = DEFAULT_GROUND_TRUTH,
    phobert_dir: str | Path | None = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}

    if verbose:
        print("[Ablation] rule-based")
    rule_metrics = evaluate_relation_extractor(
        RelationExtractor(),
        ground_truth=ground_truth,
        verbose=verbose,
    )
    results["rule_based"] = {
        "status": "ok",
        "metrics": rule_metrics,
    }

    if verbose:
        print("[Ablation] hybrid")
    hybrid_metrics = evaluate_relation_extractor(
        HybridRelationExtractor(phobert_dir=str(phobert_dir) if phobert_dir else None),
        ground_truth=ground_truth,
        verbose=verbose,
    )
    results["hybrid"] = {
        "status": "ok",
        "metrics": hybrid_metrics,
    }

    phobert_path = Path(phobert_dir) if phobert_dir else None
    if not phobert_path or not phobert_path.exists():
        results["phobert"] = {
            "status": "skipped",
            "reason": "Model directory không tồn tại.",
        }
        if verbose:
            print("[Ablation] phobert skipped: missing model directory")
        return results

    try:
        from src.preprocessing.relation_extraction_phobert import PhoBERTRelationExtractor

        phobert = PhoBERTRelationExtractor()
        phobert.load(str(phobert_path))
        if verbose:
            print("[Ablation] phobert")
        phobert_metrics = evaluate_relation_extractor(
            phobert,
            ground_truth=ground_truth,
            verbose=verbose,
        )
        results["phobert"] = {
            "status": "ok",
            "metrics": phobert_metrics,
        }
    except Exception as exc:
        results["phobert"] = {
            "status": "skipped",
            "reason": str(exc),
        }
        if verbose:
            print(f"[Ablation] phobert skipped: {exc}")

    return results


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Evaluate RE benchmark và chạy ablation rule-based vs PhoBERT vs hybrid."
    )
    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH),
        help="Path tới relation_ground_truth.json",
    )
    parser.add_argument(
        "--phobert-dir",
        default="data/phobert_re",
        help="Thư mục model PhoBERT RE đã fine-tune",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Giảm log chi tiết",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    results = run_re_ablation(
        ground_truth=args.ground_truth,
        phobert_dir=args.phobert_dir,
        verbose=not args.quiet,
    )

    print("\n[Ablation Summary]")
    for system_name in ("rule_based", "hybrid", "phobert"):
        outcome = results.get(system_name, {})
        status = outcome.get("status", "unknown")
        if status != "ok":
            print(f"  - {system_name:10s}: {status} ({outcome.get('reason', 'n/a')})")
            continue
        micro = outcome["metrics"]["micro_avg"]
        print(
            f"  - {system_name:10s}: "
            f"P={micro['precision']:.3f} R={micro['recall']:.3f} F1={micro['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
