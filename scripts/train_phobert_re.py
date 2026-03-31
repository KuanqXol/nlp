"""
train_phobert_re.py
───────────────────
Chuẩn bị dữ liệu supervised nhỏ cho PhoBERT RE và fine-tune model.

Ví dụ:
    python scripts/train_phobert_re.py --prepare-only
    python scripts/train_phobert_re.py --output-dir data/phobert_re_demo
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation_re import DEFAULT_GROUND_TRUTH, load_relation_ground_truth
from src.preprocessing.relation_extraction import R
from src.preprocessing.relation_extraction_phobert import (
    LABEL2ID,
    NO_RELATION_ID,
    PhoBERTRelationExtractor,
    RELATION_LABELS,
)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _triple_key(subject: str, relation: str, obj: str) -> Tuple[str, str, str]:
    subj_key = _normalize_text(subject)
    obj_key = _normalize_text(obj)
    rel_key = _normalize_text(relation)
    if rel_key in R.SYMMETRIC:
        ordered = tuple(sorted([subj_key, obj_key]))
        return ordered[0], rel_key, ordered[1]
    return subj_key, rel_key, obj_key


def _entity_lookup(sample: Dict) -> Dict[str, Dict]:
    lookup = {}
    for entity in sample.get("linked_entities", []):
        canonical = entity.get("canonical") or entity.get("text")
        if canonical:
            lookup[_normalize_text(canonical)] = entity
    return lookup


def build_supervised_samples(samples: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    seen = set()

    for sample in samples:
        sentence = sample.get("sentence", "").strip()
        category = sample.get("category", "")
        entities = sample.get("linked_entities", [])
        entity_lookup = _entity_lookup(sample)
        gold_triples = sample.get("triples", [])
        gold_keys = {
            _triple_key(t.get("subject", ""), t.get("relation", ""), t.get("object", ""))
            for t in gold_triples
        }
        positive_pairs = set()
        for triple in gold_triples:
            subject = _normalize_text(triple.get("subject", ""))
            obj = _normalize_text(triple.get("object", ""))
            relation = _normalize_text(triple.get("relation", ""))
            if not subject or not obj:
                continue
            positive_pairs.add((subject, obj))
            if relation in R.SYMMETRIC:
                positive_pairs.add((obj, subject))

        def _append(entity1: str, entity2: str, relation: str, source: str):
            key = (sentence, entity1, entity2, relation)
            if key in seen:
                return

            e1 = entity_lookup.get(_normalize_text(entity1), {})
            e2 = entity_lookup.get(_normalize_text(entity2), {})
            rows.append(
                {
                    "sentence": sentence,
                    "entity1": entity1,
                    "entity2": entity2,
                    "e1_type": e1.get("type", "MISC"),
                    "e2_type": e2.get("type", "MISC"),
                    "relation": relation,
                    "label_id": LABEL2ID.get(relation, NO_RELATION_ID),
                    "category": category,
                    "source": source,
                    "sample_id": sample.get("id", ""),
                }
            )
            seen.add(key)

        for triple in gold_triples:
            subject = triple.get("subject", "")
            relation = triple.get("relation", "")
            obj = triple.get("object", "")
            if relation not in LABEL2ID:
                continue
            _append(subject, obj, relation, "manual_positive")
            if relation in R.SYMMETRIC:
                _append(obj, subject, relation, "manual_positive_symmetric")
            elif _triple_key(obj, relation, subject) not in gold_keys:
                _append(obj, subject, "no_relation", "manual_reverse_negative")

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1 = entities[i].get("canonical") or entities[i].get("text", "")
                e2 = entities[j].get("canonical") or entities[j].get("text", "")
                if not e1 or not e2 or e1 == e2:
                    continue
                if (_normalize_text(e1), _normalize_text(e2)) not in positive_pairs:
                    _append(e1, e2, "no_relation", "manual_pair_negative")
                if (_normalize_text(e2), _normalize_text(e1)) not in positive_pairs:
                    _append(e2, e1, "no_relation", "manual_pair_negative")

    return rows


def write_jsonl(path: str | Path, rows: List[Dict]):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Dict):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_jsonl(path: str | Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_dataset(rows: List[Dict], source_path: str) -> Dict:
    relation_counts = Counter(row["relation"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    return {
        "source_path": source_path,
        "total_rows": len(rows),
        "relation_counts": dict(sorted(relation_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "labels": RELATION_LABELS,
    }


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Prepare supervised dataset và fine-tune PhoBERT RE."
    )
    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH),
        help="Path tới relation_ground_truth.json dùng để build supervised dataset",
    )
    parser.add_argument(
        "--train-jsonl",
        default="",
        help="Dùng sẵn JSONL training data thay vì build từ ground truth",
    )
    parser.add_argument(
        "--prepared-jsonl",
        default="data/re_train_supervised.jsonl",
        help="Path lưu JSONL sau khi chuẩn bị",
    )
    parser.add_argument(
        "--output-dir",
        default="data/phobert_re",
        help="Thư mục lưu model và artifact training",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Chỉ build dataset, không train model",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    random.seed(args.seed)

    if args.train_jsonl:
        train_jsonl = Path(args.train_jsonl)
        if not train_jsonl.exists():
            raise FileNotFoundError(f"Không tìm thấy train_jsonl: {train_jsonl}")
        prepared_rows = load_jsonl(train_jsonl)
        source_path = str(train_jsonl)
    else:
        samples = load_relation_ground_truth(args.ground_truth)
        prepared_rows = build_supervised_samples(samples)
        train_jsonl = Path(args.prepared_jsonl)
        write_jsonl(train_jsonl, prepared_rows)
        source_path = str(Path(args.ground_truth))
        print(
            f"[PhoBERTRE] Prepared supervised JSONL: {train_jsonl} ({len(prepared_rows)} rows)"
        )

    summary_path = Path(args.output_dir) / "prepared_dataset_summary.json"
    write_json(summary_path, summarize_dataset(prepared_rows, source_path))
    print(f"[PhoBERTRE] Dataset summary: {summary_path}")

    if args.prepare_only:
        print("[PhoBERTRE] Prepare-only mode, skip training.")
        return

    extractor = PhoBERTRelationExtractor()
    train_result = extractor.train(
        train_jsonl=str(train_jsonl),
        output_dir=args.output_dir,
        val_split=args.val_split,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_confidence=args.min_confidence,
    )
    print("[PhoBERTRE] Training summary:")
    print(json.dumps(train_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
