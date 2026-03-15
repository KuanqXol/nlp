"""
relation_extraction_phobert.py
───────────────────────────────
PhoBERT-based Relation Extraction dùng Distant Supervision.

Pipeline offline (chạy một lần trên toàn corpus):
  1. Wikidata lookup: lấy quan hệ đã biết giữa các canonical entity
  2. Sentence alignment: tìm câu trong corpus chứa cặp entity đó
  3. Tạo training data: positive (có quan hệ Wikidata) + negative samples
  4. Fine-tune PhoBERT classification head
  5. Inference trên toàn corpus → triples có confidence score

Khi query: KHÔNG dùng module này (chỉ dùng lúc build index).

Cài đặt:
  pip install transformers torch requests

Dùng:
  from relation_extraction_phobert import PhoBERTRelationExtractor, WikidataRelationFetcher

  # Bước 1-3: tạo training data
  fetcher  = WikidataRelationFetcher()
  trainer  = PhoBERTRelationExtractor()
  trainer.build_training_data(corpus_docs, fetcher, out_path="re_train.jsonl")

  # Bước 4: train (cần GPU, vài giờ)
  trainer.train("re_train.jsonl", model_out="phobert_re/")

  # Bước 5: inference
  trainer.load("phobert_re/")
  docs_with_triples = trainer.batch_extract(docs)
"""

import json
import re
import time
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import requests

    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    import numpy as np

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ════════════════════════════════════════════════════════════════════════════
# 1. RELATION LABEL MAP
# ════════════════════════════════════════════════════════════════════════════

# Wikidata property ID → relation label trong hệ thống
WIKIDATA_PROP_MAP: Dict[str, str] = {
    "P35": "leads",  # head of state
    "P6": "leads",  # head of government
    "P488": "leads",  # chairperson
    "P1037": "leads",  # director/manager
    "P17": "located_in",  # country
    "P131": "located_in",  # located in administrative entity
    "P159": "located_in",  # headquarters location
    "P571": "founded",  # inception (org founded)
    "P112": "founded",  # founded by
    "P127": "member_of",  # owned by
    "P749": "member_of",  # parent organization
    "P463": "member_of",  # member of
    "P102": "member_of",  # member of political party
    "P1344": "cooperates_with",  # participant in
    "P18": None,  # image (skip)
    "P569": None,  # date of birth (skip)
    "P570": None,  # date of death (skip)
}

RELATION_LABELS = sorted(set(v for v in WIKIDATA_PROP_MAP.values() if v))
RELATION_LABELS.append("no_relation")  # negative class

LABEL2ID = {label: i for i, label in enumerate(RELATION_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

NO_RELATION_ID = LABEL2ID["no_relation"]

# ── Inference constraints ───────────────────────────────────────────────────
# (source_type, target_type) -> allowed relation labels
VALID_RELATIONS: Dict[Tuple[str, str], List[str]] = {
    ("PER", "ORG"): ["leads", "member_of", "cooperates_with"],
    ("PER", "LOC"): ["located_in", "leads"],
    ("ORG", "LOC"): ["located_in", "invests_in", "supports"],
    ("ORG", "ORG"): ["cooperates_with", "member_of", "acquires"],
    ("LOC", "LOC"): ["supports", "sanctions", "cooperates_with"],
}

MAX_ENTITY_TOKEN_DISTANCE = 30


# ════════════════════════════════════════════════════════════════════════════
# 2. WIKIDATA FETCHER
# ════════════════════════════════════════════════════════════════════════════


class WikidataRelationFetcher:
    """
    Lấy quan hệ giữa entity pairs từ Wikidata SPARQL endpoint.

    Cache kết quả vào file JSON để không gọi lại API.

    Dùng:
        fetcher = WikidataRelationFetcher(cache_path="wikidata_cache.json")
        relations = fetcher.get_relations("Putin", "Nga")
        # → ["leads", "located_in"]
    """

    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    SEARCH_ENDPOINT = "https://www.wikidata.org/w/api.php"

    def __init__(self, cache_path: str = "wikidata_cache.json"):
        self.cache_path = cache_path
        self._cache: Dict[str, Dict] = self._load_cache()
        self._qid_cache: Dict[str, Optional[str]] = {}

    def _load_cache(self) -> Dict:
        if Path(self.cache_path).exists():
            with open(self.cache_path) as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def _get_qid(self, entity_name: str) -> Optional[str]:
        """Tìm Wikidata QID từ tên entity."""
        if entity_name in self._qid_cache:
            return self._qid_cache[entity_name]

        if not _REQUESTS_OK:
            return None

        try:
            r = requests.get(
                self.SEARCH_ENDPOINT,
                params={
                    "action": "wbsearchentities",
                    "search": entity_name,
                    "language": "vi",
                    "format": "json",
                    "limit": 3,
                },
                timeout=5,
                headers={"User-Agent": "NewsKGBot/1.0"},
            )
            data = r.json()
            results = data.get("search", [])
            qid = results[0]["id"] if results else None
        except Exception:
            qid = None

        self._qid_cache[entity_name] = qid
        time.sleep(0.3)  # Rate limit
        return qid

    def get_relations(self, entity1: str, entity2: str) -> List[str]:
        """
        Trả về danh sách relation label giữa entity1 và entity2.
        Gọi Wikidata nếu chưa có trong cache.
        """
        cache_key = f"{entity1}|||{entity2}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not _REQUESTS_OK:
            return []

        qid1 = self._get_qid(entity1)
        qid2 = self._get_qid(entity2)

        if not qid1 or not qid2:
            self._cache[cache_key] = []
            return []

        # SPARQL: tìm property từ qid1 đến qid2
        sparql = f"""
        SELECT ?prop WHERE {{
          wd:{qid1} ?prop wd:{qid2} .
          FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/"))
        }} LIMIT 10
        """
        try:
            r = requests.get(
                self.SPARQL_ENDPOINT,
                params={"query": sparql, "format": "json"},
                timeout=10,
                headers={"User-Agent": "NewsKGBot/1.0"},
            )
            data = r.json()
            prop_ids = [
                b["prop"]["value"].split("/")[-1] for b in data["results"]["bindings"]
            ]
            # Map property ID → relation label
            labels = list(
                set(
                    WIKIDATA_PROP_MAP[pid]
                    for pid in prop_ids
                    if pid in WIKIDATA_PROP_MAP and WIKIDATA_PROP_MAP[pid]
                )
            )
        except Exception:
            labels = []

        self._cache[cache_key] = labels
        self._save_cache()
        time.sleep(0.5)
        return labels

    def batch_lookup(
        self,
        entity_pairs: List[Tuple[str, str]],
        log_every: int = 50,
    ) -> Dict[Tuple[str, str], List[str]]:
        """Batch lookup với rate limiting và logging."""
        results = {}
        for i, (e1, e2) in enumerate(entity_pairs):
            results[(e1, e2)] = self.get_relations(e1, e2)
            if (i + 1) % log_every == 0:
                print(f"[WikidataFetcher] {i+1}/{len(entity_pairs)} pairs done")
        self._save_cache()
        return results


# ════════════════════════════════════════════════════════════════════════════
# 3. TRAINING DATA BUILDER
# ════════════════════════════════════════════════════════════════════════════


class DistantSupervisionBuilder:
    """
    Tạo training data cho PhoBERT RE bằng distant supervision.

    Positive: câu chứa cặp entity có quan hệ trong Wikidata
    Negative: câu chứa cặp entity KHÔNG có quan hệ trong Wikidata
             + câu có cặp entity có quan hệ nhưng KHÔNG thể hiện quan hệ đó

    Noise reduction:
    - Chỉ lấy câu ngắn (< MAX_SENT_LEN token)
    - Require cả 2 entity xuất hiện trong cùng câu (không cross-sentence)
    - Negative ratio: max 2 negative / 1 positive (class balance)
    """

    MAX_SENT_LEN = 128  # Token
    NEG_RATIO = 2  # Số negative per positive
    MIN_SENT_LEN = 10  # Ký tự tối thiểu của câu

    def build(
        self,
        documents: List[Dict],
        fetcher: WikidataRelationFetcher,
        out_path: str = "re_train.jsonl",
        max_pairs_per_doc: int = 20,
    ) -> int:
        """
        Build training data từ corpus.

        Args:
            documents: List doc đã qua NER + entity linking
            fetcher:   WikidataRelationFetcher
            out_path:  Output JSONL file
            max_pairs_per_doc: Giới hạn số cặp entity xử lý mỗi bài

        Returns:
            Số sample được tạo
        """
        print(f"[DistantSupervision] Build training data từ {len(documents)} bài...")

        all_samples = []
        pos_count = 0
        neg_count = 0

        for doc_idx, doc in enumerate(documents):
            text = doc.get("full_text", "")
            entities = doc.get("linked_entities", [])
            category = doc.get("category", "")

            if not text or len(entities) < 2:
                continue

            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", text)
                if len(s.strip()) >= self.MIN_SENT_LEN
            ]

            # Lấy unique entity pairs
            pairs = []
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1 = entities[i].get("canonical", "")
                    e2 = entities[j].get("canonical", "")
                    if e1 and e2 and e1 != e2:
                        pairs.append((e1, entities[i], e2, entities[j]))

            # Giới hạn số cặp
            if len(pairs) > max_pairs_per_doc:
                pairs = random.sample(pairs, max_pairs_per_doc)

            for e1_name, e1_ent, e2_name, e2_ent in pairs:
                # Wikidata lookup
                wd_relations = fetcher.get_relations(e1_name, e2_name)

                # Tìm câu chứa cả 2 entity
                for sent in sentences:
                    sent_lower = sent.lower()
                    has_e1 = e1_name.lower() in sent_lower
                    has_e2 = e2_name.lower() in sent_lower
                    if not (has_e1 and has_e2):
                        continue
                    if len(sent.split()) > self.MAX_SENT_LEN:
                        continue

                    if wd_relations:
                        # Positive sample — một sample per relation
                        for rel in wd_relations:
                            if rel in LABEL2ID:
                                all_samples.append(
                                    {
                                        "sentence": sent,
                                        "entity1": e1_name,
                                        "entity2": e2_name,
                                        "e1_type": e1_ent.get("type", ""),
                                        "e2_type": e2_ent.get("type", ""),
                                        "relation": rel,
                                        "label_id": LABEL2ID[rel],
                                        "category": category,
                                        "source": "wikidata_positive",
                                    }
                                )
                            pos_count += 1
                    else:
                        # Negative sample (rate-limited)
                        if neg_count < pos_count * self.NEG_RATIO:
                            all_samples.append(
                                {
                                    "sentence": sent,
                                    "entity1": e1_name,
                                    "entity2": e2_name,
                                    "e1_type": e1_ent.get("type", ""),
                                    "e2_type": e2_ent.get("type", ""),
                                    "relation": "no_relation",
                                    "label_id": NO_RELATION_ID,
                                    "category": category,
                                    "source": "wikidata_negative",
                                }
                            )
                            neg_count += 1
                    break  # 1 câu per pair

            if (doc_idx + 1) % 100 == 0:
                print(
                    f"  [{doc_idx+1}/{len(documents)}] pos={pos_count} neg={neg_count}"
                )

        # Shuffle và ghi ra file
        random.shuffle(all_samples)
        with open(out_path, "w", encoding="utf-8") as f:
            for s in all_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(
            f"[DistantSupervision] Done: {len(all_samples)} samples "
            f"(pos={pos_count}, neg={neg_count}) → {out_path}"
        )
        return len(all_samples)


# ════════════════════════════════════════════════════════════════════════════
# 4. PHOBERT RE DATASET
# ════════════════════════════════════════════════════════════════════════════

if _TORCH_OK:

    class REDataset(Dataset):
        """
        PyTorch Dataset cho PhoBERT RE.
        Input format: [CLS] câu [SEP] entity1 [SEP] entity2 [SEP]
        """

        def __init__(self, samples: List[Dict], tokenizer, max_length: int = 256):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            # Đánh dấu entity trong câu bằng special token
            sent = self._mark_entities(s["sentence"], s["entity1"], s["entity2"])
            text = f"{sent} [SEP] {s['entity1']} [SEP] {s['entity2']}"

            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(s["label_id"], dtype=torch.long),
            }

        @staticmethod
        def _mark_entities(sentence: str, e1: str, e2: str) -> str:
            """Thêm marker [E1] entity1 [/E1] vào câu."""
            sent = sentence
            if e1 in sent:
                sent = sent.replace(e1, f"[E1] {e1} [/E1]", 1)
            if e2 in sent:
                sent = sent.replace(e2, f"[E2] {e2} [/E2]", 1)
            return sent


# ════════════════════════════════════════════════════════════════════════════
# 5. PHOBERT RE TRAINER + INFERENCER
# ════════════════════════════════════════════════════════════════════════════


class PhoBERTRelationExtractor:
    """
    PhoBERT-based Relation Extractor.

    Workflow:
        # Offline build (một lần):
        builder = DistantSupervisionBuilder()
        fetcher = WikidataRelationFetcher()
        builder.build(docs, fetcher, "re_train.jsonl")

        extractor = PhoBERTRelationExtractor()
        extractor.train("re_train.jsonl", "phobert_re/")

        # Inference:
        extractor.load("phobert_re/")
        docs = extractor.batch_extract(docs)
    """

    MODEL_NAME = "vinai/phobert-base"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.MODEL_NAME
        self._tokenizer = None
        self._model = None
        self._loaded = False

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        train_jsonl: str,
        output_dir: str,
        val_split: float = 0.1,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        min_confidence: float = 0.60,
    ):
        """
        Fine-tune PhoBERT trên training data từ distant supervision.

        Args:
            train_jsonl:  File JSONL từ DistantSupervisionBuilder.build()
            output_dir:   Thư mục lưu model sau train
            val_split:    Tỉ lệ validation set
            num_epochs:   Số epoch train
            batch_size:   Batch size (giảm nếu GPU OOM)
            learning_rate: Learning rate
            min_confidence: Threshold inference sau khi train
        """
        if not _TORCH_OK:
            raise RuntimeError(
                "torch + transformers chưa cài: pip install torch transformers"
            )

        print(f"[PhoBERTRE] Load training data: {train_jsonl}")
        samples = []
        with open(train_jsonl, encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))

        print(f"[PhoBERTRE] {len(samples)} samples, {len(RELATION_LABELS)} classes")

        # Train/val split
        random.shuffle(samples)
        n_val = max(1, int(len(samples) * val_split))
        val_samples = samples[:n_val]
        train_samples = samples[n_val:]

        print(f"[PhoBERTRE] Train: {len(train_samples)}, Val: {len(val_samples)}")

        # Load tokenizer + model
        print(f"[PhoBERTRE] Load PhoBERT: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Thêm special tokens cho entity marking
        special_tokens = {
            "additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]
        }
        tokenizer.add_special_tokens(special_tokens)

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(RELATION_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        model.resize_token_embeddings(len(tokenizer))

        # Dataset
        train_ds = REDataset(train_samples, tokenizer)
        val_ds = REDataset(val_samples, tokenizer)

        # Training args
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=50,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )

        print("[PhoBERTRE] Bắt đầu training...")
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Lưu thêm label map và config
        with open(f"{output_dir}/re_config.json", "w") as f:
            json.dump(
                {
                    "relation_labels": RELATION_LABELS,
                    "label2id": LABEL2ID,
                    "id2label": ID2LABEL,
                    "min_confidence": min_confidence,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"[PhoBERTRE] Training xong → {output_dir}")
        self._tokenizer = tokenizer
        self._model = model
        self._loaded = True

    # ── Load ──────────────────────────────────────────────────────────────

    def load(self, model_dir: str):
        """Load model đã train từ thư mục."""
        if not _TORCH_OK:
            raise RuntimeError("torch + transformers chưa cài")

        print(f"[PhoBERTRE] Load model: {model_dir}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._model.eval()

        if torch.cuda.is_available():
            self._model = self._model.cuda()

        config_path = Path(model_dir) / "re_config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            self._min_conf = cfg.get("min_confidence", 0.60)
        else:
            self._min_conf = 0.60

        self._loaded = True
        print(f"[PhoBERTRE] Loaded. Min confidence: {self._min_conf}")

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_pair(
        self,
        sentence: str,
        entity1: str,
        entity2: str,
    ) -> Tuple[str, float]:
        """
        Predict relation giữa entity1 và entity2 trong câu.
        Returns: (relation_label, confidence)
        """
        if not self._loaded:
            raise RuntimeError("Model chưa được load. Gọi load() hoặc train() trước.")

        marked = REDataset._mark_entities(sentence, entity1, entity2)
        text = f"{marked} [SEP] {entity1} [SEP] {entity2}"

        enc = self._tokenizer(
            text,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            enc = {k: v.cuda() for k, v in enc.items()}

        with torch.no_grad():
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]

        pred_id = int(probs.argmax())
        confidence = float(probs[pred_id])
        label = ID2LABEL.get(pred_id, "no_relation")

        return label, confidence

    @staticmethod
    def _is_valid_relation_for_types(relation: str, e1_type: str, e2_type: str) -> bool:
        allowed = VALID_RELATIONS.get((e1_type, e2_type), [])
        if relation in allowed:
            return True
        # Một số quan hệ có thể đối xứng theo thứ tự cặp type.
        reverse_allowed = VALID_RELATIONS.get((e2_type, e1_type), [])
        return relation in reverse_allowed

    @staticmethod
    def _token_distance(sentence: str, entity1: str, entity2: str) -> int:
        tokens = sentence.split()
        if not tokens:
            return 10**9

        def _find_index(entity: str) -> int:
            etoks = entity.split()
            if not etoks:
                return -1
            n = len(etoks)
            for i in range(max(0, len(tokens) - n + 1)):
                if [t.lower() for t in tokens[i : i + n]] == [t.lower() for t in etoks]:
                    return i
            return -1

        i1 = _find_index(entity1)
        i2 = _find_index(entity2)
        if i1 < 0 or i2 < 0:
            return 10**9
        return abs(i1 - i2)

    def extract_from_document(
        self,
        doc: Dict,
        min_confidence: float = None,
        cross_sentence: bool = False,
    ) -> List[Dict]:
        """
        Extract relation từ 1 document dùng PhoBERT model.

        Args:
            doc:            Document đã có linked_entities
            min_confidence: Override ngưỡng mặc định
            cross_sentence: Có xét cặp entity cross-sentence không

        Returns:
            List triple dicts: {subject, relation, object, confidence, temporal, method}
        """
        if not self._loaded:
            raise RuntimeError("Model chưa load")

        threshold = min_confidence if min_confidence is not None else self._min_conf
        text = doc.get("full_text", "")
        entities = doc.get("linked_entities", [])
        doc_date = doc.get("date", "")

        if not text or len(entities) < 2:
            return []

        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 8
        ]

        triples_seen: Dict[Tuple, Dict] = {}

        def _try_pair(sent, e1_name, e1_ent, e2_name, e2_ent, decay=1.0):
            # Constraint C: loại cặp entity quá xa nhau theo token distance.
            token_dist = self._token_distance(sent, e1_name, e2_name)
            if token_dist > MAX_ENTITY_TOKEN_DISTANCE:
                return

            relation, conf = self.predict_pair(sent, e1_name, e2_name)
            conf = round(conf * decay, 3)

            # Constraint A: threshold confidence.
            if relation == "no_relation" or conf < threshold:
                return

            # Constraint B: quan hệ phải hợp lệ với cặp type thực thể.
            if not self._is_valid_relation_for_types(
                relation,
                e1_ent.get("type", "MISC"),
                e2_ent.get("type", "MISC"),
            ):
                return

            key = (e1_name, relation, e2_name)
            prev = triples_seen.get(key)
            if prev is None or conf > prev.get("confidence", 0.0):
                triples_seen[key] = {
                    "confidence": conf,
                    "sentence": sent,
                    "token_distance": token_dist,
                }

        # Per-sentence
        for sent in sentences:
            sent_lower = sent.lower()
            present = [
                e for e in entities if e.get("canonical", "").lower() in sent_lower
            ]
            if len(present) < 2:
                continue
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    e1 = present[i]
                    e2 = present[j]
                    c1 = e1.get("canonical", "")
                    c2 = e2.get("canonical", "")
                    if c1 and c2 and c1 != c2:
                        _try_pair(sent, c1, e1, c2, e2)

        # Cross-sentence (optional, slower)
        if cross_sentence:
            for i in range(len(sentences) - 1):
                window = " ".join(sentences[i : i + 2])
                wl = window.lower()
                present = [e for e in entities if e.get("canonical", "").lower() in wl]
                if len(present) < 2:
                    continue
                for i2 in range(len(present)):
                    for j2 in range(i2 + 1, len(present)):
                        c1 = present[i2].get("canonical", "")
                        c2 = present[j2].get("canonical", "")
                        if c1 and c2 and c1 != c2:
                            _try_pair(
                                window, c1, present[i2], c2, present[j2], decay=0.88
                            )

        # Build output
        result = []
        for (subj, rel, obj), info in triples_seen.items():
            triple = {
                "subject": subj,
                "relation": rel,
                "object": obj,
                "confidence": info["confidence"],
                "method": "phobert",
                "sentence": info.get("sentence", ""),
                "token_distance": info.get("token_distance", -1),
            }
            if doc_date:
                triple["temporal"] = doc_date
            result.append(triple)

        result.sort(key=lambda x: -x["confidence"])
        return result

    def batch_extract(
        self,
        documents: List[Dict],
        min_confidence: float = None,
        log_every: int = 100,
    ) -> List[Dict]:
        """Inference trên toàn bộ corpus."""
        print(f"[PhoBERTRE] Inference {len(documents)} bài...")
        result, total = [], 0
        for i, doc in enumerate(documents):
            triples = self.extract_from_document(doc, min_confidence)
            out = doc.copy()
            out["triples"] = triples
            result.append(out)
            total += len(triples)
            if (i + 1) % log_every == 0 or (i + 1) == len(documents):
                print(f"  [{i+1}/{len(documents)}] triples so far: {total}")
        print(f"[PhoBERTRE] Xong. Tổng triple: {total}")
        return result


# ════════════════════════════════════════════════════════════════════════════
# 6. HYBRID EXTRACTOR: Rule-based + PhoBERT
# ════════════════════════════════════════════════════════════════════════════


class HybridRelationExtractor:
    """
    Kết hợp rule-based RE (nhanh, high-precision) và PhoBERT RE (cao hơn recall).

    Chiến lược:
    - Rule-based chạy trước: kết quả confidence cao → đưa thẳng vào KG
    - PhoBERT chạy sau: bổ sung triple mà rule-based miss
    - Nếu cùng triple: giữ score cao hơn
    - Nếu PhoBERT chưa sẵn: fallback hoàn toàn về rule-based

    Khi có model PhoBERT:
        hybrid = HybridRelationExtractor(phobert_dir="phobert_re/")

    Khi chưa có:
        hybrid = HybridRelationExtractor()   # chỉ rule-based
    """

    def __init__(self, phobert_dir: Optional[str] = None, min_confidence: float = 0.60):
        # Rule-based (luôn có)
        from src.preprocessing.relation_extraction import RelationExtractor

        self._rule = RelationExtractor()
        self._min_conf = min_confidence

        # PhoBERT (optional)
        self._phobert = None
        if phobert_dir and Path(phobert_dir).exists() and _TORCH_OK:
            try:
                self._phobert = PhoBERTRelationExtractor()
                self._phobert.load(phobert_dir)
                print("[HybridRE] PhoBERT loaded ✓")
            except Exception as e:
                print(f"[HybridRE] PhoBERT load thất bại: {e} — chỉ dùng rule-based")
        else:
            print("[HybridRE] Chế độ: rule-based only")

    def extract(self, doc: Dict) -> List[Dict]:
        # Rule-based
        rule_doc = self._rule.process_document(doc)
        rule_triples = {
            (t["subject"], t["relation"], t["object"]): t
            for t in rule_doc.get("triples", [])
        }

        merged = dict(rule_triples)

        # PhoBERT bổ sung
        if self._phobert:
            pb_triples = self._phobert.extract_from_document(doc, self._min_conf)
            for t in pb_triples:
                key = (t["subject"], t["relation"], t["object"])
                if key not in merged:
                    merged[key] = t
                else:
                    # Giữ confidence cao hơn
                    if t["confidence"] > merged[key]["confidence"]:
                        merged[key] = t

        result = sorted(merged.values(), key=lambda x: -x["confidence"])
        return result

    def process_document(self, doc: Dict) -> Dict:
        out = doc.copy()
        out["triples"] = self.extract(doc)
        return out

    def batch_process(self, documents: List[Dict], log_every: int = 500) -> List[Dict]:
        mode = "rule+phobert" if self._phobert else "rule-only"
        print(f"[HybridRE] Xử lý {len(documents)} bài (mode={mode})...")
        result, total = [], 0
        for i, doc in enumerate(documents):
            p = self.process_document(doc)
            result.append(p)
            total += len(p.get("triples", []))
            if (i + 1) % log_every == 0 or (i + 1) == len(documents):
                print(f"  [{i+1}/{len(documents)}] triples: {total}")
        print(f"[HybridRE] Xong. Tổng triple: {total}")
        return result


# ════════════════════════════════════════════════════════════════════════════
# DEMO / USAGE GUIDE
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("HƯỚNG DẪN SỬ DỤNG PhoBERT RE Pipeline")
    print("=" * 60)

    print(
        """
BƯỚC 1 — Cài đặt:
    pip install torch transformers requests

BƯỚC 2 — Build training data (cần internet để gọi Wikidata):
    from relation_extraction_phobert import *

    fetcher = WikidataRelationFetcher(cache_path="wikidata_cache.json")
    builder = DistantSupervisionBuilder()

    # docs là list document đã qua NER + entity linking
    n = builder.build(docs, fetcher, out_path="re_train.jsonl")
    print(f"Tạo {n} training samples")

BƯỚC 3 — Train (cần GPU, ~2-4 giờ cho 10k samples):
    extractor = PhoBERTRelationExtractor()
    extractor.train(
        train_jsonl="re_train.jsonl",
        output_dir="phobert_re/",
        num_epochs=3,
        batch_size=16,
    )

BƯỚC 4 — Inference trên toàn corpus:
    extractor.load("phobert_re/")
    docs_with_triples = extractor.batch_extract(all_docs)

BƯỚC 5 — Dùng Hybrid (rule-based + PhoBERT):
    hybrid = HybridRelationExtractor(phobert_dir="phobert_re/")
    docs   = hybrid.batch_process(all_docs)

Fallback khi chưa có model:
    hybrid = HybridRelationExtractor()   # chỉ rule-based, vẫn chạy được
"""
    )

    # Demo HybridRelationExtractor với rule-based only
    print("Demo HybridRelationExtractor (rule-based only):")
    hybrid = HybridRelationExtractor()

    sample = {
        "id": "demo",
        "date": "2024-03-01",
        "category": "kinh tế",
        "full_text": "Google đầu tư vào VinAI tại Hà Nội năm 2024. Phạm Minh Chính gặp Sundar Pichai.",
        "linked_entities": [
            {"text": "Google", "canonical": "Google", "type": "ORG", "link_score": 1.0},
            {"text": "VinAI", "canonical": "VinAI", "type": "ORG", "link_score": 1.0},
            {"text": "Hà Nội", "canonical": "Hà Nội", "type": "LOC", "link_score": 1.0},
            {
                "text": "Phạm Minh Chính",
                "canonical": "Phạm Minh Chính",
                "type": "PER",
                "link_score": 1.0,
            },
            {
                "text": "Sundar Pichai",
                "canonical": "Sundar Pichai",
                "type": "PER",
                "link_score": 1.0,
            },
        ],
    }

    processed = hybrid.process_document(sample)
    for t in processed["triples"]:
        print(
            f"  ({t['subject']}) -[{t['relation']}]-> ({t['object']})  conf={t['confidence']:.2f}"
        )
