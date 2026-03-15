"""
Module: entity_linking.py
Chức năng: Entity linking dựa trên embedding + context window.

Phương pháp:
  1. Small alias map (~50 entry) cho viết tắt / variant quan trọng
  2. Vietnamese SBERT encode (mention + context ±2 câu)
  3. FAISS clustering → canonical entity
  4. Context embedding disambiguation khi có nhiều candidate
  5. String normalization fallback cho entity < min_mentions

Scale tốt với 100k bài vì không cần bảng alias lớn.
"""

import re
import unicodedata
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False


# ── Small alias map — chỉ viết tắt + variant Latin/tiếng Việt quan trọng ─────
# Không cố cover hết — embedding xử lý phần còn lại

CORE_ALIAS_MAP: Dict[str, str] = {
    # Viết tắt tổ chức
    "who": "WHO",
    "lhq": "Liên Hợp Quốc",
    "un": "Liên Hợp Quốc",
    "eu": "EU",
    "wb": "Ngân hàng Thế giới",
    # Địa danh Latin ↔ tiếng Việt
    "hanoi": "Hà Nội",
    "hn": "Hà Nội",
    "tp.hcm": "TP.HCM",
    "saigon": "TP.HCM",
    "vietnam": "Việt Nam",
    "vn": "Việt Nam",
    "russia": "Nga",
    "usa": "Mỹ",
    "hoa kỳ": "Mỹ",
    "trung quốc": "Trung Quốc",
    "china": "Trung Quốc",
    "uk": "Anh",
    # Tên bệnh
    "sars-cov-2": "COVID-19",
    "coronavirus": "COVID-19",
    "covid": "COVID-19",
    "sốt xuất huyết": "Dengue",
    # Tên người thông dụng
    "vladimir putin": "Putin",
    "joe biden": "Biden",
    "donald trump": "Trump",
    "xi jinping": "Tập Cận Bình",
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _nk(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _no_accent(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _norm_alias(text: str) -> str:
    """Chuẩn hóa để tra alias: lowercase + no-accent + strip."""
    return _no_accent(_nk(text))


# ── Embedding-based entity store ──────────────────────────────────────────────


class EntityEmbeddingStore:
    """
    Lưu trữ embedding của canonical entity và mention context.
    Dùng FAISS để tìm nearest neighbor nhanh.

    Workflow build-time (offline, chạy một lần trên toàn corpus):
        store = EntityEmbeddingStore(sbert_model)
        store.add_mentions(all_mentions)   # mention + context từ corpus
        store.cluster(min_mentions=5)      # gom mention về canonical
        store.save("entity_store.pkl")

    Workflow query-time:
        store = EntityEmbeddingStore.load("entity_store.pkl")
        canonical, score = store.link(mention_text, context)
    """

    def __init__(self, sbert_model_name: str = "keepitreal/vietnamese-sbert"):
        self._model = None
        self._model_name = sbert_model_name
        self._dim: int = 0

        # canonical → {"embedding": np.array, "frequency": int, "aliases": set}
        self._canonical_map: Dict[str, Dict] = {}

        # FAISS index trên canonical embeddings
        self._index = None
        self._index_keys: List[str] = []  # canonical tương ứng mỗi row trong index

        # Raw mentions trước khi cluster
        # {"surface": str, "context": str, "type": str, "embedding": np.array}
        self._raw_mentions: List[Dict] = []

        self._fitted = False

    def _load_model(self):
        if self._model is None:
            if not _SBERT_AVAILABLE:
                raise RuntimeError(
                    "sentence-transformers chưa được cài: pip install sentence-transformers"
                )
            print(f"[EntityStore] Tải SBERT model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_sentence_embedding_dimension()

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode batch text → normalized float32 vectors."""
        self._load_model()
        vecs = self._model.encode(
            texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        return vecs.astype(np.float32)

    def add_mention(self, surface: str, context: str, entity_type: str):
        """Thêm một mention vào raw pool (chưa cluster)."""
        self._raw_mentions.append(
            {
                "surface": surface,
                "context": context,
                "type": entity_type,
            }
        )

    def add_mentions_from_documents(
        self, documents: List[Dict], context_window: int = 2
    ):
        """
        Extract mention + context từ toàn bộ corpus.
        context_window: số câu lấy trước/sau mention.
        """
        print(f"[EntityStore] Collect mentions từ {len(documents)} bài...")
        for doc in documents:
            text = doc.get("full_text", doc.get("content", ""))
            sentences = re.split(r"(?<=[.!?])\s+", text)
            entities = doc.get("entities", [])

            for ent in entities:
                surface = ent.get("text", "")
                ent_type = ent.get("type", "MISC")
                if not surface:
                    continue
                # Tìm câu chứa entity, lấy context ±window
                for idx, sent in enumerate(sentences):
                    if surface.lower() in sent.lower():
                        start = max(0, idx - context_window)
                        end = min(len(sentences), idx + context_window + 1)
                        ctx = " ".join(sentences[start:end])
                        self.add_mention(surface, ctx, ent_type)
                        break  # Chỉ lấy lần đầu tiên
        print(f"[EntityStore] Collected {len(self._raw_mentions)} mentions.")

    def build_index(
        self,
        min_mentions: int = 5,
        similarity_threshold: float = 0.82,
    ):
        """
        Cluster raw mentions → canonical entities, build FAISS index.

        min_mentions: Entity xuất hiện < lần này → dùng string normalization thay vì cluster.
        similarity_threshold: cosine similarity tối thiểu để merge 2 mention vào cùng cluster.
        """
        if not self._raw_mentions:
            print("[EntityStore] Không có mention nào để build index.")
            return

        print(f"[EntityStore] Encode {len(self._raw_mentions)} mentions...")
        texts_to_encode = [
            f"{m['surface']} : {m['context']}" for m in self._raw_mentions
        ]
        all_vecs = self._encode(texts_to_encode)

        # Đếm frequency theo surface (normalized)
        freq: Dict[str, int] = defaultdict(int)
        surface_vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
        surface_types: Dict[str, str] = {}

        for i, m in enumerate(self._raw_mentions):
            key = _nk(m["surface"])
            freq[key] += 1
            surface_vecs[key].append(all_vecs[i])
            surface_types[key] = m["type"]

        # Tính centroid embedding cho mỗi surface
        centroids: Dict[str, np.ndarray] = {}
        for key, vecs in surface_vecs.items():
            c = np.mean(vecs, axis=0)
            c = c / (np.linalg.norm(c) + 1e-9)
            centroids[key] = c.astype(np.float32)

        # Cluster: entity có frequency đủ lớn → candidate canonical
        # Greedy: sắp xếp theo frequency giảm dần, merge nếu cosine ≥ threshold
        sorted_keys = sorted(freq.keys(), key=lambda k: -freq[k])
        canonical_to_members: Dict[str, List[str]] = {}
        key_to_canonical: Dict[str, str] = {}

        for key in sorted_keys:
            if freq[key] < min_mentions:
                # Ít mention → không cluster, giữ nguyên
                key_to_canonical[key] = key
                continue

            vec = centroids[key]
            best_can, best_sim = None, 0.0

            for can, can_members in canonical_to_members.items():
                can_vec = centroids.get(can)
                if can_vec is None:
                    continue
                sim = float(np.dot(vec, can_vec))
                if sim > best_sim:
                    best_sim = sim
                    best_can = can

            if best_can and best_sim >= similarity_threshold:
                # Merge vào cluster đã có
                canonical_to_members[best_can].append(key)
                key_to_canonical[key] = best_can
            else:
                # Tạo cluster mới — canonical là surface gốc (most frequent form)
                canonical_to_members[key] = [key]
                key_to_canonical[key] = key

        # Build canonical_map
        for can, members in canonical_to_members.items():
            # Canonical display name: chọn surface có frequency cao nhất
            best_surface = max(members, key=lambda k: freq[k])
            # Tìm tên đẹp nhất (có dấu, đúng hoa thường)
            original_surfaces = {
                _nk(m["surface"]): m["surface"] for m in self._raw_mentions
            }
            display_name = original_surfaces.get(best_surface, best_surface)

            self._canonical_map[can] = {
                "display": display_name,
                "embedding": centroids[can],
                "frequency": freq[can],
                "aliases": {original_surfaces.get(m, m) for m in members},
                "type": surface_types.get(can, "MISC"),
            }

        # Build FAISS index trên canonical centroids
        self._build_faiss()
        self._key_to_canonical = key_to_canonical
        self._fitted = True
        print(
            f"[EntityStore] Index xây xong: {len(self._canonical_map)} canonical entities."
        )

    def _build_faiss(self):
        if not self._canonical_map:
            return
        self._index_keys = list(self._canonical_map.keys())
        vecs = np.stack([self._canonical_map[k]["embedding"] for k in self._index_keys])
        self._dim = vecs.shape[1]

        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(
                self._dim
            )  # Inner product = cosine (normalized)
            self._index.add(vecs)
        else:
            # Numpy fallback
            self._matrix = vecs

    def link(
        self,
        surface: str,
        context: str = "",
        entity_type: Optional[str] = None,
        top_k: int = 1,
    ) -> Tuple[str, float]:
        """
        Link một mention về canonical.
        Returns: (canonical_display_name, confidence_score)
        """
        # Tier 1: Core alias map
        alias_key = _norm_alias(surface)
        for ak, canonical in CORE_ALIAS_MAP.items():
            if _norm_alias(ak) == alias_key:
                return canonical, 1.0

        if not self._fitted:
            return surface.strip(), 0.5

        # Tier 2: Exact surface match trong corpus
        nk = _nk(surface)
        if nk in self._key_to_canonical:
            can_key = self._key_to_canonical[nk]
            info = self._canonical_map.get(can_key, {})
            return info.get("display", surface), 0.95

        # Tier 3: Embedding similarity với context
        query_text = f"{surface} : {context}" if context else surface
        query_vec = self._encode([query_text])[0:1]

        if _FAISS_AVAILABLE and self._index is not None:
            sims, ids = self._index.search(query_vec, top_k)
            best_id = int(ids[0][0])
            best_sim = float(sims[0][0])
        else:
            sims_np = self._matrix @ query_vec.T
            best_id = int(np.argmax(sims_np))
            best_sim = float(sims_np[best_id])

        if best_sim < 0.65:
            return surface.strip(), 0.5

        can_key = self._index_keys[best_id]
        info = self._canonical_map.get(can_key, {})

        # Type constraint: nếu type không khớp thì giảm confidence
        if entity_type and info.get("type") != entity_type:
            best_sim *= 0.85

        return info.get("display", surface), round(best_sim, 3)

    def link_entities(
        self, entities: List[Dict], context_sentences: List[str] = None
    ) -> List[Dict]:
        """
        Chuẩn hóa và link toàn bộ entity list của một document.
        Gộp mention trùng canonical về một.
        """
        context = " ".join(context_sentences[:4]) if context_sentences else ""
        seen: Dict[Tuple, Dict] = {}

        for ent in entities:
            surface = ent.get("text", "")
            ent_type = ent.get("type", "MISC")
            canonical, score = self.link(surface, context, ent_type)

            key = (_nk(canonical), ent_type)
            if key not in seen:
                seen[key] = {
                    "text": surface,
                    "canonical": canonical,
                    "type": ent_type,
                    "link_score": round(score, 3),
                    "aliases": [surface],
                }
            else:
                if surface not in seen[key]["aliases"]:
                    seen[key]["aliases"].append(surface)
                if score > seen[key]["link_score"]:
                    seen[key]["link_score"] = round(score, 3)

        return list(seen.values())

    def process_document(self, doc: Dict) -> Dict:
        entities = doc.get("entities", [])
        sentences = re.split(r"(?<=[.!?])\s+", doc.get("full_text", ""))
        linked = self.link_entities(entities, context_sentences=sentences[:5])
        out = doc.copy()
        out["linked_entities"] = linked
        return out

    def batch_process(self, documents: List[Dict], log_every: int = 500) -> List[Dict]:
        print(f"[EntityLinker] Link {len(documents)} bài...")
        result = []
        for i, doc in enumerate(documents):
            result.append(self.process_document(doc))
            if (i + 1) % log_every == 0 or (i + 1) == len(documents):
                print(f"  [{i+1}/{len(documents)}]")
        print("[EntityLinker] Hoàn thành.")
        return result

    def save(self, path: str):
        data = {
            "canonical_map": self._canonical_map,
            "index_keys": self._index_keys,
            "key_to_canonical": getattr(self, "_key_to_canonical", {}),
            "dim": self._dim,
            "fitted": self._fitted,
        }
        if not _FAISS_AVAILABLE and hasattr(self, "_matrix"):
            data["matrix"] = self._matrix
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[EntityStore] Saved → {path}")

    @classmethod
    def load(cls, path: str, sbert_model_name: str = "keepitreal/vietnamese-sbert"):
        store = cls(sbert_model_name)
        with open(path, "rb") as f:
            data = pickle.load(f)
        store._canonical_map = data["canonical_map"]
        store._index_keys = data["index_keys"]
        store._key_to_canonical = data.get("key_to_canonical", {})
        store._dim = data["dim"]
        store._fitted = data["fitted"]
        if "matrix" in data:
            store._matrix = data["matrix"]
        else:
            store._build_faiss()
        print(f"[EntityStore] Loaded ← {path} ({len(store._canonical_map)} entities)")
        return store

    def stats(self) -> Dict:
        freq_list = [v["frequency"] for v in self._canonical_map.values()]
        return {
            "canonical_count": len(self._canonical_map),
            "raw_mentions": len(self._raw_mentions),
            "fitted": self._fitted,
            "top10_entities": sorted(
                [(v["display"], v["frequency"]) for v in self._canonical_map.values()],
                key=lambda x: -x[1],
            )[:10],
        }


# ── Lightweight linker: dùng khi chưa build index (alias + string norm) ────────


class SimpleEntityLinker:
    """
    Linker nhẹ dùng alias map + string normalization.
    Dùng trong pipeline test hoặc khi chưa có corpus đủ lớn để build embedding index.
    Interface giống EntityEmbeddingStore để dễ swap.
    """

    def __init__(self, alias_map: Dict[str, str] = None):
        self._alias = {
            _norm_alias(k): v for k, v in (alias_map or CORE_ALIAS_MAP).items()
        }

    def link(
        self,
        surface: str,
        context: str = "",
        entity_type: Optional[str] = None,
        top_k: int = 1,
    ) -> Tuple[str, float]:
        key = _norm_alias(surface)
        if key in self._alias:
            return self._alias[key], 1.0
        return surface.strip(), 0.5

    def add_alias(self, alias: str, canonical: str):
        self._alias[_norm_alias(alias)] = canonical

    def link_entities(
        self, entities: List[Dict], context_sentences: List[str] = None
    ) -> List[Dict]:
        seen: Dict[Tuple, Dict] = {}
        for ent in entities:
            surface = ent.get("text", "")
            ent_type = ent.get("type", "MISC")
            canonical, score = self.link(surface, entity_type=ent_type)
            key = (_nk(canonical), ent_type)
            if key not in seen:
                seen[key] = {
                    "text": surface,
                    "canonical": canonical,
                    "type": ent_type,
                    "link_score": score,
                    "aliases": [surface],
                }
            else:
                if surface not in seen[key]["aliases"]:
                    seen[key]["aliases"].append(surface)
        return list(seen.values())

    def process_document(self, doc: Dict) -> Dict:
        linked = self.link_entities(doc.get("entities", []))
        out = doc.copy()
        out["linked_entities"] = linked
        return out

    def batch_process(self, documents: List[Dict], log_every: int = 500) -> List[Dict]:
        print(f"[SimpleLinker] Link {len(documents)} bài...")
        result = [self.process_document(d) for d in documents]
        print("[SimpleLinker] Hoàn thành.")
        return result

    def add_alias(self, alias: str, canonical: str):
        self._alias[_norm_alias(alias)] = canonical


# ── Backwards-compatible class name ──────────────────────────────────────────

EntityLinker = SimpleEntityLinker  # alias cho code cũ dùng EntityLinker


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    linker = SimpleEntityLinker()

    tests = [
        ("Hanoi", "LOC"),
        ("SARS-CoV-2", "MISC"),
        ("Hoa Kỳ", "LOC"),
        ("Vladimir Putin", "PER"),
        ("WHO", "ORG"),
        ("XYZ Corp mới", "ORG"),
    ]
    print("=== SIMPLE LINKER (alias only) ===")
    for surface, etype in tests:
        canonical, score = linker.link(surface, entity_type=etype)
        print(f"  {surface:22s} [{etype}] -> {canonical:22s} score={score:.2f}")

    print("\n=== EMBEDDING STORE (cần SBERT) ===")
    print("  Dùng EntityEmbeddingStore.build_index() sau khi NER toàn corpus.")
    print("  store = EntityEmbeddingStore()")
    print("  store.add_mentions_from_documents(all_docs)")
    print("  store.build_index(min_mentions=5, similarity_threshold=0.82)")
    print("  store.save('entity_store.pkl')")
