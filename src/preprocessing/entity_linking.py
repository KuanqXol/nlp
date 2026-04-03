"""
Module: entity_linking.py
Entity Linking 3-stage cho tiếng Việt:
  1) Normalize text
  2) Exact match lookup
  3) Embedding similarity matching (FAISS / numpy fallback)

Output link chuẩn:
{
    "entity_id": "United_States",
    "surface_form": "Mỹ",
    "canonical": "Hoa Kỳ",
    "similarity": 0.91,
    "match_type": "embedding"
}
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
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

DEFAULT_MODEL = "keepitreal/vietnamese-sbert"
DEFAULT_THRESHOLD = 0.80  # lowered from 0.85 to catch more embedding matches before Levenshtein
LEVENSHTEIN_MIN_LEN = 4
LEVENSHTEIN_SHORT_MAX_DISTANCE = 1
LEVENSHTEIN_LONG_MAX_DISTANCE = 2


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _remove_diacritics(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _normalized_forms(text: str) -> Tuple[str, str]:
    norm = _normalize_text(text)
    return norm, _remove_diacritics(norm)


def _make_entity_id(canonical: str) -> str:
    base = _remove_diacritics(_normalize_text(canonical)).replace(" ", "_")
    return re.sub(r"[^a-zA-Z0-9_]", "", base) or "entity"


LEVENSHTEIN_BLACKLIST_PAIRS = {
    tuple(sorted((_remove_diacritics(_normalize_text(left)), _remove_diacritics(_normalize_text(right)))))
    for left, right in [
        ("Hà Nội", "Hà Nam"),
        ("Viettel", "Vietjet"),
        ("NATO", "nano"),
        ("VinAI", "Vincom"),
        ("Google", "Googol"),
        ("Samsung", "Samson"),
    ]
}


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, ch_left in enumerate(left, 1):
        current = [i]
        for j, ch_right in enumerate(right, 1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ch_left != ch_right)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current

    return previous[-1]


class _EmbeddingBackend:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = None
        self.dim = 0

    def _load(self):
        if self.model is not None:
            return
        if not _SBERT_AVAILABLE:
            raise RuntimeError("sentence-transformers chưa cài")
        self.model = SentenceTransformer(self.model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: List[str]) -> np.ndarray:
        self._load()
        vecs = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


class EntityLinker:
    """
    Entity linker production-style với index embedding entity.

    - Exact match: normalized key
    - Embedding match: cosine similarity qua FAISS / numpy
    - Cache:
      + embedding cache theo text
      + link result cache theo (surface, type)
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        model_name: str = DEFAULT_MODEL,
    ):
        self.similarity_threshold = similarity_threshold
        self._embed = _EmbeddingBackend(model_name=model_name)

        # entity_id -> metadata
        self._entities: Dict[str, Dict] = {}

        # normalized form -> entity_id
        self._exact_index: Dict[str, str] = {}
        self._exact_no_diacritics: Dict[str, str] = {}

        # vector index states
        self._faiss_index = None
        self._faiss_ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None

        # caches
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._link_cache: Dict[str, Dict] = {}

        # Seed aliases quan trọng
        self._seed_aliases(
            {
                "WHO": "WHO",
                "LHQ": "Liên Hợp Quốc",
                "UN": "Liên Hợp Quốc",
                "Hanoi": "Hà Nội",
                "Hoa Kỳ": "Mỹ",
                "USA": "Mỹ",
                "Vladimir Putin": "Putin",
                "SARS-CoV-2": "COVID-19",
                "covid": "COVID-19",
            }
        )

    def _seed_aliases(self, alias_map: Dict[str, str]):
        for alias, canonical in alias_map.items():
            entity_id = self._register_entity(
                canonical, entity_type="MISC", alias=alias
            )
            self._register_exact(alias, entity_id)
            self._register_exact(canonical, entity_id)

    def _register_exact(self, text: str, entity_id: str):
        norm, norm_no = _normalized_forms(text)
        if norm:
            self._exact_index[norm] = entity_id
        if norm_no:
            self._exact_no_diacritics[norm_no] = entity_id

    def _register_entity(
        self, canonical: str, entity_type: str = "MISC", alias: Optional[str] = None
    ) -> str:
        norm, _ = _normalized_forms(canonical)
        if norm in self._exact_index:
            return self._exact_index[norm]

        entity_id = _make_entity_id(canonical)
        if entity_id in self._entities:
            suffix = 2
            while f"{entity_id}_{suffix}" in self._entities:
                suffix += 1
            entity_id = f"{entity_id}_{suffix}"

        self._entities[entity_id] = {
            "entity_id": entity_id,
            "canonical": canonical,
            "type": entity_type,
            "aliases": set([alias or canonical]),
            "embedding": None,
            "frequency": 0,
        }

        self._register_exact(canonical, entity_id)
        if alias:
            self._register_exact(alias, entity_id)
        return entity_id

    def _cache_key(self, surface: str, entity_type: str) -> str:
        base = f"{surface}|{entity_type}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _encode_cached(self, text: str) -> np.ndarray:
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        vec = self._embed.encode([text])[0]
        self._embedding_cache[text] = vec
        return vec

    def _rebuild_vector_index(self):
        vectors = []
        ids = []
        for entity_id, info in self._entities.items():
            emb = info.get("embedding")
            if emb is None:
                continue
            vectors.append(emb)
            ids.append(entity_id)

        if not vectors:
            self._faiss_index = None
            self._faiss_ids = []
            self._matrix = None
            return

        mat = np.asarray(vectors, dtype=np.float32)
        self._faiss_ids = ids
        self._matrix = mat

        if _FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(mat.shape[1])
            index.add(mat)
            self._faiss_index = index
        else:
            self._faiss_index = None

    def _ensure_entity_embedding(self, entity_id: str):
        info = self._entities.get(entity_id)
        if info is None:
            return
        if info.get("embedding") is not None:
            return
        try:
            info["embedding"] = self._encode_cached(info["canonical"])
        except RuntimeError:
            return

    def _exact_lookup(self, surface_form: str) -> Optional[str]:
        norm, norm_no = _normalized_forms(surface_form)
        if norm in self._exact_index:
            return self._exact_index[norm]
        if norm_no in self._exact_no_diacritics:
            return self._exact_no_diacritics[norm_no]
        return None

    def _levenshtein_lookup(self, surface_form: str) -> Tuple[Optional[str], float]:
        _, norm_no = _normalized_forms(surface_form)
        if len(norm_no) < LEVENSHTEIN_MIN_LEN:
            return None, 0.0

        best_id = None
        best_dist = None
        best_score = 0.0

        for candidate_norm_no, entity_id in self._exact_no_diacritics.items():
            if len(candidate_norm_no) < LEVENSHTEIN_MIN_LEN:
                continue
            if norm_no[0] != candidate_norm_no[0]:
                continue
            if (
                tuple(sorted((norm_no, candidate_norm_no)))
                in LEVENSHTEIN_BLACKLIST_PAIRS
            ):
                continue

            max_len = max(len(norm_no), len(candidate_norm_no))
            max_dist = (
                LEVENSHTEIN_SHORT_MAX_DISTANCE
                if max_len < 8
                else LEVENSHTEIN_LONG_MAX_DISTANCE
            )
            if abs(len(norm_no) - len(candidate_norm_no)) > max_dist:
                continue

            dist = _levenshtein_distance(norm_no, candidate_norm_no)
            if dist > max_dist:
                continue

            sim = 1.0 - (dist / max_len)
            if best_dist is None or dist < best_dist or (
                dist == best_dist and sim > best_score
            ):
                best_id = entity_id
                best_dist = dist
                best_score = sim

        return best_id, round(best_score, 4) if best_id else 0.0

    def _embedding_lookup(self, surface_form: str) -> Tuple[Optional[str], float]:
        if not _SBERT_AVAILABLE:
            return None, 0.0
        if not self._entities:
            return None, 0.0

        # Build embeddings lazily
        changed = False
        for entity_id in list(self._entities.keys()):
            info = self._entities[entity_id]
            if info.get("embedding") is None:
                self._ensure_entity_embedding(entity_id)
                changed = True
        if changed or self._matrix is None:
            self._rebuild_vector_index()

        if self._matrix is None or len(self._faiss_ids) == 0:
            return None, 0.0

        query_vec = self._encode_cached(surface_form).reshape(1, -1)

        if self._faiss_index is not None:
            sims, idxs = self._faiss_index.search(query_vec.astype(np.float32), 1)
            idx = int(idxs[0][0])
            if idx < 0:
                return None, 0.0
            return self._faiss_ids[idx], float(sims[0][0])

        sims_np = (self._matrix @ query_vec.T).reshape(-1)
        idx = int(np.argmax(sims_np))
        return self._faiss_ids[idx], float(sims_np[idx])

    def link(
        self,
        surface_form: str,
        context: str = "",
        entity_type: Optional[str] = None,
        top_k: int = 1,
    ) -> Tuple[str, float]:
        result = self.link_mention(surface_form, entity_type=entity_type)
        return result["canonical"], float(result["similarity"])

    def link_mention(
        self, surface_form: str, entity_type: Optional[str] = None
    ) -> Dict:
        entity_type = entity_type or "MISC"
        cache_key = self._cache_key(surface_form, entity_type)
        if cache_key in self._link_cache:
            return dict(self._link_cache[cache_key])

        # Stage 1-2: normalization + exact match
        matched_id = self._exact_lookup(surface_form)
        if matched_id:
            info = self._entities[matched_id]
            info["frequency"] += 1
            info["aliases"].add(surface_form)
            out = {
                "entity_id": matched_id,
                "surface_form": surface_form,
                "canonical": info["canonical"],
                "type": info.get("type", entity_type),
                "similarity": 1.0,
                "match_type": "exact",
            }
            self._link_cache[cache_key] = dict(out)
            return out

        # Stage 3: embedding similarity (bi-encoder, BEFORE Levenshtein)
        candidate_id, sim = self._embedding_lookup(surface_form)
        if candidate_id and sim >= self.similarity_threshold:
            info = self._entities[candidate_id]
            info["frequency"] += 1
            info["aliases"].add(surface_form)
            self._register_exact(surface_form, candidate_id)
            out = {
                "entity_id": candidate_id,
                "surface_form": surface_form,
                "canonical": info["canonical"],
                "type": info.get("type", entity_type),
                "similarity": round(sim, 4),
                "match_type": "embedding",
            }
            self._link_cache[cache_key] = dict(out)
            return out

        # Stage 4: fuzzy Levenshtein match (fallback when embedding fails)
        fuzzy_id, fuzzy_sim = self._levenshtein_lookup(surface_form)
        if fuzzy_id:
            info = self._entities[fuzzy_id]
            info["frequency"] += 1
            info["aliases"].add(surface_form)
            self._register_exact(surface_form, fuzzy_id)
            out = {
                "entity_id": fuzzy_id,
                "surface_form": surface_form,
                "canonical": info["canonical"],
                "type": info.get("type", entity_type),
                "similarity": fuzzy_sim,
                "match_type": "levenshtein",
            }
            self._link_cache[cache_key] = dict(out)
            return out

        # New node
        new_id = self._register_entity(
            surface_form.strip(), entity_type=entity_type, alias=surface_form
        )
        self._ensure_entity_embedding(new_id)
        self._rebuild_vector_index()
        info = self._entities[new_id]
        info["frequency"] += 1
        out = {
            "entity_id": new_id,
            "surface_form": surface_form,
            "canonical": info["canonical"],
            "type": entity_type,
            "similarity": 1.0,
            "match_type": "new",
        }
        self._link_cache[cache_key] = dict(out)
        return out

    def link_entities(
        self, entities: List[Dict], context_sentences: Optional[List[str]] = None
    ) -> List[Dict]:
        merged: Dict[Tuple[str, str], Dict] = {}
        for ent in entities:
            surface = (
                ent.get("resolved_text")
                or ent.get("text")
                or ent.get("entity_text", "")
            )
            display_text = ent.get("mention_text") or ent.get("text") or surface
            ent_type = ent.get("type") or ent.get("entity_type", "MISC")
            if not surface:
                continue

            link = self.link_mention(surface, entity_type=ent_type)
            key = (link["entity_id"], ent_type)
            if key not in merged:
                merged[key] = {
                    "text": display_text,
                    "canonical": link["canonical"],
                    "entity_id": link["entity_id"],
                    "type": ent_type,
                    "link_score": float(link["similarity"]),
                    "match_type": link["match_type"],
                    "aliases": [display_text],
                }
            else:
                item = merged[key]
                if display_text not in item["aliases"]:
                    item["aliases"].append(display_text)
                item["link_score"] = max(
                    float(item.get("link_score", 0.0)), float(link["similarity"])
                )

        return list(merged.values())

    def process_document(self, doc: Dict) -> Dict:
        out = dict(doc)
        out["linked_entities"] = self.link_entities(doc.get("entities", []))
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

    def add_alias(self, alias: str, canonical: str):
        entity_id = self._register_entity(canonical, alias=alias)
        self._register_exact(alias, entity_id)


class SimpleEntityLinker(EntityLinker):
    """Backward-compat class name."""


if __name__ == "__main__":
    linker = EntityLinker(similarity_threshold=0.85)
    tests = ["Mỹ", "Hoa Kỳ", "Vladimir Putin", "Putin", "Hanoi", "Hà Nội"]
    for t in tests:
        print(linker.link_mention(t, entity_type="MISC"))
