"""
FastAPI web UI for Vietnamese KG-Enhanced News Search.

Features:
- Query search through the existing `NewsSearchSystem`
- Load a saved index from `data/index/`
- Optional lite demo mode when no index is available
- Show retrieval metadata and graph/rerank signals when present
- Compact query understanding and pipeline compare modes

Run:
  pip install fastapi uvicorn jinja2 python-multipart
  python -m uvicorn web_app:app --reload --port 8000
"""

from __future__ import annotations

import os
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.data_loader import NewsDataLoader, create_document
from src.graph.ranking import GraphRanker
from src.preprocessing.entity_linking import EntityLinker
from src.preprocessing.ner import VietnameseNER
from src.retrieval import EmbeddingManager, Retriever, chunk_documents, QueryProcessor
from src.reader import QAReader

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env", override=True)

DATA_DIR = ROOT_DIR / "data"
DEFAULT_DATA_PATH = DATA_DIR / "vnexpress_articles.csv"
DEFAULT_INDEX_DIR = DATA_DIR / "index"
TEMPLATES_DIR = ROOT_DIR / "web" / "templates"
STATIC_DIR = ROOT_DIR / "web" / "static"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_path(name: str) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip().strip('"')
    return value or None


@dataclass
class ServiceState:
    mode: str  # "index" | "retrieval-only" | "not-ready"
    message: str
    index_dir: str
    data_path: str
    built_at: Optional[float] = None
    n_docs: int = 0
    n_chunks: int = 0


class WebSearchService:
    def __init__(self, index_dir: Path, data_path: Path):
        self.index_dir = index_dir
        self.data_path = data_path
        self.em: Optional[EmbeddingManager] = None
        self.retriever: Optional[Retriever] = None
        self.documents: List[Dict[str, Any]] = []
        self.ranker: Optional[GraphRanker] = None
        self.importance_scores: Dict[str, float] = {}
        self.query_proc: Optional[QueryProcessor] = None
        self.reader = QAReader()
        self._ner: Optional[VietnameseNER] = None
        self._linker: Optional[EntityLinker] = None
        self._kg = None
        self._search_cache: Dict[str, Dict[str, Any]] = {}
        self._search_cache_ttl_seconds = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "120"))
        self.state = ServiceState(
            mode="not-ready",
            message="Service is starting...",
            index_dir=str(index_dir),
            data_path=str(data_path),
        )

    def _build_query_linker(self) -> Optional[EntityLinker]:
        if self.em is None:
            return None
        linker = EntityLinker(shared_encoder=self.em._enc)
        kg_added = linker.hydrate_from_knowledge_graph(self._kg)
        alias_added = linker.hydrate_safe_aliases_from_documents(self.documents)
        print(
            f"[WebUI] Query linker hydrated: canonical_entities={kg_added}, safe_aliases={alias_added}"
        )
        return linker

    def _ensure_query_proc(self):
        if self.query_proc is not None or self.em is None:
            return
        try:
            print("[WebUI] Initializing query processor...")
            self._ner = self._ner or VietnameseNER()
            self._linker = self._linker or self._build_query_linker()
            self.query_proc = QueryProcessor(self._ner, self._linker)
            print("[WebUI] Query processor ready.")
        except Exception as e:
            self.query_proc = None
            print(f"[WebUI] ERROR: failed to initialize query processor: {type(e).__name__}: {e}")

    def startup(self):
        ok, msg = self._try_load_index()
        if ok:
            return

        if _env_flag("ALLOW_LITE_BUILD", False):
            max_docs = int(os.getenv("DEMO_MAX_DOCS", "2000"))
            ok2, msg2 = self._try_build_lite(max_docs=max_docs)
            if ok2:
                return
            msg = f"{msg} | Lite build failed: {msg2}"

        self.state = ServiceState(
            mode="not-ready",
            message=msg,
            index_dir=str(self.index_dir),
            data_path=str(self.data_path),
        )

    def _try_load_index(self) -> Tuple[bool, str]:
        state_pkl = self.index_dir / "state.pkl"
        kg_pkl = self.index_dir / "knowledge_graph.pkl"
        vector_index = self.index_dir / "vector.index"

        if not state_pkl.exists():
            return False, f"Index not found at `{self.index_dir}`. Build it first, then reload."
        if not vector_index.exists():
            return False, f"FAISS index missing: `{vector_index}`. Rebuild index then reload."

        import pickle

        with open(state_pkl, "rb") as f:
            state = pickle.load(f)

        self.documents = state.get("documents", [])
        self.em = EmbeddingManager.from_state(state["embedding"])

        self.ranker = None
        self.importance_scores = {}
        if kg_pkl.exists():
            try:
                with open(kg_pkl, "rb") as f:
                    kg = pickle.load(f)
                self._kg = kg
                self.ranker = GraphRanker()
                self.ranker.compute_pagerank(kg)
                self.importance_scores = self.ranker.compute_importance_scores(kg)
            except Exception:
                self._kg = None
                self.ranker = None
                self.importance_scores = {}

        use_reranker = _env_flag("USE_RERANKER", False)
        self.retriever = Retriever(
            use_faiss=True,
            use_cross_encoder=use_reranker,
            reranker_model_dir=_env_path("RERANKER_DIR"),
            load_cross_encoder=False,
        )

        chunks_dict = state.get("chunks", {})
        doc_to_chunks = state.get("doc_to_chunks", {})
        chunks_list = list(chunks_dict.values()) if isinstance(chunks_dict, dict) else list(chunks_dict)

        self.retriever.attach_state(
            embedding_manager=self.em,
            documents=self.documents,
            chunks=chunks_list,
            doc_to_chunks=doc_to_chunks,
            graph_ranker=self.ranker,
            kg=self._kg,
            importance_scores=state.get("global_scores", self.importance_scores),
            chunk_mode=True,
        )
        try:
            self.retriever.load_artifacts(str(self.index_dir))
        except MemoryError as e:
            return False, (
                "Không đủ RAM để load FAISS index. Hãy đóng bớt process/model "
                f"đang chạy rồi thử lại. Chi tiết: {e}"
            )
        if use_reranker:
            self.retriever.load_reranker()
        self._linker = self._build_query_linker()
        self.query_proc = None
        self._ensure_query_proc()

        self.state = ServiceState(
            mode="index" if self.ranker is not None else "retrieval-only",
            message="Loaded prebuilt index successfully.",
            index_dir=str(self.index_dir),
            data_path=str(self.data_path),
            built_at=time.time(),
            n_docs=len(self.documents),
            n_chunks=len(chunks_list),
        )
        return True, "ok"

    def _try_build_lite(self, max_docs: int = 2000) -> Tuple[bool, str]:
        if not self.data_path.exists():
            return False, f"DATA_PATH not found: `{self.data_path}`"

        docs: List[Dict[str, Any]] = []
        try:
            if self.data_path.suffix.lower() == ".json":
                loader = NewsDataLoader(str(self.data_path))
                docs = loader.load_json()[:max_docs]
            else:
                import csv

                with open(self.data_path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        d = create_document(row)
                        if d:
                            docs.append(d)
                        if len(docs) >= max_docs:
                            break
        except Exception as e:
            return False, f"Failed reading data: {e}"

        if not docs:
            return False, "No documents loaded (check CSV columns: url,date,category,title,text)."

        try:
            chunks, doc_to_chunks = chunk_documents(
                docs, strategy="sentence_window", max_chars=400, overlap=1
            )
            chunk_dicts = [{"id": c["chunk_id"], "full_text": c["chunk_text"]} for c in chunks]

            self.em = EmbeddingManager()
            self.em.build_document_index(chunk_dicts)
            self.retriever = Retriever(
                use_faiss=True,
                use_cross_encoder=_env_flag("USE_RERANKER", False),
                reranker_model_dir=_env_path("RERANKER_DIR"),
                load_cross_encoder=False,
            )
            self.retriever.build(
                chunks=chunks,
                embedding_manager=self.em,
                doc_to_chunks=doc_to_chunks,
                documents=docs,
                graph_ranker=None,
                kg=None,
                importance_scores={},
            )
            if _env_flag("USE_RERANKER", False):
                self.retriever.load_reranker()

            self.documents = docs
            self.ranker = None
            self._kg = None
            self.importance_scores = {}
            self._ensure_query_proc()
            self.state = ServiceState(
                mode="retrieval-only",
                message=f"Lite demo index built from first {len(docs)} docs (no NER/KG).",
                index_dir=str(self.index_dir),
                data_path=str(self.data_path),
                built_at=time.time(),
                n_docs=len(docs),
                n_chunks=len(chunks),
            )
            return True, "ok"
        except Exception as e:
            return False, f"Failed building lite index: {e}"

    def analyze_query(self, query: str) -> Dict[str, Any]:
        self._ensure_query_proc()
        if not self.query_proc:
            return {
                "original": query,
                "normalized": query.strip(),
                "entities": [],
                "keywords": [],
                "topic": None,
                "year_filter": None,
                "intent": "news_search",
            }
        return self.query_proc.process(query)

    def search(self, query: str, top_k: int = 10, mode: str = "full", page: int = 1) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
        if not self.retriever or not self.em:
            return [], 0.0, {}
        t0 = time.time()
        seed_entities: List[str] = []
        analysis = self.analyze_query(query)
        if self.query_proc:
            seed_entities = self.query_proc.get_query_entity_names(analysis)
        rerank = mode in {"vector-rerank", "full"}
        use_graph = mode in {"vector-graph", "full"}
        use_decay = mode == "full"
        if not use_graph:
            seed_entities = []
        fetch_k = max(top_k * page, top_k)
        results = self.retriever.retrieve(
            query,
            top_k=fetch_k,
            seed_entities=seed_entities,
            rerank=rerank,
            apply_decay=use_decay,
            use_graph_boost=use_graph,
        )
        start = max(0, (page - 1) * top_k)
        end = start + top_k
        return results[start:end], time.time() - t0, analysis

    def compare_modes(self, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        modes = ["vector-only", "vector-graph", "vector-rerank", "full"]
        out = {}
        for mode in modes:
            results, _, _ = self.search(query, top_k=top_k, mode=mode)
            out[mode] = results
        return out

    def answer(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "full",
        max_context_docs: int = 5,
    ) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any], Dict[str, Any]]:
        results, elapsed, analysis = self.search(query, top_k=top_k, mode=mode)
        answer_payload = self.reader.answer(
            query,
            results,
            max_context_docs=max_context_docs,
        )
        return results, elapsed, analysis, answer_payload

    def _make_search_id(
        self,
        query: str,
        top_k: int,
        mode: str,
        compare_enabled: bool,
    ) -> str:
        normalized = " ".join((query or "").strip().split()).lower()
        raw = f"{normalized}\0{int(top_k)}\0{mode}\0{int(compare_enabled)}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    def _prune_search_cache(self):
        now = time.time()
        expired = [
            key
            for key, entry in self._search_cache.items()
            if now - float(entry.get("created_at", 0.0)) > self._search_cache_ttl_seconds
        ]
        for key in expired:
            self._search_cache.pop(key, None)

    def cache_search(
        self,
        query: str,
        top_k: int,
        mode: str,
        compare_enabled: bool,
        results: List[Dict[str, Any]],
        elapsed: float,
        analysis: Dict[str, Any],
        compare_results: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._prune_search_cache()
        search_id = self._make_search_id(query, top_k, mode, compare_enabled)
        self._search_cache[search_id] = {
            "created_at": time.time(),
            "query": query,
            "top_k": int(top_k),
            "mode": mode,
            "compare": bool(compare_enabled),
            "results": results,
            "elapsed": float(elapsed),
            "analysis": analysis,
            "compare_results": compare_results,
        }
        return search_id

    def get_cached_search(
        self,
        search_id: Optional[str],
        query: str,
        top_k: int,
        mode: str,
        compare_enabled: bool,
    ) -> Optional[Dict[str, Any]]:
        if not search_id:
            return None
        self._prune_search_cache()
        expected_id = self._make_search_id(query, top_k, mode, compare_enabled)
        if search_id != expected_id:
            return None
        entry = self._search_cache.get(search_id)
        if not entry:
            return None
        if (
            entry.get("query") != query
            or int(entry.get("top_k", 0)) != int(top_k)
            or entry.get("mode") != mode
            or bool(entry.get("compare")) != bool(compare_enabled)
        ):
            return None
        return entry

    @staticmethod
    def _format_entity_label(mention: Optional[str], canonical: Optional[str]) -> str:
        mention = (mention or "").strip()
        canonical = (canonical or "").strip()
        if mention and canonical and mention.lower() != canonical.lower():
            return f"{mention} -> {canonical}"
        return canonical or mention or "entity"

    def _expand_query_entities(
        self,
        query: str,
        analysis: Dict[str, Any],
        query_entities: List[Dict[str, Any]],
        seed_entities: List[str],
    ) -> List[Dict[str, Any]]:
        expanded: List[Dict[str, Any]] = []
        seen = set()

        def add_entity(item: Dict[str, Any]):
            key = item.get("canonical") or item.get("label") or item.get("id")
            if not key or key in seen:
                return
            seen.add(key)
            expanded.append(item)

        for idx, ent in enumerate(query_entities):
            score = float(ent.get("score", ent.get("link_score", 0.5)))
            mention = (
                ent.get("mention")
                or ent.get("surface_form")
                or ent.get("text")
                or ent.get("label")
                or ent.get("canonical")
                or f"entity_{idx}"
            )
            canonical = ent.get("canonical") or mention
            add_entity({
                "id": ent.get("id", f"entity::{ent.get('label', idx)}"),
                "label": self._format_entity_label(mention, canonical),
                "mention": mention,
                "canonical": canonical,
                "entity_type": ent.get("entity_type", ent.get("type", "MISC")),
                "score": score,
                "source": "query",
                "match_type": ent.get("match_type", "query"),
                "aliases": ent.get("aliases", []),
            })

            for alias in (ent.get("aliases") or [])[:3]:
                if alias and alias != canonical:
                    add_entity({
                        "id": f"entity::{alias}",
                        "label": self._format_entity_label(alias, canonical),
                        "mention": alias,
                        "canonical": canonical,
                        "entity_type": ent.get("entity_type", ent.get("type", "MISC")),
                        "score": max(0.35, score * 0.85),
                        "source": "expanded",
                        "match_type": "alias",
                        "aliases": [alias],
                    })

        topic = (analysis or {}).get("topic") if analysis else None
        keywords = (analysis or {}).get("keywords", []) if analysis else []
        if not expanded:
            for idx, kw in enumerate(keywords[:4]):
                add_entity({
                    "id": f"entity::kw::{kw}",
                    "label": kw,
                    "mention": kw,
                    "canonical": kw,
                    "entity_type": "KEYWORD",
                    "score": max(0.25, 0.55 - idx * 0.06),
                    "source": "fallback",
                    "match_type": "keyword",
                    "aliases": [kw],
                })
        if topic:
            add_entity({
                "id": f"entity::topic::{topic}",
                "label": topic,
                "mention": topic,
                "canonical": topic,
                "entity_type": "TOPIC",
                "score": 0.42,
                "source": "fallback",
                "match_type": "topic",
                "aliases": [topic],
            })

        for idx, seed in enumerate(seed_entities[:6]):
            add_entity({
                "id": f"entity::seed::{seed}",
                "label": seed,
                "mention": seed,
                "canonical": seed,
                "entity_type": "SEED",
                "score": 0.5 - idx * 0.03,
                "source": "seed",
                "match_type": "seed",
                "aliases": [seed],
            })

        return expanded

    def _build_expanded_ner(
        self,
        analysis: Dict[str, Any],
        seed_entities: List[str],
    ) -> List[Dict[str, Any]]:
        """Tạo danh sách entity mở rộng phục vụ graph preview.

        Mỗi item cần đủ metadata để frontend giải thích vì sao entity xuất hiện:
        - source: query / expanded / fallback / seed
        - match_type: query / alias / keyword / topic / seed
        - aliases: các biến thể tên gọi
        - reason: mô tả ngắn bằng tiếng Việt
        - evidence: nguồn gốc cụ thể từ query/keyword/topic/seed
        """
        entities = analysis.get("entities", []) if analysis else []
        keywords = analysis.get("keywords", []) if analysis else []
        topic = analysis.get("topic") if analysis else None
        expanded: List[Dict[str, Any]] = []
        seen = set()

        def add(item: Dict[str, Any]):
            key = item.get("canonical") or item.get("label") or item.get("id")
            if not key or key in seen:
                return
            seen.add(key)
            expanded.append(item)

        for idx, ent in enumerate(entities):
            canonical = ent.get("canonical") or ent.get("text") or f"entity_{idx}"
            mention = ent.get("text") or canonical
            score = float(ent.get("link_score", 0.5))
            entity_type = ent.get("type", "MISC")
            aliases = ent.get("aliases", []) or []
            add({
                "id": ent.get("entity_id") or f"entity::{canonical}",
                "label": self._format_entity_label(mention, canonical),
                "mention": mention,
                "canonical": canonical,
                "entity_type": entity_type,
                "score": score,
                "source": "query",
                "match_type": ent.get("match_type", "query"),
                "aliases": aliases,
                "reason": (
                    f"Entity này xuất hiện trực tiếp trong query. Mention '{mention}' "
                    f"được link về canonical '{canonical}'."
                ),
                "evidence": {
                    "kind": "query_entity",
                    "mention": mention,
                    "canonical": canonical,
                },
            })
            for alias in aliases[:4]:
                if alias and alias != canonical:
                    add({
                        "id": f"entity::alias::{alias}",
                        "label": self._format_entity_label(alias, canonical),
                        "mention": alias,
                        "canonical": canonical,
                        "entity_type": entity_type,
                        "score": max(0.35, score * 0.85),
                        "source": "expanded",
                        "match_type": "alias",
                        "aliases": [alias],
                        "reason": f"Đây là biến thể tên gọi/alias của entity gốc '{canonical}'.",
                        "evidence": {
                            "kind": "alias",
                            "base_entity": canonical,
                            "alias": alias,
                        },
                    })

        if not expanded:
            for idx, kw in enumerate(keywords[:4]):
                add({
                    "id": f"entity::kw::{kw}",
                    "label": kw,
                    "mention": kw,
                    "canonical": kw,
                    "entity_type": "KEYWORD",
                    "score": max(0.25, 0.55 - idx * 0.06),
                    "source": "fallback",
                    "match_type": "keyword",
                    "aliases": [kw],
                    "reason": "Query không có entity rõ ràng nên hệ thống dùng keyword quan trọng để tạo nút graph ban đầu.",
                    "evidence": {
                        "kind": "keyword",
                        "keyword": kw,
                    },
                })

        if topic:
            add({
                "id": f"entity::topic::{topic}",
                "label": topic,
                "mention": topic,
                "canonical": topic,
                "entity_type": "TOPIC",
                "score": 0.42,
                "source": "fallback",
                "match_type": "topic",
                "aliases": [topic],
                "reason": f"Topic '{topic}' được suy ra từ bộ từ khóa của query và dùng để hỗ trợ mở rộng graph.",
                "evidence": {
                    "kind": "topic",
                    "topic": topic,
                },
            })

        for idx, seed in enumerate(seed_entities[:6]):
            add({
                "id": f"entity::seed::{seed}",
                "label": seed,
                "mention": seed,
                "canonical": seed,
                "entity_type": "SEED",
                "score": max(0.15, 0.5 - idx * 0.03),
                "source": "seed",
                "match_type": "seed",
                "aliases": [seed],
                "reason": "Entity này được chọn làm seed để truy hồi theo ngữ cảnh query và mở rộng graph.",
                "evidence": {
                    "kind": "seed",
                    "seed_entity": seed,
                },
            })

        return expanded

    def _build_graph_payload(
        self,
        query: str,
        analysis: Dict[str, Any],
        results: List[Dict[str, Any]],
        seed_entities: List[str],
        mode: str,
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen_nodes = set()

        def add_node(node_id: str, kind: str, label: str, **extra: Any):
            if node_id in seen_nodes:
                return
            payload = {"id": node_id, "kind": kind, "label": label}
            payload.update(extra)
            nodes.append(payload)
            seen_nodes.add(node_id)

        def make_entity_node(
            source: Dict[str, Any],
            *,
            kind: str,
            idx: int,
            score: float,
            source_label: str,
        ) -> Dict[str, Any]:
            name = source.get("canonical") or source.get("text") or source.get("label") or f"entity_{idx}"
            mention = (
                source.get("mention")
                or source.get("surface_form")
                or source.get("text")
                or source.get("label")
                or name
            )
            canonical = source.get("canonical") or name
            entity_id = source.get("entity_id") or source.get("id") or f"entity::{name}"
            return {
                "id": entity_id,
                "kind": "entity",
                "label": source.get("label") or self._format_entity_label(mention, canonical),
                "mention": mention,
                "canonical": canonical,
                "entity_type": source.get("entity_type") or source.get("type", "MISC"),
                "score": float(score),
                "source": source_label,
                "match_type": source.get("match_type", kind),
                "aliases": source.get("aliases", []),
                "reason": source.get("reason", ""),
                "evidence": source.get("evidence", {}),
            }

        add_node("query", "query", query or "query", x=0, y=0, fx=0, fy=0, highlight=True)

        entities = analysis.get("entities", []) if analysis else []
        query_entities: List[Dict[str, Any]] = []
        if entities:
            for idx, ent in enumerate(entities[:10]):
                node = make_entity_node(ent, kind=ent.get("match_type", "query"), idx=idx, score=ent.get("link_score", 0.5), source_label="query")
                query_entities.append(node)

        if seed_entities:
            for idx, name in enumerate(seed_entities[:6]):
                if any((qe.get("canonical") == name or qe.get("label") == name) for qe in query_entities):
                    continue
                query_entities.append(
                    {
                        "id": f"entity::seed::{name}",
                        "kind": "entity",
                        "label": name,
                        "mention": name,
                        "canonical": name,
                        "entity_type": "SEED",
                        "score": max(0.2, 0.6 - idx * 0.05),
                        "source": "fallback",
                        "match_type": "seed",
                        "aliases": [name],
                        "reason": "Seed entity được suy ra từ query để kích hoạt graph expansion.",
                        "evidence": {"kind": "seed", "seed_entity": name},
                    }
                )

        if analysis and analysis.get("keywords") and not query_entities:
            for idx, kw in enumerate(analysis.get("keywords", [])[:4]):
                query_entities.append(
                    {
                        "id": f"entity::kw::{kw}",
                        "kind": "entity",
                        "label": kw,
                        "mention": kw,
                        "canonical": kw,
                        "entity_type": "KEYWORD",
                        "score": max(0.25, 0.55 - idx * 0.06),
                        "source": "fallback",
                        "match_type": "keyword",
                        "aliases": [kw],
                        "reason": "Query không có entity rõ ràng nên dùng keyword để dựng graph.",
                        "evidence": {"kind": "keyword", "keyword": kw},
                    }
                )

        expanded_entities = self._expand_query_entities(query, analysis, query_entities, seed_entities)
        if not expanded_entities and analysis and analysis.get("keywords"):
            for idx, kw in enumerate(analysis.get("keywords", [])[:4]):
                expanded_entities.append({
                    "id": f"entity::kw::{kw}",
                    "label": kw,
                    "mention": kw,
                    "canonical": kw,
                    "entity_type": "KEYWORD",
                    "score": max(0.25, 0.55 - idx * 0.06),
                    "source": "fallback",
                    "match_type": "keyword",
                    "aliases": [kw],
                    "reason": "Query không có entity rõ ràng nên hệ thống dùng keyword quan trọng để tạo nút graph ban đầu.",
                    "evidence": {"kind": "keyword", "keyword": kw},
                })
        for idx, ent in enumerate(expanded_entities):
            add_node(
                ent["id"],
                "entity",
                ent["label"],
                mention=ent.get("mention", ent["label"]),
                canonical=ent.get("canonical", ent["label"]),
                entity_type=ent.get("entity_type", "MISC"),
                score=ent.get("score", 0.5),
                source=ent.get("source", "query"),
                match_type=ent.get("match_type", "expanded"),
                aliases=ent.get("aliases", []),
                reason=ent.get("reason", ""),
                evidence=ent.get("evidence", {}),
                x=0,
                y=0,
            )
            edges.append({"source": "query", "target": ent["id"], "strength": float(ent.get("score", 0.5)), "kind": "query-entity"})

        entity_names = [ent["id"] for ent in expanded_entities]
        top_ids = set()
        for rank, doc in enumerate(results[:10], start=1):
            doc_id = doc.get("id") or f"doc_{rank}"
            top_ids.add(doc_id)
            label = doc.get("title") or doc_id
            add_node(
                f"doc::{doc_id}",
                "document",
                label,
                rank=rank,
                score=float(doc.get("retrieval_score", 0.0)),
                title=doc.get("title", ""),
                url=doc.get("url", ""),
                x=0,
                y=0,
            )
            if entity_names:
                target_entity = entity_names[(rank - 1) % len(entity_names)]
                edges.append({
                    "source": target_entity,
                    "target": f"doc::{doc_id}",
                    "strength": float(doc.get("retrieval_score", 0.0)),
                    "kind": "entity-document",
                })
            else:
                edges.append({"source": "query", "target": f"doc::{doc_id}", "strength": float(doc.get("retrieval_score", 0.0)), "kind": "query-document"})

        graph_entity_reason = {
            ent["id"]: {
                "reason": ent.get("reason", ""),
                "source": ent.get("source", "query"),
                "match_type": ent.get("match_type", ""),
                "mention": ent.get("mention", ent["label"]),
                "canonical": ent.get("canonical", ent["label"]),
                "aliases": ent.get("aliases", []),
                "entity_type": ent.get("entity_type", "MISC"),
                "score": ent.get("score", 0.5),
                "evidence": ent.get("evidence", {}),
            }
            for ent in expanded_entities
        }

        return {
            "query": query,
            "mode": mode,
            "analysis": analysis,
            "nodes": nodes,
            "edges": edges,
            "results": results,
            "top_ids": list(top_ids),
            "seed_entities": seed_entities,
            "ner_expansion": expanded_entities,
            "graph_entity_reason": graph_entity_reason,
        }


index_dir = Path(os.getenv("INDEX_DIR", str(DEFAULT_INDEX_DIR)))
data_path = Path(os.getenv("DATA_PATH", str(DEFAULT_DATA_PATH)))
service = WebSearchService(index_dir=index_dir, data_path=data_path)

app = FastAPI(title="Vietnamese News Search Demo")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def _startup():
    service.startup()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    analysis = None
    ner_expansion = []
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "state": service.state,
            "query": "",
            "top_k": 10,
            "results": [],
            "elapsed_ms": None,
            "compare": False,
            "mode": "full",
            "analysis": analysis,
            "compare_results": None,
            "ner_expansion": ner_expansion,
            "graph_payload": None,
            "answer_payload": None,
        },
    )


@app.post("/search", response_class=HTMLResponse)
def search(
    request: Request,
    query: str = Form(...),
    top_k: int = Form(10),
    mode: str = Form("full"),
    compare: Optional[str] = Form(None),
):
    query = (query or "").strip()
    top_k = max(1, min(int(top_k or 10), 50))
    mode = mode if mode in {"vector-only", "vector-graph", "vector-rerank", "full"} else "full"
    results: List[Dict[str, Any]] = []
    elapsed_ms: Optional[int] = None
    analysis = service.analyze_query(query) if query else None
    compare_results = None
    compare_enabled = bool(compare)
    graph_payload = None
    answer_payload = None
    if query:
        if service.state.mode != "not-ready":
            if compare_enabled:
                compare_results = service.compare_modes(query, top_k=min(top_k, 5))
                results = compare_results.get(mode, [])
                elapsed_ms = 0
            else:
                results, elapsed, analysis = service.search(query, top_k=top_k, mode=mode)
                elapsed_ms = int(elapsed * 1000)
        seed_entities = service.query_proc.get_query_entity_names(analysis) if service.query_proc and analysis else []
        graph_payload = service._build_graph_payload(
            query=query,
            analysis=analysis or {},
            results=results,
            seed_entities=seed_entities,
            mode=mode,
        )
        graph_payload["elapsed_ms"] = elapsed_ms
        graph_payload["compare"] = compare_enabled
        graph_payload["compare_results"] = compare_results
        graph_payload["state"] = service.state.__dict__
        graph_payload["answer"] = answer_payload

    ner_expansion = service._build_expanded_ner(analysis, service.query_proc.get_query_entity_names(analysis) if service.query_proc else []) if query else []
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "state": service.state,
            "query": query,
            "top_k": top_k,
            "results": results,
            "elapsed_ms": elapsed_ms,
            "compare": compare_enabled,
            "mode": mode,
            "analysis": analysis,
            "compare_results": compare_results,
            "ner_expansion": ner_expansion,
            "graph_payload": graph_payload,
            "answer_payload": answer_payload,
        },
    )


@app.get("/health")
def health():
    return JSONResponse(
        {
            "status": "ok",
            "mode": service.state.mode,
            "docs": service.state.n_docs,
            "chunks": service.state.n_chunks,
            "reranker_enabled": _env_flag("USE_RERANKER", False),
            "reranker_loaded": bool(
                service.retriever and getattr(service.retriever, "_reranker", None)
            ),
            "reranker_dir": _env_path("RERANKER_DIR"),
            "reranker_model": getattr(
                getattr(service.retriever, "_reranker", None), "model_dir", None
            )
            if service.retriever
            else None,
            "retriever_has_kg": bool(
                service.retriever and getattr(service.retriever, "_kg", None)
            ),
            "search_cache_size": len(service._search_cache),
            "lite_build_enabled": _env_flag("ALLOW_LITE_BUILD", False),
            "query_understanding": service.query_proc is not None,
        }
    )


@app.get("/api/status")
def api_status():
    return health()


@app.get("/api/analysis")
def api_analysis(query: str = ""):
    query = (query or "").strip()
    if not query:
        return JSONResponse({"error": "missing_query"}, status_code=400)
    return JSONResponse(service.analyze_query(query))


@app.post("/api/graph")
def api_graph(
    query: str = Form(...),
    top_k: int = Form(10),
    mode: str = Form("full"),
    compare: Optional[str] = Form(None),
):
    query = (query or "").strip()
    top_k = max(1, min(int(top_k or 10), 50))
    mode = mode if mode in {"vector-only", "vector-graph", "vector-rerank", "full"} else "full"
    if not query:
        return JSONResponse({"error": "missing_query"}, status_code=400)
    if service.state.mode == "not-ready":
        return JSONResponse({"error": "not_ready", "state": service.state.__dict__}, status_code=503)

    compare_enabled = bool(compare)
    results, elapsed, analysis = service.search(query, top_k=top_k, mode=mode)
    compare_results = service.compare_modes(query, top_k=min(top_k, 5)) if compare_enabled else None
    search_id = service.cache_search(
        query=query,
        top_k=top_k,
        mode=mode,
        compare_enabled=compare_enabled,
        results=results,
        elapsed=elapsed,
        analysis=analysis,
        compare_results=compare_results,
    )
    seed_entities = service.query_proc.get_query_entity_names(analysis) if service.query_proc else []
    payload = service._build_graph_payload(
        query=query,
        analysis=analysis,
        results=results,
        seed_entities=seed_entities,
        mode=mode,
    )
    payload.update({
        "elapsed_ms": int(elapsed * 1000),
        "compare": compare_enabled,
        "compare_results": compare_results,
        "state": service.state.__dict__,
        "search_id": search_id,
    })
    return JSONResponse(payload)


@app.post("/api/ask")
def api_ask(
    query: str = Form(...),
    top_k: int = Form(10),
    mode: str = Form("full"),
    compare: Optional[str] = Form(None),
    search_id: Optional[str] = Form(None),
):
    query = (query or "").strip()
    top_k = max(1, min(int(top_k or 10), 50))
    mode = mode if mode in {"vector-only", "vector-graph", "vector-rerank", "full"} else "full"
    if not query:
        return JSONResponse({"error": "missing_query"}, status_code=400)
    if service.state.mode == "not-ready":
        return JSONResponse({"error": "not_ready", "state": service.state.__dict__}, status_code=503)

    compare_enabled = bool(compare)
    cached = service.get_cached_search(
        search_id=search_id,
        query=query,
        top_k=top_k,
        mode=mode,
        compare_enabled=compare_enabled,
    )
    if cached:
        results = cached.get("results", [])
        elapsed = float(cached.get("elapsed", 0.0) or 0.0)
        analysis = cached.get("analysis") or service.analyze_query(query)
        compare_results = cached.get("compare_results")
        answer_payload = service.reader.answer(query, results)
    elif compare_enabled:
        compare_results = service.compare_modes(query, top_k=min(top_k, 5))
        results = compare_results.get(mode, [])
        elapsed = 0.0
        analysis = service.analyze_query(query)
        answer_payload = service.reader.answer(query, results)
    else:
        results, elapsed, analysis, answer_payload = service.answer(
            query,
            top_k=top_k,
            mode=mode,
        )
        compare_results = None

    seed_entities = service.query_proc.get_query_entity_names(analysis) if service.query_proc else []
    payload = service._build_graph_payload(
        query=query,
        analysis=analysis,
        results=results,
        seed_entities=seed_entities,
        mode=mode,
    )
    payload.update({
        "elapsed_ms": int(elapsed * 1000),
        "compare": compare_enabled,
        "compare_results": compare_results,
        "state": service.state.__dict__,
        "answer": answer_payload,
        "search_id": search_id,
        "cache_hit": bool(cached),
    })
    return JSONResponse(payload)
