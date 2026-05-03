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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.data_loader import NewsDataLoader, create_document
from src.graph.ranking import GraphRanker
from src.preprocessing.entity_linking import EntityLinker
from src.preprocessing.ner import VietnameseNER
from src.retrieval import EmbeddingManager, Retriever, chunk_documents, QueryProcessor

ROOT_DIR = Path(__file__).resolve().parent
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
        self._ner: Optional[VietnameseNER] = None
        self._linker: Optional[EntityLinker] = None
        self._kg = None
        self.state = ServiceState(
            mode="not-ready",
            message="Service is starting...",
            index_dir=str(index_dir),
            data_path=str(data_path),
        )

    def _ensure_query_proc(self):
        if self.query_proc is not None or self.em is None:
            return
        try:
            self._ner = self._ner or VietnameseNER()
            self._linker = self._linker or EntityLinker(shared_encoder=self.em._enc)
            self.query_proc = QueryProcessor(self._ner, self._linker)
        except Exception:
            self.query_proc = None

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
        self.retriever = Retriever(use_faiss=True, use_cross_encoder=use_reranker)

        chunks_dict = state.get("chunks", {})
        doc_to_chunks = state.get("doc_to_chunks", {})
        chunks_list = list(chunks_dict.values()) if isinstance(chunks_dict, dict) else list(chunks_dict)

        self.retriever.attach_state(
            embedding_manager=self.em,
            documents=self.documents,
            chunks=chunks_list,
            doc_to_chunks=doc_to_chunks,
            graph_ranker=self.ranker,
            kg=None,
            importance_scores=state.get("global_scores", self.importance_scores),
            chunk_mode=True,
        )
        self.retriever.load_artifacts(str(self.index_dir))
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
            self.retriever = Retriever(use_faiss=True, use_cross_encoder=_env_flag("USE_RERANKER", False))
            self.retriever.build(
                chunks=chunks,
                embedding_manager=self.em,
                doc_to_chunks=doc_to_chunks,
                documents=docs,
                graph_ranker=None,
                kg=None,
                importance_scores={},
            )

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

    def _build_graph_payload(
        self,
        query: str,
        analysis: Dict[str, Any],
        results: List[Dict[str, Any]],
        seed_entities: List[str],
        mode: str,
    ) -> List[Dict[str, Any]]:
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

        add_node("query", "query", query or "query", x=0, y=0, fx=0, fy=0, highlight=True)

        entities = analysis.get("entities", []) if analysis else []
        entity_names = []
        for idx, ent in enumerate(entities[:10]):
            name = ent.get("canonical") or ent.get("text") or f"entity_{idx}"
            entity_names.append(name)
            node_id = f"entity::{name}"
            strength = float(ent.get("link_score", 0.5))
            add_node(
                node_id,
                "entity",
                name,
                entity_type=ent.get("type", "MISC"),
                score=strength,
                x=140 + idx * 12,
                y=40 + idx * 8,
            )
            edges.append({"source": "query", "target": node_id, "strength": strength, "kind": "query-entity"})

        if not entity_names and seed_entities:
            for idx, name in enumerate(seed_entities[:6]):
                node_id = f"entity::{name}"
                add_node(node_id, "entity", name, entity_type="SEED", score=0.6, x=160 + idx * 10, y=60 + idx * 10)
                edges.append({"source": "query", "target": node_id, "strength": 0.6, "kind": "query-entity"})
                entity_names.append(name)

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
                x=260 + rank * 10,
                y=100 + rank * 12,
            )
            if entity_names:
                target_entity = entity_names[(rank - 1) % len(entity_names)]
                edges.append({
                    "source": f"entity::{target_entity}",
                    "target": f"doc::{doc_id}",
                    "strength": float(doc.get("retrieval_score", 0.0)),
                    "kind": "entity-document",
                })
            else:
                edges.append({"source": "query", "target": f"doc::{doc_id}", "strength": float(doc.get("retrieval_score", 0.0)), "kind": "query-document"})

        return {
            "query": query,
            "mode": mode,
            "analysis": analysis,
            "nodes": nodes,
            "edges": edges,
            "results": results,
            "top_ids": list(top_ids),
            "seed_entities": seed_entities,
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
            "analysis": None,
            "compare_results": None,
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
    if query and service.state.mode != "not-ready":
        if compare_enabled:
            compare_results = service.compare_modes(query, top_k=min(top_k, 5))
            results = compare_results.get(mode, [])
            elapsed_ms = 0
        else:
            results, elapsed, analysis = service.search(query, top_k=top_k, mode=mode)
            elapsed_ms = int(elapsed * 1000)

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
    analysis = service.analyze_query(query)
    results, elapsed, analysis = service.search(query, top_k=top_k, mode=mode)
    payload = service._build_graph_payload(
        query=query,
        analysis=analysis,
        results=results,
        seed_entities=service.query_proc.get_query_entity_names(analysis) if service.query_proc else [],
        mode=mode,
    )
    payload.update({
        "elapsed_ms": int(elapsed * 1000),
        "compare": compare_enabled,
        "compare_results": service.compare_modes(query, top_k=min(top_k, 5)) if compare_enabled else None,
        "state": service.state.__dict__,
    })
    return JSONResponse(payload)
