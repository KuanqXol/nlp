"""
main.py — Vietnamese KG-Enhanced News Search System

Pipeline:
  1. DataLoader: load CSV/JSON VnExpress
  2. PhoBERT NER fine-tuned trên VLSP2016
  3. EntityLinker + RelationExtractor (rule-based, cho graph)
  4. KnowledgeGraph + PPR query expansion
  5. FAISS vector search (vietnamese-bi-encoder) top-50 chunks
  6. Cross-encoder rerank → top-10
  7. Trả về link + title + snippet

Cách chạy:
    python main.py                                       # interactive
    python main.py --query "kinh tế việt nam"            # one-shot
    python main.py --load-index                          # load index đã build
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).parent / "src"
DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = DATA_DIR / "index"
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import NewsDataLoader
from src.preprocessing import (
    VietnameseNER,
    EntityLinker,
    RelationExtractor,
    ner_with_checkpoint,
    resolve_coreference,
)
from src.graph import (
    KnowledgeGraph,
    GraphRanker,
    SimilarityGraphBuilder,
    KnowledgeGraphVisualizer,
)
from src.retrieval import (
    EmbeddingManager,
    Retriever,
    chunk_documents,
    QueryProcessor,
    QueryExpander,
)


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════╗
║   🇻🇳  Vietnamese KG-Enhanced News Search System         ║
║   PhoBERT NER + Knowledge Graph + FAISS + Reranking     ║
╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Lệnh hỗ trợ:
  <query>        Tìm kiếm tin tức
  :kg            Thống kê Knowledge Graph
  :top           Top entity quan trọng nhất
  :suggest       Gợi ý query phổ biến từ KG
  :viz           Xuất visualization KG
  :help          Hiển thị trợ giúp
  :quit / :exit  Thoát

Ví dụ:
  chiến tranh nga ukraine
  Samsung đầu tư Việt Nam
  WHO cảnh báo COVID-19
"""


# ── Popular Query Suggestions ─────────────────────────────────────────────────


def get_popular_queries(
    kg: KnowledgeGraph, graph_ranker: GraphRanker, n: int = 8
) -> list:
    """
    Lấy gợi ý query từ KG dựa trên PageRank score.
    Trả về list query string từ top entity.
    """
    try:
        scores = graph_ranker.get_importance_scores(kg)
        top_entities = sorted(scores.items(), key=lambda x: -x[1])[: n * 2]

        queries = []
        seen_types = {}
        for entity, score in top_entities:
            etype = (
                kg._graph.nodes.get(entity, {}).get("type", "MISC")
                if hasattr(kg, "_graph")
                else "MISC"
            )
            # Giới hạn tối đa 3 entity mỗi loại để đa dạng
            seen_types[etype] = seen_types.get(etype, 0) + 1
            if seen_types[etype] <= 3:
                queries.append(entity)
            if len(queries) >= n:
                break
        return queries
    except Exception:
        return []


# ── Display Results ───────────────────────────────────────────────────────────


def display_results(results: list, query: str, elapsed: float):
    """Hiển thị kết quả tìm kiếm: rank, title, link, snippet."""
    print(f"\n{'─'*60}")
    print(f'🔍  Kết quả cho: "{query}"  ({elapsed*1000:.0f}ms)')
    print(f"{'─'*60}")

    if not results:
        print("  Không tìm thấy kết quả phù hợp.")
        return

    for i, doc in enumerate(results, 1):
        title = doc.get("title", "(không có tiêu đề)")
        url = doc.get("url", "")
        date = doc.get("date", "")
        cat = doc.get("category", "")
        score = doc.get("retrieval_score", 0.0)

        # Snippet: ưu tiên chunk_text, fallback content
        snippet = doc.get("chunk_text", "") or doc.get("content", "")
        if snippet:
            snippet = snippet[:160].replace("\n", " ").strip()
            if len(doc.get("chunk_text", doc.get("content", ""))) > 160:
                snippet += "..."

        print(f"\n  [{i}] {title}")
        if url:
            print(f"      🔗 {url}")
        meta_parts = []
        if date:
            meta_parts.append(date)
        if cat:
            meta_parts.append(cat)
        meta_parts.append(f"score={score:.3f}")
        print(f"      📌 {' · '.join(meta_parts)}")
        if snippet:
            print(f"      {snippet}")

    print(f"\n{'─'*60}")


# ── NewsSearchSystem ──────────────────────────────────────────────────────────


class NewsSearchSystem:
    def __init__(
        self,
        data_path: str = None,
        use_faiss: bool = True,
        index_dir: str = None,
        ner_model_dir: str = None,
        reranker_model_dir: str = None,
        # backward-compat kwargs bỏ qua
        **kwargs,
    ):
        self.data_path = data_path or str(DATA_DIR / "vnexpress_articles.csv")
        self._index_dir = Path(index_dir or INDEX_DIR)

        print(BANNER)
        print("🔧 Đang khởi tạo hệ thống...\n")

        self.ner = VietnameseNER(model_dir=ner_model_dir)
        self.linker = EntityLinker()
        self.rel_extractor = RelationExtractor()
        self.kg = KnowledgeGraph()
        self.em = EmbeddingManager()
        self.ranker = GraphRanker()
        self.query_proc = QueryProcessor(self.ner, self.linker)
        self.retriever = Retriever(
            use_faiss=use_faiss, reranker_model_dir=reranker_model_dir
        )

        self._expander: QueryExpander = None
        self._documents = []
        self._built = False

    # ── Build pipeline ────────────────────────────────────────────────────

    def build(self):
        """Build toàn bộ pipeline từ raw data."""
        print(f"📂 Đang load data từ: {self.data_path}")
        loader = NewsDataLoader()
        ext = Path(self.data_path).suffix.lower()
        self._documents = (
            loader.load_csv(self.data_path)
            if ext == ".csv"
            else loader.load_json(self.data_path)
        )
        print(f"   Đã load {len(self._documents)} bài báo.")

        # NER với checkpoint để resume nếu bị ngắt
        print("\n🏷️  Đang chạy NER...")
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._documents = ner_with_checkpoint(
            self._documents,
            self.ner,
            checkpoint_path=str(self._index_dir / "ner_checkpoint.json"),
            cache_path=str(self._index_dir / "ner_cache.json"),
            results_path=str(self._index_dir / "ner_results.jsonl"),
        )

        # Entity linking
        print("\n🔗 Đang link entities...")
        for doc in self._documents:
            doc["linked_entities"] = self.linker.link_entities(doc.get("entities", []))

        # Coreference resolution
        self._documents = resolve_coreference(self._documents)

        # Relation extraction (rule-based cho graph)
        print("\n🕸️  Đang trích xuất quan hệ và xây dựng KG...")
        for doc in self._documents:
            triples = self.rel_extractor.extract_relations(
                doc.get("linked_entities", []),
                doc.get("full_text", doc.get("content", "")),
                doc_id=doc.get("id", ""),
                date=doc.get("date", ""),
            )
            for triple in triples:
                self.kg.add_triple(triple)

        # Similarity edges (entity embedding similarity)
        sim_builder = SimilarityGraphBuilder(self.em)
        sim_builder.add_similar_edges(self.kg)

        # PageRank
        print("\n📊 Đang tính PageRank...")
        self.ranker.compute_pagerank(self.kg)
        importance_scores = self.ranker.get_importance_scores(self.kg)

        # Query expander
        self._expander = QueryExpander(self.kg, self.ranker)

        # Chunking + embedding
        print("\n📦 Đang chunking và embedding...")
        chunks, doc_to_chunks = chunk_documents(
            self._documents, strategy="sentence_window", max_chars=400
        )
        chunk_dicts = [
            {"id": c["chunk_id"], "full_text": c["chunk_text"]} for c in chunks
        ]
        self.em.build_document_index(chunk_dicts)

        # Build FAISS index
        self.retriever.build(
            chunks,
            self.em,
            doc_to_chunks,
            self._documents,
            graph_ranker=self.ranker,
            kg=self.kg,
            importance_scores=importance_scores,
        )

        self._built = True
        print("\n✅ Build hoàn tất!\n")

    # ── Save / Load ───────────────────────────────────────────────────────

    def save_index(self, index_dir: str = None):
        target = Path(index_dir or self._index_dir)
        target.mkdir(parents=True, exist_ok=True)

        state = {
            "documents": self._documents,
            "embedding": self.em.export_state(),
            "chunks": self.retriever._chunks,
            "doc_to_chunks": self.retriever._doc_to_chunks,
            "global_scores": self.retriever._global_scores,
        }
        with open(target / "state.pkl", "wb") as f:
            pickle.dump(state, f, protocol=4)

        with open(target / "knowledge_graph.pkl", "wb") as f:
            pickle.dump(self.kg, f, protocol=4)

        self.retriever.save_artifacts(str(target))
        print(f"💾 Index lưu tại: {target}")

    def load_index(self, index_dir: str = None):
        target = Path(index_dir or self._index_dir)

        with open(target / "state.pkl", "rb") as f:
            state = pickle.load(f)
        with open(target / "knowledge_graph.pkl", "rb") as f:
            self.kg = pickle.load(f)

        self._documents = state["documents"]
        self.em = EmbeddingManager.from_state(state["embedding"])

        self.ranker.compute_pagerank(self.kg)
        importance_scores = self.ranker.get_importance_scores(self.kg)
        self._expander = QueryExpander(self.kg, self.ranker)

        chunks_dict = state.get("chunks", {})
        doc_to_chunks = state.get("doc_to_chunks", {})
        chunks_list = list(chunks_dict.values())

        self.retriever._attach_state(
            embedding_manager=self.em,
            documents=self._documents,
            chunks=chunks_list,
            doc_to_chunks=doc_to_chunks,
            graph_ranker=self.ranker,
            kg=self.kg,
            importance_scores=state.get("global_scores", importance_scores),
            chunk_mode=True,
        )
        self.retriever.load_artifacts(str(target))

        self._built = True
        print(f"✅ Index loaded từ {target} ({len(self._documents)} bài báo)")

    # ── Search ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list:
        """Tìm kiếm và trả về top_k bài báo liên quan."""
        t0 = time.time()
        processed = self.query_proc.process(query)

        # Graph-based query expansion
        if self._expander:
            expansion = self._expander.expand(processed, hops=2, use_ppr=True)
            multi_queries = expansion.get("multi_queries", [query])
            seed_entities = expansion.get("seed_entities", [])
        else:
            multi_queries = [query]
            seed_entities = []

        # Retrieve
        if len(multi_queries) > 1:
            results = self.retriever.multi_query_retrieve(
                multi_queries, top_k=top_k, seed_entities=seed_entities
            )
        else:
            results = self.retriever.retrieve(
                multi_queries[0], top_k=top_k, seed_entities=seed_entities
            )

        elapsed = time.time() - t0
        return results, elapsed

    # ── Interactive loop ──────────────────────────────────────────────────

    def run_interactive(self, top_k: int = 10):
        print(HELP_TEXT)

        # Gợi ý query ban đầu
        suggestions = get_popular_queries(self.kg, self.ranker, n=6)
        if suggestions:
            print("💡 Gợi ý query phổ biến:")
            for i, s in enumerate(suggestions, 1):
                print(f"   {i}. {s}")
            print()

        while True:
            try:
                user_input = input("🔍 Nhập query (hoặc :help): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nThoát.")
                break

            if not user_input:
                continue

            # Lệnh hệ thống
            if user_input.startswith(":"):
                cmd = user_input.lower()
                if cmd in (":quit", ":exit", ":q"):
                    print("Thoát.")
                    break
                elif cmd == ":help":
                    print(HELP_TEXT)
                elif cmd == ":kg":
                    self._print_kg_stats()
                elif cmd == ":top":
                    self._print_top_entities()
                elif cmd == ":suggest":
                    self._print_suggestions()
                elif cmd == ":viz":
                    self._export_viz()
                else:
                    print(
                        f"Lệnh không hợp lệ: {user_input}. Gõ :help để xem danh sách."
                    )
                continue

            results, elapsed = self.search(user_input, top_k=top_k)
            display_results(results, user_input, elapsed)

    def _print_kg_stats(self):
        n_nodes = len(self.kg._graph.nodes) if hasattr(self.kg, "_graph") else 0
        n_edges = len(self.kg._graph.edges) if hasattr(self.kg, "_graph") else 0
        print(f"\n📊 Knowledge Graph: {n_nodes} entity, {n_edges} quan hệ")

    def _print_top_entities(self, n: int = 20):
        scores = self.ranker.get_importance_scores(self.kg)
        top = sorted(scores.items(), key=lambda x: -x[1])[:n]
        print(f"\n🏆 Top {n} entity quan trọng nhất:")
        for i, (entity, score) in enumerate(top, 1):
            etype = ""
            if hasattr(self.kg, "_graph") and entity in self.kg._graph.nodes:
                etype = f" ({self.kg._graph.nodes[entity].get('type', '')})"
            print(f"   {i:2d}. {entity}{etype}  —  {score:.4f}")

    def _print_suggestions(self, n: int = 10):
        suggestions = get_popular_queries(self.kg, self.ranker, n=n)
        print(f"\n💡 Gợi ý query phổ biến ({n} entity top PageRank):")
        for i, s in enumerate(suggestions, 1):
            print(f"   {i}. {s}")

    def _export_viz(self):
        try:
            viz = KnowledgeGraphVisualizer(self.kg)
            out_path = DATA_DIR / "kg_visualization.html"
            viz.export_html(str(out_path))
            print(f"✅ Đã xuất visualization: {out_path}")
        except Exception as e:
            print(f"❌ Không xuất được: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Vietnamese KG-Enhanced News Search")
    p.add_argument(
        "--query", "-q", type=str, default=None, help="Chạy một query rồi thoát"
    )
    p.add_argument(
        "--data", "-d", type=str, default=None, help="Đường dẫn dataset CSV/JSON"
    )
    p.add_argument("--top-k", "-k", type=int, default=10, help="Số bài báo trả về")
    p.add_argument("--load-index", action="store_true", help="Load index từ disk")
    p.add_argument("--index-dir", type=str, default=None, help="Thư mục index")
    p.add_argument(
        "--ner-model-dir", type=str, default=None, help="Thư mục PhoBERT NER checkpoint"
    )
    p.add_argument(
        "--reranker-dir",
        type=str,
        default=None,
        help="Thư mục cross-encoder checkpoint",
    )
    p.add_argument(
        "--viz", action="store_true", help="Xuất KG visualization sau khi build"
    )
    return p.parse_args()


def main():
    args = parse_args()

    system = NewsSearchSystem(
        data_path=args.data,
        index_dir=args.index_dir,
        ner_model_dir=args.ner_model_dir,
        reranker_model_dir=args.reranker_dir,
    )

    if args.load_index:
        system.load_index(args.index_dir)
    else:
        system.build()
        system.save_index(args.index_dir)

    if args.viz:
        system._export_viz()

    if args.query:
        results, elapsed = system.search(args.query, top_k=args.top_k)
        display_results(results, args.query, elapsed)
    else:
        system.run_interactive(top_k=args.top_k)


if __name__ == "__main__":
    main()
