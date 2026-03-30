"""
main.py — Entry point của hệ thống Vietnamese KG-Enhanced News Search with RAG

Cách chạy:
    python main.py

Hoặc chạy với query trực tiếp:
    python main.py --query "chiến tranh nga ukraine"

Pipeline tổng thể:
  1. Load dataset tin tức
  2. NER → Entity Linking → Relation Extraction
  3. Xây dựng Knowledge Graph
  4. Tính PageRank / Importance Score
  5. Build Embedding Index
  6. Interactive search loop:
     a. Nhận query từ người dùng
     b. NER + Entity Linking trên query
     c. Graph-based Query Expansion
     d. Vector Retrieval
     e. RAG Summary
     f. Hiển thị kết quả
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

# Thêm src vào path
SRC_DIR = Path(__file__).parent / "src"
DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = DATA_DIR / "index"
sys.path.insert(0, str(SRC_DIR))

# ── Import các module ────────────────────────────────────────────────────────

from src.data_loader import NewsDataLoader
from src.preprocessing import VietnameseNER, EntityLinker, RelationExtractor
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
from src.rag import RAGPipeline


# ── Banner ───────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   🇻🇳  Vietnamese KG-Enhanced News Search & RAG System       ║
║   Knowledge Graph + Embedding + PageRank + RAG              ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Lệnh hỗ trợ:
  <query>        Tìm kiếm tin tức
  :kg            Hiển thị thống kê Knowledge Graph
  :top           Top 20 entity quan trọng nhất
  :viz           Xuất visualization Knowledge Graph
  :help          Hiển thị trợ giúp này
  :quit / :exit  Thoát

Ví dụ query:
  chiến tranh nga ukraine
  WHO cảnh báo dịch COVID-19
  Samsung đầu tư Việt Nam
  bầu cử tổng thống Mỹ 2024
  Phạm Minh Chính kinh tế
"""


# ── System Builder ───────────────────────────────────────────────────────────


class NewsSearchSystem:
    """
    Lớp tổng hợp toàn bộ hệ thống.
    Xử lý build pipeline và interactive search.
    """

    def __init__(
        self,
        data_path: str = None,
        use_model: bool = False,
        use_faiss: bool = True,
        use_llm: bool = True,
    ):
        """
        Args:
            data_path: Đường dẫn file JSON dataset
            use_model: True → dùng HuggingFace NER model (cần internet/GPU)
            use_faiss: True → dùng FAISS index (cần cài faiss-cpu)
            use_llm: True → thử dùng Claude API (cần ANTHROPIC_API_KEY)
        """
        self.data_path = data_path or str(DATA_DIR / "vnexpress_articles.csv")

        print(BANNER)
        print("🔧 Đang khởi tạo hệ thống...\n")

        # ── Khởi tạo các component ──────────────────────────────────────────
        self.ner = VietnameseNER(use_model=use_model)
        self.linker = EntityLinker()
        self.rel_extractor = RelationExtractor()
        self.kg = KnowledgeGraph()
        self.em = EmbeddingManager(use_sbert=use_model)
        self.ranker = GraphRanker()
        self.query_proc = QueryProcessor(self.ner, self.linker)
        self.retriever = Retriever(use_faiss=use_faiss)
        self.rag = RAGPipeline(self.retriever, use_llm=use_llm)
        self.viz = KnowledgeGraphVisualizer()

        # State
        self._documents = []
        self._chunks = []
        self._doc_to_chunks = {}
        self._importance_scores = {}
        self._query_expander = None

    def build(self):
        """
        Chạy toàn bộ pipeline xây dựng hệ thống từ dữ liệu.
        Gọi một lần khi khởi động.
        """
        t0 = time.time()

        # ── 1. Load data ───────────────────────────────────────────────────
        print("📂 Bước 1/6: Đang load dataset...")
        loader = NewsDataLoader(self.data_path)
        docs = loader.load()

        # ── 2. NER ────────────────────────────────────────────────────────
        print("\n🏷️  Bước 2/6: Nhận dạng thực thể (NER)...")
        docs = self.ner.batch_extract(docs)

        # ── 3. Entity Linking ──────────────────────────────────────────────
        print("\n🔗 Bước 3/6: Chuẩn hóa entity (Entity Linking)...")
        docs = self.linker.batch_process(docs)

        # ── 4. Relation Extraction ────────────────────────────────────────
        print("\n🕸️  Bước 4/6: Trích xuất quan hệ (Relation Extraction)...")
        docs = self.rel_extractor.batch_process(docs)

        # ── 5. Knowledge Graph ────────────────────────────────────────────
        print("\n🗺️  Bước 5/6: Xây dựng Knowledge Graph...")
        self.kg.build_from_documents(docs)
        print("   Thêm similarity edges...")
        SimilarityGraphBuilder(threshold=0.80).build(self.kg, self.em)

        # PageRank
        print("   Tính PageRank...")
        self._importance_scores = self.ranker.compute_importance_scores(self.kg)

        # ── 6. Embedding + Retrieval Index ────────────────────────────────
        print("\n🔢 Bước 6/6: Tạo Embedding Index...")
        chunks, doc_to_chunks = chunk_documents(
            docs,
            strategy="sentence_window",
            max_chars=400,
            overlap=1,
            prepend_title=True,
        )
        self.em.build_document_index(
            [{"id": c["chunk_id"], "full_text": c["chunk_text"]} for c in chunks]
        )
        self.retriever.build(
            chunks,
            self.em,
            doc_to_chunks,
            docs,
            graph_ranker=self.ranker,
            kg=self.kg,
            importance_scores=self._importance_scores,
        )
        self._chunks = chunks
        self._doc_to_chunks = doc_to_chunks

        # ── Query Expander ────────────────────────────────────────────────
        self._query_expander = QueryExpander(
            self.kg,
            graph_ranker=self.ranker,
            importance_scores=self._importance_scores,
            max_hop1=5,
            max_hop2=4,
        )

        self._documents = docs

        elapsed = time.time() - t0
        print(f"\n✅ Hệ thống đã sẵn sàng! ({elapsed:.1f}s)")
        self._print_stats()

    def save_index(self, index_dir: str = None):
        if not self._documents or not self._chunks:
            raise RuntimeError("Chưa có state để lưu. Hãy chạy build() trước.")

        target_dir = Path(index_dir or INDEX_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "metadata": {
                "data_path": self.data_path,
                "saved_at": int(time.time()),
            },
            "documents": self._documents,
            "chunks": self._chunks,
            "doc_to_chunks": self._doc_to_chunks,
            "importance_scores": self._importance_scores,
            "embedding_state": self.em.export_state(),
        }

        with open(target_dir / "state.pkl", "wb") as f:
            pickle.dump(state, f)
        self.kg.save(str(target_dir / "knowledge_graph.pkl"))
        print(f"[Index] Đã lưu index vào: {target_dir}")

    def load_index(self, index_dir: str = None):
        target_dir = Path(index_dir or INDEX_DIR)
        state_path = target_dir / "state.pkl"
        kg_path = target_dir / "knowledge_graph.pkl"

        if not state_path.exists():
            raise FileNotFoundError(f"Không tìm thấy state index: {state_path}")
        if not kg_path.exists():
            raise FileNotFoundError(f"Không tìm thấy knowledge graph index: {kg_path}")

        print(f"[Index] Đang load index từ: {target_dir}")
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        self.kg.load(str(kg_path))
        self.data_path = state.get("metadata", {}).get("data_path", self.data_path)
        self._documents = state.get("documents", [])
        self._chunks = state.get("chunks", [])
        self._doc_to_chunks = state.get("doc_to_chunks", {})
        self._importance_scores = state.get("importance_scores", {})
        self.em = EmbeddingManager.from_state(state.get("embedding_state", {}))

        self.ranker._global_pagerank = dict(getattr(self.kg, "_pagerank", {}))
        self.ranker._importance_scores = dict(self._importance_scores)

        self.retriever.build(
            self._chunks,
            self.em,
            self._doc_to_chunks,
            self._documents,
            graph_ranker=self.ranker,
            kg=self.kg,
            importance_scores=self._importance_scores,
        )
        self._query_expander = QueryExpander(
            self.kg,
            graph_ranker=self.ranker,
            importance_scores=self._importance_scores,
            max_hop1=5,
            max_hop2=4,
        )

        print("[Index] Load xong. Hệ thống sẵn sàng.")
        self._print_stats()

    def _print_stats(self):
        """In thống kê nhanh sau khi build."""
        kg_stats = self.kg.stats()
        loader_stats = {
            "total": len(self._documents),
        }
        print("\n📊 THỐNG KÊ HỆ THỐNG:")
        print(f"  Bài báo   : {loader_stats['total']}")
        print(f"  KG Nodes  : {kg_stats['nodes']} entities")
        print(f"  KG Edges  : {kg_stats['edges']} relations")
        print(f"  Entity types: {kg_stats['entity_types']}")

    def search(self, query: str, top_k: int = 7, hops: int = 2) -> dict:
        """
        Tìm kiếm tin tức và trả về kết quả đầy đủ.

        Args:
            query: Câu hỏi / từ khóa tìm kiếm
            top_k: Số bài báo trả về
            hops: Số hop mở rộng query (1 hoặc 2)

        Returns:
            RAG result dict
        """
        print(f"\n🔍 Đang xử lý query: '{query}'")

        # ── Query Processing ──────────────────────────────────────────────
        processed = self.query_proc.process(query)
        print("\n" + self.query_proc.format_for_display(processed))

        # ── Query Expansion ───────────────────────────────────────────────
        if self._query_expander:
            expansion = self._query_expander.expand(processed, hops=hops)
            if expansion["hop1_entities"] or expansion["hop2_entities"]:
                print("\n" + self._query_expander.explain(expansion))
        else:
            expansion = {
                "expanded_query": query,
                "all_entities": [],
                "seed_entities": [],
            }

        # ── RAG ───────────────────────────────────────────────────────────
        result = self.rag.run(
            query=expansion.get("expanded_query", query),
            expansion_result=expansion,
            top_k=top_k,
        )

        return result

    def run_interactive(self):
        """Vòng lặp tương tác với người dùng."""
        print(HELP_TEXT)

        while True:
            try:
                user_input = input("\n🔎 Nhập query (hoặc :help): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Tạm biệt!")
                break

            if not user_input:
                continue

            # ── Commands ─────────────────────────────────────────────────
            if user_input.lower() in (":quit", ":exit", ":q"):
                print("👋 Tạm biệt!")
                break

            elif user_input.lower() == ":help":
                print(HELP_TEXT)

            elif user_input.lower() == ":kg":
                print("\n📊 KNOWLEDGE GRAPH STATS:")
                for k, v in self.kg.stats().items():
                    print(f"  {k}: {v}")

            elif user_input.lower() == ":top":
                top_entities = self.kg.get_top_entities(top_k=20)
                print("\n🏆 TOP 20 ENTITY QUAN TRỌNG NHẤT:")
                for rank, (entity, score) in enumerate(top_entities, 1):
                    etype = self.kg.graph.nodes.get(entity, {}).get("type", "?")
                    print(f"  {rank:2d}. {entity:25s} ({etype:4s}) score={score:.4f}")

            elif user_input.lower() == ":viz":
                print("\n🎨 Đang tạo visualization...")
                output = self.viz.visualize(
                    self.kg,
                    output_path="knowledge_graph",
                    top_k=50,
                    interactive=True,
                )
                if output:
                    print(f"✅ Đã lưu: {output}")
                    print(
                        "   Mở file này trong trình duyệt để xem Knowledge Graph tương tác!"
                    )

            # ── Search ───────────────────────────────────────────────────
            else:
                result = self.search(user_input)
                print(self.rag.format_result(result))


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vietnamese KG-Enhanced News Search & RAG System"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Chạy một query rồi thoát (không interactive)",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="Đường dẫn file JSON/CSV dataset (mặc định: data/vnexpress_articles.csv)",
    )
    parser.add_argument(
        "--load-index",
        action="store_true",
        help="Load state từ disk thay vì rebuild toàn bộ pipeline",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=str(INDEX_DIR),
        help="Thư mục index trên disk (mặc định: data/index)",
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=7, help="Số bài báo trả về (mặc định: 7)"
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Số hop query expansion (1 hoặc 2, mặc định: 2)",
    )
    parser.add_argument(
        "--use-model",
        action="store_true",
        help="Dùng HuggingFace NER model (cần internet + nhiều RAM)",
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="Không dùng LLM (chỉ template summary)"
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Xuất visualization Knowledge Graph sau khi build",
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    system = NewsSearchSystem(
        data_path=args.data,
        use_model=args.use_model,
        use_faiss=True,
        use_llm=not args.no_llm,
    )

    if args.load_index:
        system.load_index(args.index_dir)
    else:
        system.build()

    # Visualization sau khi build (nếu yêu cầu)
    if args.viz:
        print("\n🎨 Tạo Knowledge Graph visualization...")
        out = system.viz.visualize(system.kg, "knowledge_graph", top_k=50)
        if out:
            print(f"✅ Đã lưu: {out}")

    # Chế độ chạy
    if args.query:
        # Single query mode
        result = system.search(args.query, top_k=args.top_k, hops=args.hops)
        print(system.rag.format_result(result))
    else:
        # Interactive mode
        system.run_interactive()


if __name__ == "__main__":
    main()
