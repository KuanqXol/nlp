"""
Module: rag_pipeline.py
Chức năng: RAG (Retrieval Augmented Generation) pipeline cho tin tức tiếng Việt.

Bước:
  1. Retrieve documents liên quan (từ Retriever)
  2. Xây dựng context từ các bài báo
  3. Gọi LLM để sinh câu trả lời / tóm tắt

LLM backend:
  - Anthropic Claude API (nếu có API key)
  - Template-based summary (fallback không cần LLM)

Output:
  - Danh sách bài báo liên quan (có score, title, source, url)
  - Summary / câu trả lời tổng hợp bằng tiếng Việt
"""

import os
import textwrap
from datetime import datetime
from typing import Dict, List, Optional

# Import late để tránh circular (retrieval → rag → retrieval)
_multi_query_retrieve = None


def _get_multi_query_retrieve():
    global _multi_query_retrieve
    if _multi_query_retrieve is None:
        from src.retrieval.query_expansion import multi_query_retrieve as _mqr
        _multi_query_retrieve = _mqr
    return _multi_query_retrieve


# ── Context Builder ──────────────────────────────────────────────────────────


def build_context(documents: List[Dict], max_chars_per_doc: int = 800) -> str:
    """
    Tạo context string từ danh sách document để đưa vào LLM.

    Args:
        documents: Danh sách document đã retrieve
        max_chars_per_doc: Giới hạn ký tự trích dẫn từ mỗi bài

    Returns:
        Chuỗi context có cấu trúc
    """
    parts = []
    for i, doc in enumerate(documents, 1):
        title = doc.get("title", "Không có tiêu đề")
        content = doc.get("content", doc.get("full_text", ""))[:max_chars_per_doc]
        source = doc.get("source", "")
        date = doc.get("date", "")
        score = doc.get("retrieval_score", 0)

        part = (
            f"[Bài {i}] {title}\n"
            f"Nguồn: {source} | Ngày: {date} | Điểm liên quan: {score:.3f}\n"
            f"Nội dung: {content}...\n"
        )
        parts.append(part)

    return "\n---\n".join(parts)


# ── Template-based Summarizer (không cần LLM) ────────────────────────────────


class TemplateSummarizer:
    """
    Tổng hợp tin tức theo template khi không có LLM.
    Trích xuất điểm chính từ các bài báo retrieved.
    """

    def summarize(
        self, query: str, documents: List[Dict], expansion_result: Dict = None
    ) -> str:
        """Tạo summary dạng template."""
        if not documents:
            return f"Không tìm thấy bài báo liên quan đến: '{query}'"

        entities = []
        if expansion_result:
            entities = expansion_result.get("all_entities", [])

        lines = [
            f"📋 TỔNG HỢP TIN TỨC: {query.upper()}",
            "=" * 60,
            f"Tìm thấy {len(documents)} bài báo liên quan.\n",
        ]

        if entities:
            lines.append(f"🔍 Entity liên quan: {', '.join(entities[:8])}\n")

        lines.append("📰 NỘI DUNG CHÍNH:")
        for i, doc in enumerate(documents[:5], 1):
            title = doc.get("title", "")
            source = doc.get("source", "")
            date = doc.get("date", "")
            score = doc.get("retrieval_score", 0)

            # Trích xuất câu đầu của content
            content = doc.get("content", doc.get("full_text", ""))
            first_sentence = content.split(".")[0] + "." if content else ""

            lines.append(
                f"\n  [{i}] {title}\n"
                f"       ↳ {first_sentence[:150]}\n"
                f"       📌 {source} | {date} | relevance={score:.3f}"
            )

        lines += [
            "\n" + "=" * 60,
            f"⏰ Được tạo lúc: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        ]

        return "\n".join(lines)


# ── LLM-based RAG (dùng Anthropic API) ─────────────────────────────────────


class ClaudeRAG:
    """
    RAG dùng Anthropic Claude API để sinh câu trả lời.
    Cần có biến môi trường ANTHROPIC_API_KEY.
    """

    SYSTEM_PROMPT = """Bạn là trợ lý tổng hợp tin tức tiếng Việt. 
Dựa trên các bài báo được cung cấp, hãy:
1. Tóm tắt thông tin chính liên quan đến câu hỏi
2. Nêu các sự kiện, nhân vật, địa điểm quan trọng
3. Đưa ra bức tranh toàn cảnh về chủ đề
4. Trả lời bằng tiếng Việt, rõ ràng và súc tích
5. Ghi chú nguồn tin quan trọng"""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._client = None
        self._available = False
        self._init_client()

    def _init_client(self):
        """Khởi tạo Anthropic client."""
        try:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
                self._available = True
                print("[ClaudeRAG] Anthropic API sẵn sàng.")
            else:
                print(
                    "[ClaudeRAG] Không có ANTHROPIC_API_KEY → dùng template summarizer."
                )
        except ImportError:
            print("[ClaudeRAG] Thư viện anthropic chưa cài → dùng template summarizer.")

    def generate(self, query: str, context: str, max_tokens: int = 800) -> str:
        """Gọi Claude API để sinh câu trả lời."""
        if not self._available or self._client is None:
            return None

        user_message = (
            f"Câu hỏi: {query}\n\n"
            f"Thông tin từ các bài báo:\n{context}\n\n"
            f"Hãy tổng hợp và trả lời câu hỏi trên dựa vào các bài báo đã cung cấp."
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except Exception as e:
            print(f"[ClaudeRAG] Lỗi API: {e}")
            return None


# ── RAG Pipeline Chính ───────────────────────────────────────────────────────


class RAGPipeline:
    """
    Pipeline RAG hoàn chỉnh cho hệ thống tìm kiếm tin tức tiếng Việt.

    Ví dụ:
        rag = RAGPipeline(retriever, use_llm=True)
        result = rag.run(
            query="chiến tranh nga ukraine",
            expansion_result=expansion_result,
        )
        print(result['summary'])
        for article in result['articles']:
            print(article['title'])
    """

    def __init__(
        self,
        retriever,
        use_llm: bool = True,
        top_k: int = 7,
    ):
        """
        Args:
            retriever: Retriever instance đã được build
            use_llm: Thử dùng LLM (Claude API) để sinh summary
            top_k: Số bài báo retrieve
        """
        self.retriever = retriever
        self.top_k = top_k

        # Khởi tạo LLM nếu cần
        self._llm = ClaudeRAG() if use_llm else None
        self._template_summarizer = TemplateSummarizer()

    def run(
        self,
        query: str,
        expansion_result: Dict = None,
        top_k: int = None,
    ) -> Dict:
        """
        Chạy toàn bộ RAG pipeline.

        Args:
            query: Query gốc hoặc expanded query
            expansion_result: Output từ QueryExpander (optional)
            top_k: Override số bài báo retrieve

        Returns:
            {
                'query':        str,
                'articles':     list of document dicts,
                'context':      str,
                'summary':      str,
                'entities':     list,
                'sources':      list of {'title', 'source', 'url', 'date', 'score'},
            }
        """
        k = top_k or self.top_k

        # ── Bước 1: Retrieve ─────────────────────────────────────────────
        if expansion_result:
            multi_queries = expansion_result.get("multi_queries", [])
            seed_entities = expansion_result.get("seed_entities", [])
            if multi_queries:
                # Multi-query strategy: search voi tung variant, merge + dedup
                # Score cuoi = max score khi 1 doc xuat hien o nhieu queries
                mqr = _get_multi_query_retrieve()
                articles = mqr(
                    multi_queries,
                    self.retriever,
                    top_k=k,
                    seed_entities=seed_entities,
                )
            else:
                # Fallback: expanded_query don neu khong co multi_queries
                articles = self.retriever.retrieve_with_expansion(expansion_result, top_k=k)
        else:
            articles = self.retriever.retrieve(query, top_k=k)

        # ── Bước 2: Build context ─────────────────────────────────────────
        context = build_context(articles, max_chars_per_doc=800)

        # ── Bước 3: Generate summary ──────────────────────────────────────
        summary = None

        # Thử LLM trước
        if self._llm and self._llm._available:
            summary = self._llm.generate(query, context)

        # Fallback về template
        if not summary:
            summary = self._template_summarizer.summarize(
                query, articles, expansion_result
            )

        # ── Bước 4: Format sources ─────────────────────────────────────────
        sources = [
            {
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "url": doc.get("url", ""),
                "date": doc.get("date", ""),
                "score": doc.get("retrieval_score", 0),
                "category": doc.get("category", ""),
            }
            for doc in articles
        ]

        # Entity từ expansion
        entities = []
        if expansion_result:
            entities = expansion_result.get("all_entities", [])

        return {
            "query": query,
            "articles": articles,
            "context": context,
            "summary": summary,
            "entities": entities,
            "sources": sources,
        }

    def format_result(self, result: Dict) -> str:
        """
        Định dạng kết quả RAG để hiển thị terminal.
        """
        lines = [
            "\n" + "═" * 65,
            f"  KẾT QUẢ TÌM KIẾM: {result['query'].upper()}",
            "═" * 65,
        ]

        # Summary
        lines += ["\n📝 TÓM TẮT:\n", result["summary"], ""]

        # Top articles
        lines.append(f"📰 TOP {len(result['sources'])} BÀI BÁO LIÊN QUAN:\n")
        for i, src in enumerate(result["sources"], 1):
            lines.append(
                f"  {i:2d}. [{src['score']:.3f}] {src['title']}\n"
                f"       📌 {src['source']} | {src['date']} | {src['category']}\n"
                f"       🔗 {src['url']}"
            )

        # Entity
        if result["entities"]:
            lines.append(f"\n🏷️  Entity liên quan: {', '.join(result['entities'][:10])}")

        lines.append("\n" + "═" * 65)
        return "\n".join(lines)


# ── Demo standalone ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.retrieval.embedding import EmbeddingManager
    from src.retrieval.retriever import Retriever

    # Tạo doc giả
    sample_docs = [
        {
            "id": "1",
            "title": "Nga Ukraine xung đột leo thang",
            "content": "Putin tuyên bố tiếp tục chiến dịch tại Ukraine.",
            "full_text": "Putin tuyên bố tiếp tục chiến dịch tại Ukraine. Zelensky kêu gọi NATO.",
            "source": "VnExpress",
            "date": "2024-01-15",
            "url": "https://vnexpress.net/001",
            "category": "thế giới",
        },
        {
            "id": "2",
            "title": "WHO cảnh báo dịch cúm mới",
            "content": "WHO phát cảnh báo về dịch H5N1 tại châu Á.",
            "full_text": "WHO phát cảnh báo về dịch H5N1 tại châu Á. Việt Nam tăng cường phòng chống.",
            "source": "VietnamNet",
            "date": "2024-01-16",
            "url": "https://vietnamnet.vn/002",
            "category": "y tế",
        },
    ]

    em = EmbeddingManager(use_sbert=False)
    em.build_document_index(sample_docs)

    ret = Retriever(use_faiss=False)
    ret.build_simple(sample_docs, em)

    rag = RAGPipeline(ret, use_llm=False)
    result = rag.run("chiến tranh nga ukraine")
    print(rag.format_result(result))
