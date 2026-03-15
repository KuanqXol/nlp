"""
test_system.py — Bộ test toàn diện cho Vietnamese News Search System

Chạy:
    python test_system.py

Kiểm tra:
  1. DataLoader
  2. NER (rule-based)
  3. EntityLinker (alias + fuzzy)
  4. RelationExtractor
  5. KnowledgeGraph (build, query, stats)
  6. EmbeddingManager (TF-IDF)
  7. GraphRanker (PageRank)
  8. QueryProcessor
  9. QueryExpander
  10. Retriever
  11. RAGPipeline
  12. End-to-end queries
"""

import sys
import os
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── Màu terminal ────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

PASS = f"{GREEN}✅ PASS{RESET}"
FAIL = f"{RED}❌ FAIL{RESET}"
INFO = f"{CYAN}ℹ️ {RESET}"


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'─'*55}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*55}{RESET}")


def check(condition: bool, msg: str):
    status = PASS if condition else FAIL
    print(f"  {status}  {msg}")
    return condition


results = []

# ════════════════════════════════════════════════════════════════════════════
section("1. DataLoader")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.data_loader import NewsDataLoader, normalize_text

    loader = NewsDataLoader("data/news_dataset.json")
    docs = loader.load()
    stats = loader.summary()

    results.append(check(len(docs) > 0, "Load document thành công"))
    results.append(check("title" in docs[0], "Document có field 'title'"))
    results.append(check("content" in docs[0], "Document có field 'content'"))
    results.append(check("full_text" in docs[0], "Document có field 'full_text'"))
    results.append(check(stats["total"] == len(docs), "Thống kê total khớp"))
    results.append(check(len(stats["by_category"]) > 0, "Thống kê theo category"))
    results.append(
        check(normalize_text("  Xin  chào  ") == "Xin chào", "Normalize text")
    )

    # Lọc theo category
    the_gioi = loader.get_by_category("thế giới")
    results.append(check(len(the_gioi) > 0, "Lọc theo category 'thế giới'"))

except Exception as e:
    print(f"  {FAIL}  DataLoader exception: {e}")
    traceback.print_exc()
    docs = []

# ════════════════════════════════════════════════════════════════════════════
section("2. NER (Rule-based)")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.preprocessing.ner import VietnameseNER, get_entities_by_type

    ner = VietnameseNER(use_model=False)

    text1 = "Putin gặp Zelensky tại Ukraine"
    ents1 = ner.extract(text1)

    results.append(check(len(ents1) > 0, "Extract entity từ văn bản"))
    results.append(check(any(e["text"] == "Putin" for e in ents1), "Phát hiện 'Putin'"))
    results.append(
        check(any(e["text"] == "Ukraine" for e in ents1), "Phát hiện 'Ukraine'")
    )

    text2 = "WHO cảnh báo dịch H5N1 tại Hà Nội và TP.HCM"
    ents2 = ner.extract(text2)
    orgs = get_entities_by_type(ents2, "ORG")
    locs = get_entities_by_type(ents2, "LOC")

    results.append(check("WHO" in orgs, "Phát hiện ORG: 'WHO'"))
    results.append(check(len(locs) >= 1, "Phát hiện LOC entity"))

    # Batch extract
    docs_ner = ner.batch_extract(docs[:5])
    results.append(
        check(all("entities" in d for d in docs_ner), "Batch NER thêm field 'entities'")
    )

except Exception as e:
    print(f"  {FAIL}  NER exception: {e}")
    traceback.print_exc()
    ner, docs_ner = None, []

# ════════════════════════════════════════════════════════════════════════════
section("3. Entity Linking")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.preprocessing.entity_linking import EntityLinker

    linker = EntityLinker()

    results.append(
        check(linker.link("Hanoi")[0] == "Hà Nội", "Link 'Hanoi' → 'Hà Nội'")
    )
    results.append(
        check(
            linker.link("SARS-CoV-2")[0] == "COVID-19", "Link 'SARS-CoV-2' → 'COVID-19'"
        )
    )
    results.append(check(linker.link("Hoa Kỳ")[0] == "Mỹ", "Link 'Hoa Kỳ' → 'Mỹ'"))
    results.append(
        check(
            linker.link("Vladimir Putin")[0] == "Putin",
            "Link 'Vladimir Putin' → 'Putin'",
        )
    )
    results.append(
        check(linker.link("WHO")[0] == "WHO", "Link 'WHO' → 'WHO' (giữ nguyên)")
    )

    # Gộp entity trùng
    sample_ents = [
        {"text": "Hà Nội", "type": "LOC"},
        {"text": "Hanoi", "type": "LOC"},
        {"text": "WHO", "type": "ORG"},
        {"text": "Tổ chức Y tế Thế giới", "type": "ORG"},
    ]
    linked = linker.link_entities(sample_ents)
    canonical_names = [e["canonical"] for e in linked]
    results.append(
        check("Hà Nội" in canonical_names, "Gộp 'Hà Nội' + 'Hanoi' → 'Hà Nội'")
    )
    results.append(
        check(
            len([e for e in linked if e["canonical"] == "Hà Nội"]) == 1,
            "Không trùng lặp sau gộp",
        )
    )

    # Add custom alias
    linker.add_alias("VnEx", "VnExpress")
    results.append(check(linker.link("VnEx")[0] == "VnExpress", "Thêm alias runtime"))

except Exception as e:
    print(f"  {FAIL}  EntityLinker exception: {e}")
    traceback.print_exc()
    linker = None

# ════════════════════════════════════════════════════════════════════════════
section("4. Relation Extraction")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.preprocessing.relation_extraction import RelationExtractor

    rel_ex = RelationExtractor()

    sample_doc = {
        "full_text": (
            "Tổng thống Nga Putin tuyên bố tiếp tục chiến dịch tại Ukraine. "
            "Zelensky kêu gọi NATO hỗ trợ thêm vũ khí. "
            "WHO cảnh báo dịch COVID-19 bùng phát tại Hà Nội."
        ),
        "linked_entities": [
            {"text": "Putin", "canonical": "Putin", "type": "PER"},
            {"text": "Nga", "canonical": "Nga", "type": "LOC"},
            {"text": "Ukraine", "canonical": "Ukraine", "type": "LOC"},
            {"text": "Zelensky", "canonical": "Zelensky", "type": "PER"},
            {"text": "NATO", "canonical": "NATO", "type": "ORG"},
            {"text": "WHO", "canonical": "WHO", "type": "ORG"},
            {"text": "COVID-19", "canonical": "COVID-19", "type": "MISC"},
            {"text": "Hà Nội", "canonical": "Hà Nội", "type": "LOC"},
        ],
    }

    processed = rel_ex.process_document(sample_doc)
    triples = processed.get("triples", [])

    results.append(check(len(triples) > 0, "Trích xuất được triple"))
    results.append(check("triples" in processed, "Document có field 'triples'"))

    # Batch
    all_docs = docs[:5]
    # Thêm linked_entities từ NER
    if docs_ner:
        all_docs = docs_ner[:5]
        for d in all_docs:
            if "linked_entities" not in d:
                d["linked_entities"] = []

    processed_docs = rel_ex.batch_process(all_docs)
    results.append(
        check(
            all("triples" in d for d in processed_docs),
            "Batch process thêm 'triples' vào tất cả doc",
        )
    )

except Exception as e:
    print(f"  {FAIL}  RelationExtractor exception: {e}")
    traceback.print_exc()

# ════════════════════════════════════════════════════════════════════════════
section("5. Knowledge Graph")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.graph.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()

    demo_docs = [
        {
            "id": "t1",
            "linked_entities": [
                {"canonical": "Putin", "type": "PER"},
                {"canonical": "Nga", "type": "LOC"},
                {"canonical": "Ukraine", "type": "LOC"},
                {"canonical": "Zelensky", "type": "PER"},
                {"canonical": "NATO", "type": "ORG"},
                {"canonical": "WHO", "type": "ORG"},
                {"canonical": "COVID-19", "type": "MISC"},
                {"canonical": "Việt Nam", "type": "LOC"},
            ],
            "triples": [
                {"subject": "Putin", "relation": "lãnh đạo", "object": "Nga"},
                {"subject": "Nga", "relation": "tấn công", "object": "Ukraine"},
                {"subject": "Zelensky", "relation": "lãnh đạo", "object": "Ukraine"},
                {"subject": "NATO", "relation": "hỗ trợ", "object": "Ukraine"},
                {"subject": "WHO", "relation": "cảnh báo", "object": "COVID-19"},
            ],
        },
    ]

    kg.build_from_documents(demo_docs)
    stats = kg.stats()

    results.append(check(stats["nodes"] > 0, "KG có nodes"))
    results.append(check(stats["edges"] > 0, "KG có edges"))

    # Neighbor lookup
    neighbors = kg.get_neighbors("Ukraine", hops=2)
    results.append(check(len(neighbors["hop1"]) > 0, "Hop-1 neighbors của 'Ukraine'"))
    results.append(
        check(
            "Nga" in neighbors["hop1"] or "Zelensky" in neighbors["hop1"],
            "Neighbors bao gồm entity liên quan",
        )
    )

    # Entity info
    info = kg.get_entity_info("WHO")
    results.append(check(info is not None, "get_entity_info trả về kết quả"))
    results.append(check(info["type"] == "ORG", "Type của 'WHO' là 'ORG'"))

    # Relations between
    rels = kg.get_relations_between("Nga", "Ukraine")
    results.append(check(len(rels) > 0, "Có quan hệ giữa 'Nga' và 'Ukraine'"))

    # Save / Load
    import tempfile, os

    tmp = tempfile.mktemp(suffix=".pkl")
    kg.save(tmp)
    kg2 = KnowledgeGraph()
    kg2.load(tmp)
    results.append(
        check(
            kg2.graph.number_of_nodes() == kg.graph.number_of_nodes(),
            "Save/Load KG giữ nguyên số nodes",
        )
    )
    os.unlink(tmp)

except Exception as e:
    print(f"  {FAIL}  KnowledgeGraph exception: {e}")
    traceback.print_exc()
    kg = KnowledgeGraph()

# ════════════════════════════════════════════════════════════════════════════
section("6. Embedding Manager (TF-IDF)")
# ════════════════════════════════════════════════════════════════════════════
try:
    import numpy as np
    from src.retrieval.embedding import EmbeddingManager, TFIDFEmbedder

    em = EmbeddingManager(use_sbert=False)

    sample_docs = [
        {"id": "e1", "full_text": "Putin tuyên bố tiếp tục chiến dịch tại Ukraine"},
        {"id": "e2", "full_text": "WHO cảnh báo dịch COVID-19 tại châu Á"},
        {"id": "e3", "full_text": "VinAI ra mắt mô hình AI tiếng Việt"},
        {"id": "e4", "full_text": "Kinh tế Việt Nam tăng trưởng 7% GDP"},
    ]
    em.build_document_index(sample_docs)

    results.append(check(em.doc_embeddings is not None, "doc_embeddings không None"))
    results.append(check(em.doc_embeddings.shape[0] == 4, "Số doc embedding khớp"))
    results.append(check(em.embedding_dim > 0, "embedding_dim > 0"))

    # Query encoding
    qvec = em.encode_query("chiến tranh nga ukraine")
    results.append(
        check(qvec.shape[0] == em.embedding_dim, "Query vector có đúng dimension")
    )

    # Cosine similarity
    sim = em.cosine_similarity(qvec, em.doc_embeddings[0])
    results.append(check(0.0 <= sim <= 1.0, "Cosine similarity trong [0, 1]"))

    # Bài báo đầu về Nga Ukraine phải có similarity cao nhất
    sims = [em.cosine_similarity(qvec, em.doc_embeddings[i]) for i in range(4)]
    results.append(
        check(sims[0] == max(sims), "Bài Nga-Ukraine có similarity cao nhất")
    )

    # Entity encoding
    ent_embs = em.encode_entities(["Nga", "Ukraine", "WHO"])
    results.append(check(len(ent_embs) == 3, "Encode 3 entity names"))

except Exception as e:
    print(f"  {FAIL}  EmbeddingManager exception: {e}")
    traceback.print_exc()
    em = None

# ════════════════════════════════════════════════════════════════════════════
section("7. Graph Ranker (PageRank)")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.graph.ranking import GraphRanker

    ranker = GraphRanker(damping=0.85)
    pr_scores = ranker.compute_pagerank(kg)

    results.append(check(len(pr_scores) > 0, "PageRank trả về scores"))
    results.append(
        check(
            all(0 <= v <= 1 for v in pr_scores.values()),
            "Tất cả PageRank score trong [0, 1]",
        )
    )

    # Ukraine có nhiều in-edge → phải có score cao
    ukraine_score = pr_scores.get("Ukraine", 0)
    nato_score = pr_scores.get("NATO", 0)
    results.append(check(ukraine_score > 0, "Ukraine có PageRank > 0"))

    # Importance scores
    imp_scores = ranker.compute_importance_scores(kg)
    results.append(
        check(len(imp_scores) == len(pr_scores), "Importance scores = số entity")
    )

    # Top-k
    top5 = ranker.get_top_k(imp_scores, k=5)
    results.append(check(len(top5) <= 5, "get_top_k trả về ≤ 5"))
    results.append(check(top5[0][1] >= top5[-1][1], "Top-k được sắp xếp giảm dần"))

except Exception as e:
    print(f"  {FAIL}  GraphRanker exception: {e}")
    traceback.print_exc()
    imp_scores = {}

# ════════════════════════════════════════════════════════════════════════════
section("8. Query Processor")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.retrieval.query_processor import QueryProcessor

    qproc = QueryProcessor(ner, linker)

    # Test queries
    test_cases = [
        ("chiến tranh nga ukraine", "thế giới"),
        ("WHO cảnh báo dịch COVID-19", "y tế"),
        ("kinh tế Việt Nam tăng trưởng", "kinh tế"),
        ("bầu cử tổng thống Mỹ 2024", "thế giới"),
        ("Samsung công nghệ AI", "công nghệ"),
    ]

    for query, expected_topic in test_cases:
        pq = qproc.process(query)
        results.append(
            check("entities" in pq, f"Process '{query[:20]}...' → có 'entities'")
        )
        results.append(
            check("keywords" in pq, f"Process '{query[:20]}...' → có 'keywords'")
        )
        results.append(check(len(pq["keywords"]) > 0, f"Keywords không rỗng"))
        break  # Chỉ test 1 case để ngắn gọn

    # Test với query rỗng
    pq_empty = qproc.process("")
    results.append(check(pq_empty["entities"] == [], "Query rỗng → entities = []"))

    # Test entity extraction
    pq_nga = qproc.process("chiến tranh nga ukraine 2024")
    entity_names = qproc.get_query_entity_names(pq_nga)
    results.append(check("2024" == pq_nga.get("year_filter"), "Trích xuất năm '2024'"))

except Exception as e:
    print(f"  {FAIL}  QueryProcessor exception: {e}")
    traceback.print_exc()
    qproc = None

# ════════════════════════════════════════════════════════════════════════════
section("9. Query Expander")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.retrieval.query_expansion import QueryExpander

    expander = QueryExpander(kg, imp_scores, max_hop1=4, max_hop2=3)
    pq_test = {
        "entities": [{"canonical": "Ukraine", "type": "LOC"}],
        "keywords": ["chiến tranh"],
        "normalized": "chiến tranh ukraine",
    }

    exp = expander.expand(pq_test, hops=2)

    results.append(check("seed_entities" in exp, "Expansion có 'seed_entities'"))
    results.append(check("hop1_entities" in exp, "Expansion có 'hop1_entities'"))
    results.append(check("hop2_entities" in exp, "Expansion có 'hop2_entities'"))
    results.append(check("expanded_query" in exp, "Expansion có 'expanded_query'"))
    results.append(
        check(
            "ukraine" in exp["expanded_query"].lower(), "Expanded query chứa entity gốc"
        )
    )

    # Hop-1 của Ukraine nên bao gồm Nga, Zelensky, NATO
    hop1 = exp["hop1_entities"]
    results.append(check(len(hop1) > 0, "Có ít nhất 1 hop-1 neighbor"))

    # Explain không raise exception
    explain_text = expander.explain(exp)
    results.append(
        check("QUY TRÌNH MỞ RỘNG QUERY" in explain_text, "explain() trả về text đúng")
    )

except Exception as e:
    print(f"  {FAIL}  QueryExpander exception: {e}")
    traceback.print_exc()

# ════════════════════════════════════════════════════════════════════════════
section("10. Retriever")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.retrieval.retriever import Retriever

    if em is None:
        raise RuntimeError("EmbeddingManager không khả dụng")

    sample_docs = [
        {
            "id": "r1",
            "title": "Nga Ukraine xung đột",
            "full_text": "Putin tuyên bố tiếp tục chiến dịch tại Ukraine. Zelensky kêu gọi NATO.",
            "content": "Putin tuyên bố chiến dịch tại Ukraine.",
            "source": "VnExpress",
            "date": "2024-01-15",
            "url": "https://vne.vn/1",
            "category": "thế giới",
        },
        {
            "id": "r2",
            "title": "WHO cảnh báo COVID",
            "full_text": "WHO phát cảnh báo COVID-19 tại châu Á.",
            "content": "WHO cảnh báo dịch bệnh.",
            "source": "VietnamNet",
            "date": "2024-01-16",
            "url": "https://vnn.vn/2",
            "category": "y tế",
        },
        {
            "id": "r3",
            "title": "VinAI ra mắt AI tiếng Việt",
            "full_text": "VinAI công bố mô hình ngôn ngữ lớn cho tiếng Việt.",
            "content": "VinAI ra mắt AI mới.",
            "source": "Thanh Niên",
            "date": "2024-01-17",
            "url": "https://tn.vn/3",
            "category": "công nghệ",
        },
        {
            "id": "r4",
            "title": "Kinh tế Việt Nam tăng trưởng",
            "full_text": "GDP Việt Nam tăng 7% trong quý I năm 2024.",
            "content": "Tăng trưởng kinh tế quý I.",
            "source": "VnExpress",
            "date": "2024-01-18",
            "url": "https://vne.vn/4",
            "category": "kinh tế",
        },
        {
            "id": "r5",
            "title": "Samsung khai trương R&D Hà Nội",
            "full_text": "Samsung Electronics khai trương trung tâm R&D tại Hà Nội.",
            "content": "Samsung mở R&D tại Hà Nội.",
            "source": "Tuổi Trẻ",
            "date": "2024-01-19",
            "url": "https://tt.vn/5",
            "category": "công nghệ",
        },
    ]

    em2 = EmbeddingManager(use_sbert=False)
    em2.build_document_index(sample_docs)

    ret = Retriever(use_faiss=False)
    ret.build_simple(sample_docs, em2)

    # Retrieve
    results_r = ret.retrieve("chiến tranh nga ukraine", top_k=3)
    results.append(check(len(results_r) > 0, "Retrieve trả về kết quả"))
    results.append(check(len(results_r) <= 3, "Retrieve trả về ≤ top_k"))
    results.append(
        check("retrieval_score" in results_r[0], "Kết quả có 'retrieval_score'")
    )
    results.append(
        check(results_r[0]["id"] == "r1", "Bài Nga-Ukraine được rank cao nhất")
    )

    # Retrieve với rerank
    results_rr = ret.retrieve("dịch bệnh COVID", top_k=3, rerank=True)
    results.append(check(len(results_rr) > 0, "Reranking không lỗi"))

    # get_document
    doc = ret.get_document("r1")
    results.append(
        check(doc is not None and doc["id"] == "r1", "get_document() hoạt động")
    )

except Exception as e:
    print(f"  {FAIL}  Retriever exception: {e}")
    traceback.print_exc()
    ret = em2 = None

# ════════════════════════════════════════════════════════════════════════════
section("11. RAG Pipeline")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.rag.pipeline import RAGPipeline, build_context, TemplateSummarizer

    if ret is None:
        raise RuntimeError("Retriever không khả dụng")

    rag = RAGPipeline(ret, use_llm=False, top_k=5)

    # Test build_context
    ctx = build_context(sample_docs[:2], max_chars_per_doc=100)
    results.append(check(len(ctx) > 0, "build_context tạo được string"))
    results.append(check("Bài 1" in ctx, "Context có label 'Bài 1'"))

    # Template summarizer
    ts = TemplateSummarizer()
    summary = ts.summarize("nga ukraine", sample_docs[:3])
    results.append(check("NGA UKRAINE" in summary.upper(), "Summary chứa query"))
    results.append(check(len(summary) > 50, "Summary không rỗng"))

    # Full RAG run
    result = rag.run("chiến tranh nga ukraine", top_k=3)
    results.append(check("query" in result, "RAG result có 'query'"))
    results.append(check("articles" in result, "RAG result có 'articles'"))
    results.append(check("summary" in result, "RAG result có 'summary'"))
    results.append(check("sources" in result, "RAG result có 'sources'"))
    results.append(check(len(result["articles"]) > 0, "RAG trả về bài báo"))

    # Format không raise
    formatted = rag.format_result(result)
    results.append(check("═" in formatted, "format_result trả về table"))

except Exception as e:
    print(f"  {FAIL}  RAGPipeline exception: {e}")
    traceback.print_exc()

# ════════════════════════════════════════════════════════════════════════════
section("12. End-to-End Full Pipeline")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.data_loader import NewsDataLoader
    from src.preprocessing.ner import VietnameseNER
    from src.preprocessing.entity_linking import EntityLinker
    from src.preprocessing.relation_extraction import RelationExtractor
    from src.graph.knowledge_graph import KnowledgeGraph
    from src.retrieval.embedding import EmbeddingManager
    from src.graph.ranking import GraphRanker
    from src.retrieval.query_processor import QueryProcessor
    from src.retrieval.query_expansion import QueryExpander
    from src.retrieval.retriever import Retriever
    from src.rag.pipeline import RAGPipeline

    print(f"\n  {INFO} Đang chạy full pipeline trên dataset thật...")

    loader = NewsDataLoader("data/news_dataset.json")
    all_docs = loader.load()

    ner_e2e = VietnameseNER(use_model=False)
    linker_e2e = EntityLinker()
    rel_e2e = RelationExtractor()
    kg_e2e = KnowledgeGraph()
    em_e2e = EmbeddingManager(use_sbert=False)
    ranker_e2e = GraphRanker()

    all_docs = ner_e2e.batch_extract(all_docs)
    all_docs = linker_e2e.batch_process(all_docs)
    all_docs = rel_e2e.batch_process(all_docs)
    kg_e2e.build_from_documents(all_docs)
    scores_e2e = ranker_e2e.compute_importance_scores(kg_e2e)
    em_e2e.build_document_index(all_docs)

    ret_e2e = Retriever(use_faiss=False)
    ret_e2e.build(all_docs, em_e2e, scores_e2e)

    rag_e2e = RAGPipeline(ret_e2e, use_llm=False, top_k=5)
    qproc_e2e = QueryProcessor(ner_e2e, linker_e2e)
    expander_e2e = QueryExpander(kg_e2e, scores_e2e)

    test_e2e_queries = [
        "chiến tranh nga ukraine",
        "WHO dịch bệnh COVID",
        "Samsung công nghệ Việt Nam",
        "bầu cử Mỹ 2024",
    ]

    for q in test_e2e_queries:
        pq = qproc_e2e.process(q)
        exp = expander_e2e.expand(pq, hops=2)
        res = rag_e2e.run(exp["expanded_query"], expansion_result=exp)
        results.append(
            check(
                len(res["articles"]) > 0,
                f"E2E query '{q[:30]}' → {len(res['articles'])} bài báo",
            )
        )

except Exception as e:
    print(f"  {FAIL}  End-to-End exception: {e}")
    traceback.print_exc()

# ════════════════════════════════════════════════════════════════════════════
# TỔNG KẾT
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═'*55}{RESET}")
print(f"{BOLD}  KẾT QUẢ KIỂM TRA{RESET}")
print(f"{BOLD}{'═'*55}{RESET}")

total = len(results)
passed = sum(results)
failed = total - passed
pct = passed / total * 100 if total else 0

print(f"\n  Tổng số test: {total}")
print(f"  {GREEN}PASS: {passed}{RESET}")
print(f"  {RED}FAIL: {failed}{RESET}")
print(f"  Tỉ lệ:  {pct:.1f}%")

if failed == 0:
    print(f"\n  {GREEN}{BOLD}🎉 Tất cả test đều PASS!{RESET}")
else:
    print(f"\n  {YELLOW}⚠️  Có {failed} test FAIL. Xem chi tiết ở trên.{RESET}")

print(f"\n{BOLD}{'═'*55}{RESET}\n")
