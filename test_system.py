"""
test_system.py — Bộ smoke test cho Vietnamese News Search System

Chạy:
    python test_system.py

Phạm vi:
  1. DataLoader (CSV, parse date, strip author, dedup, lang filter)
  2. NER
  3. EntityLinker
  4. RelationExtractor
  5. KnowledgeGraph
  6. EmbeddingManager
  7. GraphRanker
  8. QueryProcessor
  9. QueryExpander
  10. Retriever
  11. RAGPipeline
  12. NewsSearchSystem build/save/load index
"""

import atexit
import csv
import json
import shutil
import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

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


def record_exception(label: str, exc: Exception):
    results.append(check(False, f"{label} exception: {exc}"))
    traceback.print_exc()


TEMP_ROOT = ROOT_DIR / ".tmp_test_system"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
FIXTURE_CSV = TEMP_ROOT / "sample_news.csv"
INDEX_DIR = TEMP_ROOT / "index"
atexit.register(lambda: shutil.rmtree(TEMP_ROOT, ignore_errors=True))


def write_fixture_csv():
    rows = [
        {
            "url": "https://vnexpress.net/putin-zelensky.html",
            "date": "Thứ sáu, 31/7/2020, 18:15 (GMT+7)",
            "category": "thế giới",
            "title": "Putin gặp Zelensky tại Ukraine",
            "text": (
                "Putin gặp Zelensky tại Ukraine. NATO hỗ trợ Ukraine tại Hà Nội.\n"
                "Nguyễn Nam"
            ),
        },
        {
            "url": "https://vnexpress.net/putin-zelensky.html",
            "date": "Thứ sáu, 31/7/2020, 18:30 (GMT+7)",
            "category": "thế giới",
            "title": "Bản trùng bài Putin",
            "text": "Bản trùng cần bị loại.\nNguyễn Nam",
        },
        {
            "url": "https://vnexpress.net/english-story.html",
            "date": "Thứ bảy, 1/8/2020, 09:30 (GMT+7)",
            "category": "thế giới",
            "title": "English article",
            "text": (
                "This article is mostly English and should be filtered out because "
                "it lacks Vietnamese diacritics."
            ),
        },
        {
            "url": "https://vnexpress.net/gdp-vietnam.html",
            "date": "Chủ nhật, 2/8/2020, 08:00 (GMT+7)",
            "category": "kinh tế",
            "title": "Kinh tế Việt Nam tăng trưởng",
            "text": (
                "Kinh tế Việt Nam tăng trưởng 7% GDP trong quý III tại Hà Nội.\n"
                "Trần Minh"
            ),
        },
        {
            "url": "https://vnexpress.net/who-covid.html",
            "date": "Thứ hai, 3/8/2020, 07:45 (GMT+7)",
            "category": "y tế",
            "title": "WHO cảnh báo COVID-19 tại châu Á",
            "text": (
                "WHO cảnh báo COVID-19 tại châu Á. Việt Nam tăng cường giám sát "
                "dịch bệnh.\nLê Anh"
            ),
        },
        {
            "url": "https://vnexpress.net/samsung-rd.html",
            "date": "Thứ ba, 4/8/2020, 10:30 (GMT+7)",
            "category": "công nghệ",
            "title": "Samsung khai trương R&D tại Hà Nội",
            "text": (
                "Samsung khai trương trung tâm R&D tại Hà Nội với 3000 kỹ sư.\n"
                "Mai Lan"
            ),
        },
    ]

    with open(FIXTURE_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["url", "date", "category", "title", "text"],
        )
        writer.writeheader()
        writer.writerows(rows)


write_fixture_csv()

results = []

# ════════════════════════════════════════════════════════════════════════════
section("1. DataLoader")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.data_loader import (
        NewsDataLoader,
        normalize_text,
        parse_vn_date,
        strip_author,
        viet_ratio,
    )

    loader = NewsDataLoader(str(FIXTURE_CSV))
    docs = loader.load()
    stats = loader.summary()

    results.append(check(len(docs) == 4, "CSV load sau dedup/lang filter còn 4 docs"))
    results.append(check("title" in docs[0], "Document có field 'title'"))
    results.append(check("content" in docs[0], "Document có field 'content'"))
    results.append(check("full_text" in docs[0], "Document có field 'full_text'"))
    results.append(check(stats["total"] == len(docs), "Thống kê total khớp"))
    results.append(check(len(stats["by_category"]) >= 3, "Thống kê theo category"))
    results.append(
        check(normalize_text("  Xin  chào  ") == "Xin chào", "normalize_text()")
    )
    results.append(
        check(
            parse_vn_date("Thứ sáu, 31/7/2020, 18:15 (GMT+7)") == "2020-07-31",
            "parse_vn_date()",
        )
    )
    results.append(
        check(
            strip_author("Nội dung chính của bài viết.\nNguyễn Nam")
            == "Nội dung chính của bài viết.",
            "strip_author()",
        )
    )
    results.append(
        check(viet_ratio("This is English only.") < 0.05, "viet_ratio() filterable"))
    results.append(
        check(
            stats["load_stats"]["skipped_dedup"] == 1,
            "Dedup theo URL hoạt động",
        )
    )
    results.append(
        check(
            stats["load_stats"]["skipped_lang"] == 1,
            "Language filter hoạt động",
        )
    )
    results.append(
        check(
            all(d.get("source") == "VnExpress" for d in docs),
            "Tự suy luận source từ URL",
        )
    )
    the_gioi = loader.get_by_category("thế giới")
    results.append(check(len(the_gioi) > 0, "Lọc theo category 'thế giới'"))

except Exception as e:
    record_exception("DataLoader", e)
    docs = []

# ════════════════════════════════════════════════════════════════════════════
section("2. NER")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.preprocessing.ner import (
        VietnameseNER,
        get_entities_by_type,
        ner_with_checkpoint,
        resolve_coreference,
    )

    ner = VietnameseNER(use_model=False)

    text1 = "Putin gặp Zelensky tại Ukraine"
    ents1 = ner.extract(text1)

    results.append(check(len(ents1) > 0, "Extract entity từ văn bản"))
    results.append(check(any(e["text"] == "Putin" for e in ents1), "Phát hiện 'Putin'"))
    results.append(
        check(any(e["text"] == "Ukraine" for e in ents1), "Phát hiện 'Ukraine'")
    )

    ents_name = ner.extract("Nguyễn Văn A làm việc tại Hà Nội.")
    if ner.backend_name == "underthesea":
        results.append(
            check(
                any(e["text"] == "Nguyễn Văn A" for e in ents_name),
                "Merge proper noun liên tiếp thành một entity",
            )
        )

    text2 = "WHO cảnh báo dịch H5N1 tại Hà Nội và TP.HCM"
    ents2 = ner.extract(text2)
    per_or_org = get_entities_by_type(ents2, "PER") + get_entities_by_type(ents2, "ORG")

    results.append(
        check(any(e["text"].lower() == "who" for e in ents2), "Nhận ra entity 'WHO'"))
    results.append(
        check(
            any("hà nội" in e["text"].lower() or "tp.hcm" in e["text"].lower() for e in ents2),
            "Nhận ra ít nhất một địa danh Việt Nam",
        )
    )
    results.append(check(isinstance(per_or_org, list), "get_entities_by_type() hoạt động"))

    docs_ner = ner.batch_extract(docs[:3])
    results.append(
        check(all("entities" in d for d in docs_ner), "Batch NER thêm field 'entities'")
    )

    ner_cache = TEMP_ROOT / "ner_cache.json"
    ner_checkpoint = TEMP_ROOT / "ner_checkpoint.json"
    ner_results = TEMP_ROOT / "ner_results.jsonl"
    docs_checkpoint = ner_with_checkpoint(
        docs[:2],
        ner,
        checkpoint_path=str(ner_checkpoint),
        cache_path=str(ner_cache),
        results_path=str(ner_results),
        log_every=1,
    )
    docs_resume = ner_with_checkpoint(
        docs[:2],
        ner,
        checkpoint_path=str(ner_checkpoint),
        cache_path=str(ner_cache),
        results_path=str(ner_results),
        log_every=1,
    )
    ner_cached = VietnameseNER(use_model=False, cache_path=str(ner_cache))
    results.append(check(ner_cache.exists(), "Lưu được ner_cache.json"))
    results.append(check(ner_checkpoint.exists(), "Lưu được NER checkpoint"))
    results.append(check(ner_results.exists(), "Lưu được NER results JSONL"))
    results.append(check(len(docs_checkpoint) == 2, "ner_with_checkpoint() xử lý đủ 2 docs"))
    results.append(check(len(docs_resume) == 2, "ner_with_checkpoint() resume không lỗi"))
    results.append(
        check(len(getattr(ner_cached, "_extract_cache", {})) > 0, "Load lại NER cache từ disk")
    )

    coref_docs = resolve_coreference(
        [
            {
                "id": "coref-1",
                "full_text": "Phạm Minh Chính phát biểu. Ông nhấn mạnh cải cách.",
                "entities": [
                    {
                        "text": "Phạm Minh Chính",
                        "type": "PER",
                        "start": 0,
                        "end": 15,
                        "sentence_id": 0,
                        "pos": "Np",
                        "score": 0.9,
                        "entity_text": "Phạm Minh Chính",
                        "entity_type": "PER",
                    }
                ],
            }
        ]
    )
    results.append(
        check(
            any(entity.get("coref") for entity in coref_docs[0]["entities"]),
            "resolve_coreference() thêm coref entity",
        )
    )
    results.append(
        check(
            any(entity.get("mention_text") == "Ông" for entity in coref_docs[0]["entities"]),
            "resolve_coreference() giữ mention pronoun",
        )
    )

except Exception as e:
    record_exception("NER", e)
    ner, docs_ner = None, []

# ════════════════════════════════════════════════════════════════════════════
section("3. Entity Linking")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.preprocessing.entity_linking import EntityLinker

    linker = EntityLinker()

    results.append(check(linker.link("Hanoi")[0] == "Hà Nội", "Link 'Hanoi' → 'Hà Nội'"))
    results.append(check(linker.link("Hanoy")[0] == "Hà Nội", "Levenshtein 'Hanoy' → 'Hà Nội'"))
    results.append(
        check(linker.link("SARS-CoV-2")[0] == "COVID-19", "Link 'SARS-CoV-2' → 'COVID-19'")
    )
    results.append(check(linker.link("Hoa Kỳ")[0] == "Mỹ", "Link 'Hoa Kỳ' → 'Mỹ'"))
    results.append(
        check(linker.link("Vladimir Putin")[0] == "Putin", "Link 'Vladimir Putin' → 'Putin'")
    )
    results.append(check(linker.link("WHO")[0] == "WHO", "Link 'WHO' → 'WHO' (giữ nguyên)"))

    sample_ents = [
        {"text": "Hà Nội", "type": "LOC"},
        {"text": "Hanoi", "type": "LOC"},
        {"text": "Hanoy", "type": "LOC"},
    ]
    linked = linker.link_entities(sample_ents)
    canonical_names = [e["canonical"] for e in linked]
    results.append(check("Hà Nội" in canonical_names, "Gộp các biến thể về 'Hà Nội'"))
    results.append(
        check(
            len([e for e in linked if e["canonical"] == "Hà Nội"]) == 1,
            "Không trùng lặp sau khi gộp entity",
        )
    )

    linker.add_alias("VnEx", "VnExpress")
    results.append(check(linker.link("VnEx")[0] == "VnExpress", "Thêm alias runtime"))

except Exception as e:
    record_exception("EntityLinker", e)
    linker = None

# ════════════════════════════════════════════════════════════════════════════
section("4. Relation Extraction")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.preprocessing.relation_extraction import RelationExtractor

    rel_ex = RelationExtractor()

    sample_doc = {
        "full_text": (
            "Putin tuyên bố tiếp tục chiến dịch tại Ukraine. "
            "Zelensky kêu gọi NATO hỗ trợ thêm vũ khí. "
            "WHO cảnh báo dịch COVID-19 bùng phát tại Hà Nội."
        ),
        "linked_entities": [
            {"text": "Putin", "canonical": "Putin", "type": "PER", "link_score": 1.0},
            {"text": "Nga", "canonical": "Nga", "type": "LOC", "link_score": 1.0},
            {"text": "Ukraine", "canonical": "Ukraine", "type": "LOC", "link_score": 1.0},
            {"text": "Zelensky", "canonical": "Zelensky", "type": "PER", "link_score": 1.0},
            {"text": "NATO", "canonical": "NATO", "type": "ORG", "link_score": 1.0},
            {"text": "WHO", "canonical": "WHO", "type": "ORG", "link_score": 1.0},
            {"text": "COVID-19", "canonical": "COVID-19", "type": "MISC", "link_score": 1.0},
            {"text": "Hà Nội", "canonical": "Hà Nội", "type": "LOC", "link_score": 1.0},
        ],
        "date": "2024-01-15",
        "category": "thế giới",
    }

    processed = rel_ex.process_document(sample_doc)
    triples = processed.get("triples", [])

    results.append(check(len(triples) > 0, "Trích xuất được triple"))
    results.append(check("triples" in processed, "Document có field 'triples'"))

    processed_docs = rel_ex.batch_process(docs[:3])
    results.append(
        check(all("triples" in d for d in processed_docs), "Batch process thêm 'triples'"))

except Exception as e:
    record_exception("RelationExtractor", e)

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
                {"canonical": "Putin", "type": "PER", "link_score": 1.0},
                {"canonical": "Nga", "type": "LOC", "link_score": 1.0},
                {"canonical": "Ukraine", "type": "LOC", "link_score": 1.0},
                {"canonical": "Zelensky", "type": "PER", "link_score": 1.0},
                {"canonical": "NATO", "type": "ORG", "link_score": 1.0},
                {"canonical": "WHO", "type": "ORG", "link_score": 1.0},
                {"canonical": "COVID-19", "type": "MISC", "link_score": 1.0},
                {"canonical": "Việt Nam", "type": "LOC", "link_score": 1.0},
            ],
            "triples": [
                {"subject": "Putin", "relation": "leads", "object": "Nga", "confidence": 0.9},
                {"subject": "Nga", "relation": "attacks", "object": "Ukraine", "confidence": 0.92},
                {"subject": "Zelensky", "relation": "leads", "object": "Ukraine", "confidence": 0.9},
                {"subject": "NATO", "relation": "supports", "object": "Ukraine", "confidence": 0.85},
                {"subject": "WHO", "relation": "warns_about", "object": "COVID-19", "confidence": 0.88},
            ],
        },
    ]

    kg.build_from_documents(demo_docs)
    stats = kg.stats()

    results.append(check(stats["nodes"] > 0, "KG có nodes"))
    results.append(check(stats["edges"] > 0, "KG có edges"))

    neighbors = kg.get_neighbors("Ukraine", hops=2)
    results.append(check(len(neighbors["hop1"]) > 0, "Hop-1 neighbors của 'Ukraine'"))
    results.append(
        check(
            "Nga" in neighbors["hop1"] or "Zelensky" in neighbors["hop1"],
            "Neighbors bao gồm entity liên quan",
        )
    )

    info = kg.get_entity_info("WHO")
    results.append(check(info is not None, "get_entity_info trả về kết quả"))
    results.append(check(info["type"] == "ORG", "Type của 'WHO' là 'ORG'"))

    rels = kg.get_relations_between("Nga", "Ukraine")
    results.append(check(len(rels) > 0, "Có quan hệ giữa 'Nga' và 'Ukraine'"))

    tmp = TEMP_ROOT / "kg_test.pkl"
    kg.save(str(tmp))
    kg2 = KnowledgeGraph()
    kg2.load(str(tmp))
    results.append(
        check(
            kg2.graph.number_of_nodes() == kg.graph.number_of_nodes(),
            "Save/Load KG giữ nguyên số nodes",
        )
    )

except Exception as e:
    record_exception("KnowledgeGraph", e)
    from src.graph.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()

# ════════════════════════════════════════════════════════════════════════════
section("6. Embedding Manager")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.retrieval.embedding import EmbeddingManager

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

    qvec = em.encode_query("chiến tranh nga ukraine")
    results.append(check(qvec.shape[0] == em.embedding_dim, "Query vector đúng dimension"))

    sim = em.cosine_similarity(qvec, em.doc_embeddings[0])
    results.append(check(0.0 <= sim <= 1.0, "Cosine similarity trong [0, 1]"))

    sims = [em.cosine_similarity(qvec, em.doc_embeddings[i]) for i in range(4)]
    results.append(check(sims[0] == max(sims), "Doc Nga-Ukraine có similarity cao nhất"))

    ent_embs = em.encode_entities(["Nga", "Ukraine", "WHO"])
    results.append(check(len(ent_embs) == 3, "Encode 3 entity names"))

    em_state = em.export_state()
    em_loaded = EmbeddingManager.from_state(em_state)
    results.append(
        check(
            em_loaded.doc_embeddings.shape == em.doc_embeddings.shape,
            "EmbeddingManager export/load state hoạt động",
        )
    )

except Exception as e:
    record_exception("EmbeddingManager", e)
    em = None

# ════════════════════════════════════════════════════════════════════════════
section("7. Graph Ranker")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.graph.ranking import GraphRanker

    ranker = GraphRanker(damping=0.85)
    pr_scores = ranker.compute_pagerank(kg)

    results.append(check(len(pr_scores) > 0, "PageRank trả về scores"))
    results.append(
        check(all(0 <= v <= 1 for v in pr_scores.values()), "Tất cả score trong [0, 1]"))
    results.append(check(pr_scores.get("Ukraine", 0) > 0, "Ukraine có PageRank > 0"))

    imp_scores = ranker.compute_importance_scores(kg)
    results.append(check(len(imp_scores) == len(pr_scores), "Importance scores hợp lệ"))

    top5 = ranker.get_top_k(imp_scores, k=5)
    results.append(check(len(top5) <= 5, "get_top_k trả về ≤ 5"))
    results.append(check(top5[0][1] >= top5[-1][1], "Top-k được sắp xếp giảm dần"))

except Exception as e:
    record_exception("GraphRanker", e)
    imp_scores = {}
    ranker = None

# ════════════════════════════════════════════════════════════════════════════
section("8. Query Processor")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.retrieval.query_processor import QueryProcessor

    qproc = QueryProcessor(ner, linker)

    pq = qproc.process("chiến tranh nga ukraine 2024")
    results.append(check("entities" in pq, "Process query → có 'entities'"))
    results.append(check("keywords" in pq, "Process query → có 'keywords'"))
    results.append(check(len(pq["keywords"]) > 0, "Keywords không rỗng"))
    results.append(check(pq.get("topic") == "thế giới", "Phát hiện topic 'thế giới'"))
    results.append(check(pq.get("year_filter") == "2024", "Trích xuất năm '2024'"))

    pq_empty = qproc.process("")
    results.append(check(pq_empty["entities"] == [], "Query rỗng → entities = []"))

except Exception as e:
    record_exception("QueryProcessor", e)
    qproc = None

# ════════════════════════════════════════════════════════════════════════════
section("9. Query Expander")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.retrieval.query_expansion import QueryExpander

    expander = QueryExpander(
        kg,
        graph_ranker=ranker,
        importance_scores=imp_scores,
        max_hop1=4,
        max_hop2=3,
    )
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
    results.append(check("multi_queries" in exp, "Expansion có 'multi_queries'"))
    results.append(check("ukraine" in exp["expanded_query"].lower(), "Expanded query chứa entity gốc"))
    results.append(check(len(exp["hop1_entities"]) > 0, "Có ít nhất 1 hop-1 neighbor"))

    explain_text = expander.explain(exp)
    results.append(check("QUERY EXPANSION" in explain_text, "explain() trả về heading đúng"))
    results.append(check("Multi-queries" in explain_text, "explain() hiển thị multi-queries"))

except Exception as e:
    record_exception("QueryExpander", e)

# ════════════════════════════════════════════════════════════════════════════
section("10. Retriever")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.retrieval.retriever import Retriever, _FAISS_AVAILABLE

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
            "linked_entities": [{"canonical": "Ukraine"}, {"canonical": "NATO"}],
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
            "linked_entities": [{"canonical": "WHO"}, {"canonical": "COVID-19"}],
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
            "linked_entities": [{"canonical": "VinAI"}],
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
            "linked_entities": [{"canonical": "Việt Nam"}],
        },
    ]

    em2 = EmbeddingManager(use_sbert=False)
    em2.build_document_index(sample_docs)

    ret = Retriever(use_faiss=False)
    ret.build_simple(sample_docs, em2)

    results_r = ret.search("chiến tranh nga ukraine", top_k=3)
    results.append(check(len(results_r) > 0, "search() trả về kết quả"))
    results.append(check(len(results_r) <= 3, "search() trả về ≤ top_k"))
    results.append(check("retrieval_score" in results_r[0], "Kết quả có 'retrieval_score'"))
    results.append(check("bm25_score" in results_r[0], "Kết quả có 'bm25_score'"))
    results.append(check("rrf_score" in results_r[0], "Kết quả có 'rrf_score'"))
    results.append(check(results_r[0]["id"] == "r1", "Bài Nga-Ukraine được rank cao nhất"))

    results_rr = ret.retrieve("dịch bệnh COVID", top_k=3, rerank=True)
    results.append(check(len(results_rr) > 0, "Reranking không lỗi"))

    doc = ret.get_document("r1")
    results.append(check(doc is not None and doc["id"] == "r1", "get_document() hoạt động"))

    retriever_dir = TEMP_ROOT / "retriever_artifacts"
    ret.save_artifacts(str(retriever_dir))
    results.append(check((retriever_dir / "bm25.pkl").exists(), "Lưu được bm25.pkl"))

    ret_loaded = Retriever(use_faiss=False, use_cross_encoder=False)
    ret_loaded.attach_state(em2, sample_docs, chunk_mode=False)
    ret_loaded.load_artifacts(str(retriever_dir))
    loaded_hits = ret_loaded.search("chiến tranh nga ukraine", top_k=2)
    results.append(check(len(loaded_hits) > 0, "Load BM25 artifact và search được"))

    if _FAISS_AVAILABLE:
        ret_faiss = Retriever(use_faiss=True, use_cross_encoder=False)
        ret_faiss.build_simple(sample_docs, em2)
        faiss_dir = TEMP_ROOT / "retriever_faiss"
        ret_faiss.save_artifacts(str(faiss_dir))
        results.append(check((faiss_dir / "vector.index").exists(), "Lưu được vector.index"))

        ret_faiss_loaded = Retriever(use_faiss=True, use_cross_encoder=False)
        ret_faiss_loaded.attach_state(em2, sample_docs, chunk_mode=False)
        ret_faiss_loaded.load_artifacts(str(faiss_dir))
        faiss_hits = ret_faiss_loaded.search("chiến tranh nga ukraine", top_k=2)
        results.append(check(len(faiss_hits) > 0, "Load FAISS index và search được"))

except Exception as e:
    record_exception("Retriever", e)
    ret = None

# ════════════════════════════════════════════════════════════════════════════
section("11. RAG Pipeline")
# ════════════════════════════════════════════════════════════════════════════
try:
    from src.rag.pipeline import RAGPipeline, TemplateSummarizer, build_context

    if ret is None:
        raise RuntimeError("Retriever không khả dụng")

    rag = RAGPipeline(ret, use_llm=False, top_k=5)

    ctx = build_context(sample_docs[:2], max_chars_per_doc=100)
    results.append(check(len(ctx) > 0, "build_context tạo được string"))
    results.append(check("Bài 1" in ctx, "Context có label 'Bài 1'"))

    ts = TemplateSummarizer()
    summary = ts.summarize("nga ukraine", sample_docs[:3])
    results.append(check("NGA UKRAINE" in summary.upper(), "Summary chứa query"))
    results.append(check(len(summary) > 50, "Summary không rỗng"))

    result = rag.run("chiến tranh nga ukraine", top_k=3)
    results.append(check("query" in result, "RAG result có 'query'"))
    results.append(check("articles" in result, "RAG result có 'articles'"))
    results.append(check("summary" in result, "RAG result có 'summary'"))
    results.append(check("sources" in result, "RAG result có 'sources'"))
    results.append(check(len(result["articles"]) > 0, "RAG trả về bài báo"))

    formatted = rag.format_result(result)
    results.append(check("═" in formatted, "format_result trả về table"))

except Exception as e:
    record_exception("RAGPipeline", e)

# ════════════════════════════════════════════════════════════════════════════
section("12. NewsSearchSystem + Index")
# ════════════════════════════════════════════════════════════════════════════
try:
    from main import NewsSearchSystem, parse_args

    print(f"\n  {INFO} Đang build pipeline trên fixture CSV...")

    system = NewsSearchSystem(
        data_path=str(FIXTURE_CSV),
        use_model=False,
        use_faiss=False,
        use_llm=False,
        use_phobert_re=True,
        phobert_dir=str(TEMP_ROOT / "missing_phobert"),
        index_dir=str(INDEX_DIR),
    )
    system.build()
    system.save_index(str(INDEX_DIR))

    results.append(check((INDEX_DIR / "state.pkl").exists(), "Lưu state.pkl thành công"))
    results.append(
        check((INDEX_DIR / "knowledge_graph.pkl").exists(), "Lưu knowledge_graph.pkl thành công")
    )
    results.append(check((INDEX_DIR / "bm25.pkl").exists(), "Lưu retriever bm25.pkl thành công"))
    results.append(check((INDEX_DIR / "ner_cache.json").exists(), "Build tạo ner_cache.json"))
    results.append(
        check((INDEX_DIR / "ner_checkpoint.json").exists(), "Build tạo ner_checkpoint.json"))
    results.append(
        check((INDEX_DIR / "ner_results.jsonl").exists(), "Build tạo ner_results.jsonl"))

    loaded_system = NewsSearchSystem(
        data_path=str(FIXTURE_CSV),
        use_model=False,
        use_faiss=False,
        use_llm=False,
        use_phobert_re=True,
        phobert_dir=str(TEMP_ROOT / "missing_phobert"),
        index_dir=str(INDEX_DIR),
    )
    loaded_system.load_index(str(INDEX_DIR))
    loaded_result = loaded_system.search("kinh tế việt nam", top_k=3, hops=1)

    results.append(check(len(loaded_result["articles"]) > 0, "Load index và search thành công"))
    results.append(
        check(
            any("Kinh tế Việt Nam" in article.get("title", "") for article in loaded_result["articles"]),
            "Query 'kinh tế việt nam' tìm được bài kinh tế phù hợp",
        )
    )

    cli_args = parse_args(
        [
            "--load-index",
            "--index-dir",
            str(INDEX_DIR),
            "--use-phobert-re",
            "--phobert-dir",
            str(TEMP_ROOT / "missing_phobert"),
        ]
    )
    results.append(check(cli_args.use_phobert_re, "parse_args nhận --use-phobert-re"))
    results.append(
        check(cli_args.phobert_dir == str(TEMP_ROOT / "missing_phobert"), "parse_args nhận --phobert-dir")
    )

except Exception as e:
    record_exception("NewsSearchSystem", e)

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
