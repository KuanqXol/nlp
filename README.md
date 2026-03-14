# 🇻🇳 Vietnamese KG-Enhanced News Search & RAG System

Hệ thống tìm kiếm tin tức tiếng Việt kết hợp **Knowledge Graph**, **Embedding Similarity**, **PageRank** và **RAG** (Retrieval Augmented Generation).

---

## 📐 Kiến trúc hệ thống

```
Dataset tin tức (JSON)
        ↓
┌──────────────────┐
│   Data Loader    │  ← Chuẩn hóa, validate document
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Vietnamese NER  │  ← NlpHUST/ner-vietnamese-electra-base
│  (ner_extraction)│    hoặc rule-based fallback
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Entity Linking  │  ← Alias map + Levenshtein fuzzy match
│ (entity_linking) │    "Hanoi" → "Hà Nội", "Hoa Kỳ" → "Mỹ"
└────────┬─────────┘
         ↓
┌──────────────────────┐
│  Relation Extraction │  ← Rule-based pattern matching
│(relation_extraction) │    (S, R, O) triple
└────────┬─────────────┘
         ↓
┌──────────────────┐
│ Knowledge Graph  │  ← networkx DiGraph
│(knowledge_graph) │    Node=entity, Edge=relation
└────────┬─────────┘
         ↓
┌──────────────────┐        ┌──────────────────┐
│   Graph Ranking  │        │   Embedding Gen  │
│ (graph_ranking)  │        │   (embedding)    │
│   PageRank       │        │ vietnamese-sbert │
└────────┬─────────┘        └────────┬─────────┘
         └──────────┬────────────────┘
                    ↓
         ┌─────────────────────┐
         │    FAISS / NumPy    │
         │      Index          │
         └──────────┬──────────┘
                    ↓
         ┌─────────────────────┐
         │   User Query        │
         │  (query_processor)  │
         │  NER + normalize    │
         └──────────┬──────────┘
                    ↓
         ┌─────────────────────┐
         │   Graph-based       │
         │  Query Expansion    │
         │ (query_expansion)   │
         │ 1-hop + 2-hop KG   │
         └──────────┬──────────┘
                    ↓
         ┌─────────────────────┐
         │  Vector Retrieval   │
         │   (retriever)       │
         │  + KG re-ranking    │
         └──────────┬──────────┘
                    ↓
         ┌─────────────────────┐
         │   RAG Pipeline      │
         │  (rag_pipeline)     │
         │  Claude / Template  │
         └──────────┬──────────┘
                    ↓
         ┌─────────────────────┐
         │    OUTPUT           │
         │  • Top articles     │
         │  • RAG summary      │
         │  • Entity graph     │
         └─────────────────────┘
```

---

## 📁 Cấu trúc project

```
news_kg_rag/
├── data/
│   └── news_dataset.json        ← 20 bài báo tiếng Việt mẫu
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           ← Module 1: Đọc & chuẩn hóa dữ liệu
│   ├── ner_extraction.py        ← Module 2: NER tiếng Việt
│   ├── entity_linking.py        ← Module 3: Chuẩn hóa & gộp entity
│   ├── relation_extraction.py   ← Module 4: Trích xuất (S,R,O) triple
│   ├── knowledge_graph.py       ← Module 5: Xây dựng KG (networkx)
│   ├── embedding.py             ← Module 6: Vector embedding
│   ├── similarity_graph.py      ← Module 7: Edge similarity giữa entity
│   ├── graph_ranking.py         ← Module 8: PageRank + importance score
│   ├── query_processor.py       ← Module 9: Xử lý query người dùng
│   ├── query_expansion.py       ← Module 10: Mở rộng query qua KG
│   ├── retriever.py             ← Module 11: FAISS / NumPy retrieval
│   ├── rag_pipeline.py          ← Module 12: RAG answer generation
│   └── graph_visualization.py   ← Module 13: Visualize KG (pyvis/mpl)
│
├── main.py                      ← Entry point (interactive search)
├── test_system.py               ← 78 unit tests
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt

### 1. Cài đặt Python dependencies

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Cài đặt core (chạy được ngay, không cần GPU)
pip install networkx numpy matplotlib

# Cài đặt đầy đủ (cần tải model ~1GB)
pip install -r requirements.txt
```

### 2. Cài đặt tùy chọn

```bash
# FAISS (tăng tốc retrieval)
pip install faiss-cpu            # CPU
pip install faiss-gpu            # GPU (cần CUDA)

# Interactive visualization
pip install pyvis

# Vietnamese NLP models (cần internet)
pip install underthesea sentence-transformers transformers torch
```

### 3. Cấu hình API Key (tùy chọn, cho RAG bằng Claude)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## ▶️ Cách chạy

### Chế độ Interactive (Khuyến nghị)

```bash
python main.py
```

Sau khi build xong, nhập query:

```
🔎 Nhập query (hoặc :help): chiến tranh nga ukraine
```

### Chế độ Single Query

```bash
python main.py --query "WHO cảnh báo dịch COVID-19"
python main.py --query "Samsung đầu tư Việt Nam" --top-k 5
python main.py --query "bầu cử tổng thống Mỹ" --hops 1
```

### Options

| Flag             | Default                  | Mô tả                      |
| ---------------- | ------------------------ | -------------------------- |
| `--query` / `-q` | None                     | Single query mode          |
| `--data` / `-d`  | `data/news_dataset.json` | Đường dẫn dataset          |
| `--top-k` / `-k` | 7                        | Số bài báo trả về          |
| `--hops`         | 2                        | Số hop query expansion     |
| `--use-model`    | False                    | Dùng HuggingFace NER model |
| `--no-llm`       | False                    | Tắt Claude API             |
| `--viz`          | False                    | Xuất KG visualization      |

### Lệnh trong interactive mode

| Lệnh      | Chức năng                     |
| --------- | ----------------------------- |
| `<query>` | Tìm kiếm tin tức              |
| `:kg`     | Thống kê Knowledge Graph      |
| `:top`    | Top 20 entity quan trọng nhất |
| `:viz`    | Xuất KG ra file HTML          |
| `:help`   | Hướng dẫn                     |
| `:quit`   | Thoát                         |

### Chạy tests

```bash
python test_system.py
```

---

## 💡 Ví dụ Query Demo

```
chiến tranh nga ukraine
WHO cảnh báo dịch COVID-19 tại châu Á
Samsung khai trương Hà Nội công nghệ
bầu cử tổng thống Mỹ 2024
Phạm Minh Chính kinh tế Việt Nam
VinAI trí tuệ nhân tạo tiếng Việt
Ngân hàng Thế giới đầu tư năng lượng
dịch cúm H5N1 Đông Nam Á
```

---

## 🧩 Mô tả từng module

### `data_loader.py`

- Đọc JSON, validate, chuẩn hóa text tiếng Việt
- Lọc theo `category`, `source`, `date`

### `ner_extraction.py`

- **Primary**: `NlpHUST/ner-vietnamese-electra-base` (HuggingFace)
- **Fallback**: Rule-based dictionary (PER/LOC/ORG/MISC)
- Output: `[{'text': 'Putin', 'type': 'PER'}, ...]`

### `entity_linking.py`

- Bảng alias tĩnh (~80 mapping tiếng Việt)
- Fuzzy match Levenshtein (threshold ≤ 2)
- Gộp entity trùng: `"Hà Nội"` + `"Hanoi"` → canonical: `"Hà Nội"`

### `relation_extraction.py`

- Pattern regex 13 loại quan hệ tiếng Việt
- Keyword detection giữa entity pairs
- Output triple: `("WHO", "cảnh báo", "dịch cúm")`

### `knowledge_graph.py`

- NetworkX DiGraph với node/edge attributes
- Co-occurrence edge tự động
- Query: `get_neighbors(entity, hops=2)`
- Persistence: pickle save/load

### `embedding.py`

- **Primary**: `keepitreal/vietnamese-sbert`
- **Fallback**: TF-IDF + IDF weighting + L2 normalize
- Cache entity embeddings

### `graph_ranking.py`

- NetworkX PageRank (damping=0.85)
- Combined score: `0.6 × PageRank + 0.4 × frequency`
- Top-k query by entity type

### `query_processor.py`

- NER + Entity Linking trên query
- Phát hiện topic (6 loại), năm tháng
- Fallback keyword extraction

### `query_expansion.py`

- 1-hop + 2-hop KG neighbor traversal
- Ưu tiên neighbor có importance score cao
- Max expanded entities: 15 (configurable)

### `retriever.py`

- **FAISS** `IndexFlatIP` (cosine) nếu có
- **NumPy** brute-force fallback
- Re-ranking: `0.8 × vector_score + 0.2 × entity_importance`

### `rag_pipeline.py`

- Build context từ top-k documents
- **LLM**: Claude API (nếu có ANTHROPIC_API_KEY)
- **Template**: Fallback không cần LLM
- Output: summary + source list

### `graph_visualization.py`

- **Pyvis**: HTML tương tác, drag/zoom
- **Matplotlib**: PNG tĩnh
- Màu sắc theo entity type, size theo PageRank

---

## 🔧 Mở rộng hệ thống

### Thêm dữ liệu thật

Thêm bài báo vào `data/news_dataset.json` theo format:

```json
{
  "id": "unique_id",
  "title": "Tiêu đề bài báo",
  "content": "Nội dung...",
  "date": "2024-01-15",
  "source": "VnExpress",
  "url": "https://...",
  "category": "thế giới"
}
```

### Dùng NER model thật

```bash
# Cài đặt
pip install transformers torch underthesea

# Chạy với model
python main.py --use-model
```

### Dùng Vietnamese SBERT

```python
em = EmbeddingManager(
    use_sbert=True,
    model_name="keepitreal/vietnamese-sbert"
)
```

### Bật Claude RAG

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python main.py   # Tự động dùng Claude để generate summary
```

---

## 📊 Benchmark (Dataset 20 docs)

| Component           | Thời gian | Ghi chú              |
| ------------------- | --------- | -------------------- |
| DataLoader          | ~0.01s    | 20 docs              |
| NER (rule-based)    | ~0.05s    | 20 docs              |
| Entity Linking      | ~0.02s    | 20 docs              |
| Relation Extraction | ~0.1s     | 136 triples          |
| KG Build            | ~0.05s    | 149 nodes, 293 edges |
| PageRank            | ~0.01s    |                      |
| TF-IDF Embedding    | ~0.02s    | dim=532              |
| Retrieval (NumPy)   | ~0.001s   | per query            |
| **Total build**     | **~0.3s** |                      |

---

## 🛠️ Troubleshooting

**Lỗi `ModuleNotFoundError`:**

```bash
pip install networkx numpy matplotlib
```

**Lỗi `FAISS not found`:**

- Hệ thống tự động dùng NumPy fallback
- Cài: `pip install faiss-cpu`

**Model không tải được:**

- Hệ thống tự động dùng rule-based NER
- Cần internet: `pip install transformers && python main.py --use-model`

**ANTHROPIC_API_KEY không có:**

- Hệ thống tự động dùng template summary
- Vẫn hoạt động đầy đủ (chỉ summary kém phong phú hơn)
