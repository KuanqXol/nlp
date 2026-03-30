# Vietnamese KG-Enhanced News Search & RAG System

Hệ thống tìm kiếm tin tức tiếng Việt kết hợp:

- Knowledge Graph
- Personalized PageRank
- Chunk-based retrieval
- FAISS / NumPy vector search
- BM25 + Reciprocal Rank Fusion
- Cross-encoder reranking
- RAG summary

Project hiện ưu tiên dữ liệu CSV thật từ VnExpress, nhưng `DataLoader` cũng hỗ trợ JSON array.

## Kiến trúc hiện tại

```text
CSV / JSON news data
        ↓
DataLoader
  - parse_vn_date()
  - strip_author()
  - dedup theo URL
  - lang filter (viet_ratio)
        ↓
VietnameseNER
  - underthesea mặc định
  - HuggingFace nếu --use-model và load được
  - ner_with_checkpoint() + ner_cache.json
        ↓
EntityLinker
  - exact alias
  - Levenshtein fuzzy match
  - embedding match nếu có sentence-transformers
        ↓
RelationExtractor
        ↓
KnowledgeGraph + SimilarityGraphBuilder
        ↓
GraphRanker
  - Global PageRank
  - Query-time Personalized PageRank
        ↓
Chunking + EmbeddingManager
        ↓
Retriever
  - FAISS mặc định nếu có
  - NumPy fallback
  - BM25 + RRF hybrid retrieval
  - Cross-encoder rerank top-20 nếu model load được
        ↓
RAGPipeline
  - Claude nếu có ANTHROPIC_API_KEY
  - Template summary fallback
```

## Cấu trúc project

```text
news_kg_rag/
├── data/
│   ├── vnexpress_articles.csv
│   └── index/
│       ├── state.pkl
│       ├── bm25.pkl
│       └── knowledge_graph.pkl
├── scripts/
│   └── build_index.py
├── src/
│   ├── data_loader.py
│   ├── preprocessing/
│   │   ├── ner.py
│   │   ├── entity_linking.py
│   │   ├── relation_extraction.py
│   │   └── relation_extraction_phobert.py
│   ├── graph/
│   │   ├── knowledge_graph.py
│   │   ├── ranking.py
│   │   ├── similarity.py
│   │   └── visualization.py
│   ├── retrieval/
│   │   ├── chunking.py
│   │   ├── embedding.py
│   │   ├── query_processor.py
│   │   ├── query_expansion.py
│   │   └── retriever.py
│   └── rag/
│       └── pipeline.py
├── main.py
├── test_system.py
└── README.md
```

## Cài đặt

### Core

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### Tùy chọn

```bash
# Nếu muốn Claude summary
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Dữ liệu đầu vào

### CSV

CSV chuẩn hiện tại:

```text
url,date,category,title,text
```

Ví dụ:

```csv
https://vnexpress.net/example.html,"Thứ sáu, 31/7/2020, 18:15 (GMT+7)",thế giới,Tiêu đề bài báo,"Nội dung bài báo..."
```

Khi load CSV, hệ thống sẽ tự:

- parse ngày kiểu VnExpress sang `YYYY-MM-DD`
- cắt dòng tác giả ở cuối bài bằng `strip_author()`
- dedup theo URL
- bỏ bài có tỷ lệ tiếng Việt quá thấp (`viet_ratio < 0.05`)
- suy luận `source=VnExpress` từ URL nếu thiếu

### JSON

JSON cần là array các object:

```json
[
  {
    "id": "doc_1",
    "title": "Tiêu đề bài báo",
    "content": "Nội dung bài báo",
    "date": "2024-01-15",
    "source": "VnExpress",
    "url": "https://...",
    "category": "thế giới"
  }
]
```

## Cách chạy

### 1. Build offline index

Khuyến nghị build một lần rồi load lại:

```bash
python scripts/build_index.py
python scripts/build_index.py --data data/vnexpress_articles.csv
python scripts/build_index.py --data data/custom_news.json --index-dir data/index_custom
```

Sau bước này, state sẽ được lưu trong `data/index/` mặc định.
Các artifact thường gặp:

- `state.pkl`: documents, chunks, embedding state
- `knowledge_graph.pkl`: graph đã build
- `bm25.pkl`: BM25 backend đã serialize
- `vector.index`: FAISS index nếu môi trường có FAISS
- `ner_cache.json`: cache text → entities
- `ner_checkpoint.json`: tiến độ NER để resume nếu build bị ngắt
- `ner_results.jsonl`: kết quả NER incremental

### 2. Search từ index đã build

```bash
python main.py --load-index
python main.py --load-index --query "kinh tế việt nam" --no-llm
python main.py --load-index --index-dir data/index_custom --query "WHO COVID-19"
python main.py --load-index --use-phobert-re --phobert-dir data/phobert_re
```

### 3. Rebuild trực tiếp từ raw data

```bash
python main.py
python main.py --data data/vnexpress_articles.csv --query "chiến tranh nga ukraine"
python main.py --data data/custom_news.json --query "Samsung Hà Nội"
python main.py --data data/vnexpress_articles.csv --use-phobert-re --phobert-dir data/phobert_re
```

## CLI options

| Flag | Default | Ý nghĩa |
| --- | --- | --- |
| `--query`, `-q` | `None` | Chạy một query rồi thoát |
| `--data`, `-d` | `data/vnexpress_articles.csv` | Đường dẫn raw dataset CSV/JSON |
| `--top-k`, `-k` | `7` | Số bài báo trả về |
| `--hops` | `2` | Số hop query expansion |
| `--use-model` | `False` | Thử dùng HuggingFace NER / SBERT |
| `--no-llm` | `False` | Tắt Claude summary |
| `--viz` | `False` | Xuất KG visualization |
| `--load-index` | `False` | Load index từ disk thay vì rebuild |
| `--index-dir` | `data/index` | Thư mục chứa `state.pkl` và `knowledge_graph.pkl` |
| `--use-phobert-re` | `False` | Bật Hybrid Relation Extractor với PhoBERT nếu có model |
| `--phobert-dir` | `data/phobert_re` | Thư mục model PhoBERT RE đã fine-tune |

## Interactive commands

| Lệnh | Chức năng |
| --- | --- |
| `<query>` | Tìm kiếm tin tức |
| `:kg` | Thống kê Knowledge Graph |
| `:top` | Top entity quan trọng nhất |
| `:viz` | Xuất KG ra file HTML/PNG |
| `:help` | Hướng dẫn |
| `:quit` | Thoát |

## Mô tả nhanh các module

### `src/data_loader.py`

- `load_json()` và `load_csv()`
- `parse_vn_date()`
- `strip_author()`
- dedup URL + language filter

### `src/preprocessing/ner.py`

- `underthesea` là backend mặc định hiện tại
- nếu `--use-model` và model load được, có thể dùng HuggingFace
- có `ner_with_checkpoint()` để resume batch lớn
- cache NER được persist ra `ner_cache.json`
- fallback cuối là rule-based

### `src/preprocessing/entity_linking.py`

- alias map
- Levenshtein fuzzy match cho typo ngắn
- embedding match nếu có `sentence-transformers`

### `src/graph/*`

- `knowledge_graph.py`: `MultiDiGraph`, temporal edges, confidence filtering
- `ranking.py`: global PR + PPR
- `similarity.py`: thêm `similar_to` edges bằng embedding
- `visualization.py`: Pyvis / Matplotlib

### `src/retrieval/*`

- `chunking.py`: sentence-window chunking
- `embedding.py`: SBERT hoặc TF-IDF fallback
- `retriever.py`: FAISS mặc định nếu có, BM25 + RRF, cross-encoder rerank top-20
- `query_expansion.py`: relation-aware expansion + multi-query support

### `src/rag/pipeline.py`

- build context từ top documents
- context hiện lấy tối đa `800` ký tự mỗi bài
- Claude nếu có API key, nếu không dùng template summary

## Test

```bash
python test_system.py
```

`test_system.py` hiện dùng fixture CSV nhỏ tự tạo trong thư mục tạm, nên không phụ thuộc vào file dữ liệu lớn trong `data/`.

## Lưu ý thực tế

- FAISS được bật mặc định trong `main.py`, nhưng nếu import FAISS thất bại hệ thống sẽ tự rơi về NumPy backend.
- Nếu có FAISS, index sẽ được lưu thêm ra `vector.index`; nếu không có thì hệ thống vẫn load lại từ embedding state.
- BM25 luôn được serialize ra `bm25.pkl` để `--load-index` không phải rebuild lexical backend.
- Cross-encoder được bật mặc định trong `Retriever`, và sẽ rerank top-20 candidates; nếu `sentence-transformers` hoặc model không load được thì hệ thống tự fallback.
- `--use-phobert-re` sẽ dùng `HybridRelationExtractor`; nếu thiếu model/GPU thì tự fallback về rule-based extractor.
- Build dài có thể resume nhờ `ner_checkpoint.json` và `ner_results.jsonl`.
- `--use-model` là “best effort”: nếu HuggingFace model không tải được, hệ thống vẫn fallback về backend sẵn có.

## Troubleshooting

### Không load được index

```bash
python main.py --load-index
```

Nếu báo thiếu file, hãy build lại:

```bash
python scripts/build_index.py
```

Nếu muốn resume NER batch dài sau khi bị ngắt, giữ nguyên `--index-dir` cũ để hệ thống dùng lại `ner_checkpoint.json` và `ner_cache.json`.

### Không có FAISS

- Hệ thống tự dùng NumPy
- Nếu muốn ép FAISS, cài lại dependency phù hợp môi trường

### Không có `sentence-transformers`

- embedding fallback về TF-IDF
- entity linking embedding match và cross-encoder rerank sẽ tự giảm cấp

### Không có `ANTHROPIC_API_KEY`

- hệ thống vẫn chạy bình thường
- phần summary sẽ dùng template thay vì LLM
