# Vietnamese KG-Enhanced News Search

Hệ thống tìm kiếm tin tức tiếng Việt kết hợp Knowledge Graph, FAISS vector search và cross-encoder reranking. Dữ liệu: 150k bài báo VnExpress (CSV).

## Kiến trúc

```
CSV / JSON
    ↓
DataLoader          parse ngày, dedup URL, lọc ngôn ngữ
    ↓
VietnameseNER       PhoBERT fine-tuned → PER / LOC / ORG
    ↓
EntityLinker        exact match → Levenshtein → embedding
                    (dùng chung vietnamese-bi-encoder với FAISS)
    ↓
KnowledgeGraph      co-occurrence + relation edges, temporal, confidence filter
SimilarityGraphBuilder  thêm edge dựa trên embedding similarity
GraphRanker         Global PageRank (offline) + PPR query-time
    ↓
Chunking            sentence_window, ~400 ký tự, 1 câu overlap
EmbeddingManager    vietnamese-bi-encoder (bkai-foundation-models)
FAISS index         FlatIP ≤50k chunks | IVFFlat >50k chunks
    ↓
QueryProcessor      normalize → NER → entity link → intent detection
QueryExpander       PPR-guided, 2-hop, relation-weighted, multi-query
    ↓
Retriever           FAISS top-50 → graph boost → cross-encoder rerank → date decay
    ↓
Kết quả: title + URL + snippet + score
```

## Cấu trúc thư mục

```
nlp/
├── main.py                         entry point, NewsSearchSystem
├── requirements.txt
├── data/
│   ├── vnexpress_articles.csv      dữ liệu 150k bài (bạn cung cấp)
│   ├── ner_model/                  PhoBERT NER sau khi fine-tune
│   ├── reranker_model/             ViDeBERTa cross-encoder sau khi fine-tune
│   └── index/                      index build xong (tự tạo khi chạy)
│       ├── state.pkl
│       ├── knowledge_graph.pkl
│       ├── vector.index            FAISS
│       ├── ner_checkpoint.json     resume nếu bị ngắt
│       └── ner_cache.json
├── scripts/
│   ├── build_index.py              build index offline
│   ├── train_ner.py                fine-tune PhoBERT NER
│   ├── train_reranker.py           fine-tune ViDeBERTa cross-encoder
│   └── evaluate_system.py          đánh giá toàn pipeline
└── src/
    ├── data_loader.py
    ├── evaluation_nlp.py
    ├── utils/
    │   └── text.py                 split_sentences dùng chung NER + chunking
    ├── preprocessing/
    │   ├── ner.py                  PhoBERT NER + cache + checkpoint
    │   └── entity_linking.py       3-stage linker, shared encoder
    ├── graph/
    │   ├── knowledge_graph.py      MultiDiGraph, temporal, confidence filter
    │   ├── ranking.py              PageRank + PPR
    │   ├── similarity.py           embedding-based edges
    │   └── visualization.py        Pyvis export
    └── retrieval/
        ├── chunking.py             sentence_window
        ├── embedding.py            VietnameseBiEncoder, EmbeddingManager
        ├── query_processor.py      normalize, NER, intent detection
        ├── query_expansion.py      multi-query, PPR-guided, relation-weighted
        └── retriever.py            FAISS + graph boost + rerank + date decay
```

## Cài đặt

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Yêu cầu Python 3.10+. GPU không bắt buộc để search (chỉ cần cho training).

## Thứ tự chạy lần đầu

### Bước 1 — Fine-tune NER (Kaggle GPU, ~45 phút)

Upload `kaggle_train_ner.ipynb` lên Kaggle, thêm CSV, chạy. Download `ner_model.zip`, giải nén vào `data/ner_model/`.

Nếu muốn bỏ qua bước này: hệ thống dùng `underthesea` làm fallback NER, chất lượng thấp hơn nhưng chạy được ngay.

### Bước 2 — Fine-tune cross-encoder (Kaggle GPU, ~2-3 giờ)

Upload `kaggle_train_reranker.ipynb` lên Kaggle, thêm CSV, chạy. Download `reranker_model.zip`, giải nén vào `data/reranker_model/`.

Nếu muốn bỏ qua: hệ thống dùng `cross-encoder/ms-marco-MiniLM-L6-v2` (tiếng Anh) làm fallback.

### Bước 3 — Build index (chạy 1 lần, ~1-8 giờ tùy phần cứng)

```bash
python scripts/build_index.py --data data/vnexpress_articles.csv
```

NER chạy qua 150k bài với checkpoint tự động — nếu bị ngắt thì chạy lại lệnh trên, hệ thống resume từ điểm dừng.

### Bước 4 — Chạy demo

```bash
# Interactive
python main.py --load-index

# One-shot query
python main.py --load-index --query "Samsung đầu tư Việt Nam"

# Chỉ định NER và reranker model
python main.py --load-index \
    --ner-model-dir data/ner_model \
    --reranker-dir data/reranker_model
```

## Định dạng dữ liệu đầu vào

### CSV (VnExpress)

```
url,date,category,title,text
```

```csv
https://vnexpress.net/bai-bao.html,"Thứ hai, 15/1/2024, 08:00 (GMT+7)",kinh-te,Tiêu đề,"Nội dung..."
```

DataLoader tự parse ngày, dedup URL, lọc bài không phải tiếng Việt.

### JSON

```json
[
  {
    "id": "doc_1",
    "title": "Tiêu đề",
    "content": "Nội dung",
    "date": "2024-01-15",
    "url": "https://...",
    "category": "kinh-te"
  }
]
```

## CLI

```bash
python main.py [options]
```

| Flag              | Mặc định                      | Mô tả                  |
| ----------------- | ----------------------------- | ---------------------- |
| `--query`, `-q`   | None                          | Chạy 1 query rồi thoát |
| `--data`, `-d`    | `data/vnexpress_articles.csv` | Đường dẫn CSV/JSON     |
| `--top-k`, `-k`   | 10                            | Số bài trả về          |
| `--load-index`    | False                         | Load index từ disk     |
| `--index-dir`     | `data/index`                  | Thư mục index          |
| `--ner-model-dir` | `data/ner_model`              | Thư mục PhoBERT NER    |
| `--reranker-dir`  | `data/reranker_model`         | Thư mục cross-encoder  |
| `--viz`           | False                         | Xuất KG visualization  |

## Lệnh trong interactive mode

| Lệnh       | Chức năng                |
| ---------- | ------------------------ |
| `<query>`  | Tìm kiếm tin tức         |
| `:kg`      | Thống kê Knowledge Graph |
| `:top`     | Top entity theo PageRank |
| `:suggest` | Gợi ý query từ KG        |
| `:viz`     | Xuất KG ra file HTML     |
| `:help`    | Hiển thị trợ giúp        |
| `:quit`    | Thoát                    |

## Models

| Model                                          | Vai trò                          | Nguồn                                          |
| ---------------------------------------------- | -------------------------------- | ---------------------------------------------- |
| `vinai/phobert-base-v2`                        | NER backbone                     | VinAI, fine-tune trên VLSP2016 + silver data   |
| `bkai-foundation-models/vietnamese-bi-encoder` | Embedding FAISS + entity linking | BKAI, dùng sẵn không cần train                 |
| `Fsoft-AIC/videberta-base`                     | Cross-encoder reranker backbone  | Fsoft, fine-tune trên MMARCO-Vi + ViQuAD + báo |

## Training data

| Dataset                     | Dùng cho                | Link                                      |
| --------------------------- | ----------------------- | ----------------------------------------- |
| VLSP2016 NER                | Fine-tune PhoBERT NER   | `datnth1709/VLSP2016-NER-data`            |
| 150k bài VnExpress (silver) | Mix vào NER training    | File CSV của bạn                          |
| MMARCO-Vi                   | Fine-tune cross-encoder | `unicamp-dl/mmarco` (config `vietnamese`) |
| UIT-ViQuAD 2.0              | Fine-tune cross-encoder | `taidng/UIT-ViQuAD2.0`                    |
| 150k bài VnExpress (pseudo) | Fine-tune cross-encoder | File CSV của bạn                          |

## Đánh giá

```bash
# Đánh giá toàn pipeline
python scripts/evaluate_system.py --load-index

# Chỉ đánh giá NER
python scripts/evaluate_system.py --load-index --tasks ner

# Đánh giá retrieval với qrels file
python scripts/evaluate_system.py --load-index --tasks retrieval \
    --retrieval-qrels data/qrels.json

# Lưu kết quả
python scripts/evaluate_system.py --load-index --output data/eval_results.json
```

Metrics: Precision/Recall/F1 cho NER, Recall@K / MRR@K / NDCG@K cho retrieval.

## Xử lý sự cố

**NER crash hoặc bị ngắt giữa chừng**

Chạy lại lệnh build y hệt, hệ thống tự resume từ `data/index/ner_checkpoint.json`.

**Không có FAISS**

Tự fallback về NumPy. Cài lại: `pip install faiss-cpu`.

**Không có `data/ner_model/`**

Hệ thống dùng `underthesea` fallback. Train model: chạy `kaggle_train_ner.ipynb` trên Kaggle GPU.

**Không có `data/reranker_model/`**

Hệ thống dùng `cross-encoder/ms-marco-MiniLM-L6-v2` fallback. Train model: chạy `kaggle_train_reranker.ipynb`.

**Build index lần đầu rất chậm**

NER trên CPU mất 5-8 giờ cho 150k bài. Dùng GPU hoặc chạy trên Kaggle. Sau khi build xong dùng `--load-index` thì gần như instantaneous.
