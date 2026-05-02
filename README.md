# Vietnamese KG-Enhanced News Search

H�� thống tìm kiếm tin tức tiếng Việt kết hợp Knowledge Graph, FAISS vector search và cross-encoder reranking. Dữ liệu: 155k bài báo VnExpress (CSV).

## Kiến trúc

```
CSV / JSON
    ↓
DataLoader          parse ngày, dedup URL, lọc ngôn ngữ
    ↓
VietnameseNER       VnCoreNLP word segment → PhoBERT fine-tuned → PER / LOC / ORG
    ↓
EntityLinker        exact match → normalized → embedding cosine → Levenshtein fuzzy
                    (dùng chung vietnamese-bi-encoder với FAISS)
    ↓
KnowledgeGraph      co-occurrence edges (top-5/bài), confidence filter
SimilarityGraphBuilder  thêm edge dựa trên embedding cosine >= 0.75
GraphRanker         Global PageRank (offline) + PPR query-time
    ↓
Chunking            sentence_window, ~800-1500 ký tự, 1 câu overlap
EmbeddingManager    vietnamese-bi-encoder (bkai-foundation-models), 768 dim
FAISS index         FlatIP ≤50k chunks | IVFFlat >50k chunks
    ↓
QueryProcessor      normalize → NER → entity link → intent detection
QueryExpander       PPR 2-hop, relation-weighted, multi-query variants
    ↓
Retriever           FAISS top-50 → graph boost → cross-encoder rerank → date decay
    ↓
Kết quả: title + URL + snippet + score
```

## Cấu trúc thư mục

```
nlp/
├── main.py                         entry point, NewsSearchSystem
├── setup.py                        cài đặt môi trường tự động (chạy 1 lần)
├── requirements.txt
├── data/
│   ├── vnexpress_articles.csv      dữ liệu 155k bài (bạn cung cấp)
│   ├── ner_model/                  PhoBERT NER sau khi fine-tune
│   ├── bi_encoder_model/           vietnamese-bi-encoder (auto download qua setup.py)
│   ├── reranker_model/             ViDeBERTa cross-encoder sau khi fine-tune
│   ├── vncorenlp/                  VnCoreNLP-1.2.jar + models (auto download qua setup.py)
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
    ├── utils/text.py
    ├── preprocessing/
    │   ├── ner.py                  VnCoreNLP + PhoBERT NER + cache + checkpoint
    │   └── entity_linking.py       4-stage linker, shared encoder
    ├── graph/
    │   ├── knowledge_graph.py      MultiDiGraph, co-occurrence, confidence filter
    │   ├── ranking.py              PageRank + PPR
    │   ├── similarity.py           embedding-based similarity edges
    │   └── visualization.py        Pyvis export
    └── retrieval/
        ├── chunking.py             sentence_window overlap
        ├── embedding.py            VietnameseBiEncoder, EmbeddingManager
        ├── query_processor.py      normalize, NER, intent detection
        ├── query_expansion.py      multi-query, PPR-guided, relation-weighted
        └── retriever.py            FAISS + graph boost + rerank + date decay
```

## Cài đặt

### Yêu cầu

- Python >= 3.9
- Java 8+ (cho VnCoreNLP word segmenter) — [Download](https://www.java.com/en/download/)
- GPU không bắt buộc để search, nhưng cần để build index nhanh

### Setup tự động (khuyến nghị)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
python setup.py
```

`setup.py` tự động: cài dependencies đúng version, download VnCoreNLP, download bi-encoder, kiểm tra NER model, patch các file cần thiết.

## Thứ tự chạy lần đầu

### Bước 1 — Fine-tune NER (Kaggle GPU, ~45 phút)

Upload `kaggle_train_ner.ipynb` lên Kaggle, thêm CSV, chạy. Download `ner_model.zip`, giải nén vào `data/ner_model/`.

Nếu muốn bỏ qua: hệ thống dùng `underthesea` làm fallback NER, chất lượng thấp hơn nhưng chạy được ngay.

### Bước 2 — Fine-tune cross-encoder (Kaggle GPU, ~2-3 giờ)

Upload `kaggle_train_reranker.ipynb` lên Kaggle, thêm CSV, chạy. Download `reranker_model.zip`, giải nén vào `data/reranker_model/`.

Nếu muốn bỏ qua: hệ thống dùng `cross-encoder/ms-marco-MiniLM-L6-v2` (tiếng Anh) làm fallback.

### Bước 3 — Build index (chạy 1 lần, ~2-8 giờ tùy GPU)

```bash
# Windows: set encoding trước để tránh lỗi Unicode
set PYTHONIOENCODING=utf-8

python scripts/build_index.py --data data/vnexpress_articles.csv
```

NER chạy qua 155k bài với checkpoint tự động — nếu bị ngắt thì chạy lại lệnh trên, hệ thống resume từ điểm dừng. **Lưu ý: xóa `data/index/ner_cache.json` nếu muốn NER chạy lại từ đầu** (ví dụ sau khi fix VnCoreNLP).

### Bước 4 — Chạy demo

```bash
# Windows
set PYTHONIOENCODING=utf-8

# Interactive
python main.py --load-index

# One-shot query
python main.py --load-index --query "Samsung đầu tư Việt Nam"
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
| 155k bài VnExpress (silver) | Mix vào NER training    | File CSV của bạn                          |
| MMARCO-Vi                   | Fine-tune cross-encoder | `unicamp-dl/mmarco` (config `vietnamese`) |
| UIT-ViQuAD 2.0              | Fine-tune cross-encoder | `taidng/UIT-ViQuAD2.0`                    |
| 155k bài VnExpress (pseudo) | Fine-tune cross-encoder | File CSV của bạn                          |

## Đánh giá

```bash
python scripts/evaluate_system.py --load-index
python scripts/evaluate_system.py --load-index --tasks ner
python scripts/evaluate_system.py --load-index --tasks retrieval --retrieval-qrels data/qrels.json
python scripts/evaluate_system.py --load-index --output data/eval_results.json
```

Metrics: Precision/Recall/F1 cho NER, Recall@K / MRR@K / NDCG@K cho retrieval.

## Xử lý sự cố

**UnicodeEncodeError trên Windows**

```bash
set PYTHONIOENCODING=utf-8
```

**NER crash hoặc bị ngắt giữa chừng**

Chạy lại lệnh build y hệt, hệ thống tự resume từ `data/index/ner_checkpoint.json`.

**VnCoreNLP không load được**

Kiểm tra Java đã cài (`java -version`). Kiểm tra `data/vncorenlp/VnCoreNLP-1.2.jar` tồn tại và > 20MB. Chạy `python setup.py` để tự download lại.

**Không có FAISS**

Cài lại: `pip install faiss-cpu`

**Không có `data/ner_model/`**

H�� thống dùng `underthesea` fallback. Train model: chạy `kaggle_train_ner.ipynb` trên Kaggle GPU.

**Không có `data/reranker_model/`**

H�� thống dùng `cross-encoder/ms-marco-MiniLM-L6-v2` fallback. Train model: chạy `kaggle_train_reranker.ipynb`.

**Build index lần đầu rất chậm**

NER trên CPU mất 5-8 giờ cho 155k bài. Dùng GPU (RTX 3050 Ti ~20 phút). Sau khi build xong dùng `--load-index` thì load < 1 phút.

**Muốn chạy lại NER từ đầu (ví dụ sau khi fix VnCoreNLP)**

```bash
del data\index\ner_cache.json
del data\index\ner_checkpoint.json
del data\index\ner_results.jsonl
python scripts/build_index.py --data data/vnexpress_articles.csv
```
