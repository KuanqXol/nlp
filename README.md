# Vietnamese KG-Enhanced News Search

Hệ thống tìm kiếm tin tức tiếng Việt trên tập VnExpress, kết hợp Knowledge Graph, FAISS dense retrieval, query expansion, cross-encoder reranking và Reader QA để trả lời ngắn gọn có dẫn nguồn.

Điểm cập nhật quan trọng: reranker dùng **model train sẵn** `BAAI/bge-reranker-v2-m3`; Reader QA dùng **ViT5 local** để tóm tắt top kết quả, không cần API key bên ngoài.

## Tổng Quan Pipeline

```text
CSV / JSON
  -> NewsDataLoader
     parse ngày VnExpress, chuẩn hóa text, cắt byline, dedup URL, lọc tiếng Việt

  -> VietnameseNER
     ưu tiên PhoBERT NER trong data/ner_model/
     dùng underthesea nếu chưa có checkpoint

  -> EntityLinker
     alias/exact -> embedding similarity -> Levenshtein -> tạo entity mới
     dùng chung encoder với EmbeddingManager để cùng vector space

  -> KnowledgeGraph
     node = entity canonical
     edge = triples nếu document có field triples, co_occurrence nếu không có relation rõ
     thêm temporal metadata, majority vote type, confidence filter

  -> SimilarityGraphBuilder + GraphRanker
     thêm edge similar_to bằng embedding similarity
     tính global PageRank / importance score
     query-time Personalized PageRank nếu có seed entity

  -> Chunking + Embedding + FAISS
     sentence window ~400 ký tự, overlap 1 câu, prepend title
     embedding bằng bkai-foundation-models/vietnamese-bi-encoder
     FAISS FlatIP hoặc IVFFlat tùy số chunk

  -> QueryProcessor + QueryExpander
     normalize query, NER/link entity, keyword/topic/year/intent
     mở rộng multi-query qua KG/PPR khi có seed entity

  -> Retriever
     FAISS top chunk -> doc dedupe -> graph boost -> BGE rerank -> date decay

  -> QAReader
     tự nhận diện intent; factoid thì trích đáp án, câu hỏi tổng hợp thì dùng ViT5 tóm tắt

  -> Kết quả
     answer + citations + title, URL, snippet chunk, score
```

## Cấu Trúc Thư Mục

```text
nlp/
├── main.py                         CLI và class NewsSearchSystem
├── web_app.py                      FastAPI web demo
├── newsurl.py                      crawl URL VnExpress
├── news.py                         crawl nội dung bài báo từ URL
├── requirements.txt
├── Ezpl.md                         giải thích pipeline bằng lời
├── rerank.md                       ghi chú riêng về luồng rerank
├── web/
│   ├── templates/index.html
│   └── static/app.css
├── data/                           không commit, dữ liệu/model/index local
│   ├── vnexpress_articles.csv
│   ├── index/
│   │   ├── state.pkl
│   │   ├── knowledge_graph.pkl
│   │   ├── vector.index
│   │   ├── ner_checkpoint.json
│   │   ├── ner_cache.json
│   │   └── ner_results.jsonl
│   ├── ner_model/                  PhoBERT NER đã fine-tune, tùy chọn
│   ├── bi_encoder_model/           cache local cho bi-encoder, tùy chọn
│   ├── reranker_bge_v2_m3/         BGE reranker train sẵn, ưu tiên nếu có
│   ├── benchmarks/
│   └── vncorenlp/
├── scripts/
│   ├── build_index.py
│   ├── train_ner.py
│   ├── evaluate_test_model.py
│   ├── evaluate_test_bm25.py
│   ├── build_retrieval_eval_set.py
│   └── benchmark_retrieval.py
└── src/
    ├── data_loader.py
    ├── evaluation_nlp.py
    ├── utils/text.py
    ├── preprocessing/
    │   ├── ner.py
    │   └── entity_linking.py
    ├── graph/
    │   ├── knowledge_graph.py
    │   ├── ranking.py
    │   ├── similarity.py
    │   └── visualization.py
    ├── reader/
    │   └── qa_reader.py
    └── retrieval/
        ├── chunking.py
        ├── embedding.py
        ├── query_processor.py
        ├── query_expansion.py
        └── retriever.py
```

## Cài Đặt

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING="utf-8"
pip install -r requirements.txt
python setup.py
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Yêu cầu thực tế: Python 3.10+, RAM đủ lớn để load index, GPU nếu muốn build/train nhanh hơn. Search có thể chạy CPU nhưng reranker BGE v2 M3 khá nặng.

## Chạy Lần Đầu

### 1. Chuẩn bị dữ liệu

Đặt CSV tại:

```text
data/vnexpress_articles.csv
```

Schema CSV:

```csv
url,date,category,title,text
https://vnexpress.net/bai-bao.html,"Thứ hai, 15/1/2024, 08:00 (GMT+7)",kinh-te,Tiêu đề,"Nội dung..."
```

Hệ thống cũng hỗ trợ JSON array với các field:

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

### 2. NER model

Nếu có checkpoint PhoBERT NER, đặt vào:

```text
data/ner_model/
```

Nếu chưa có, hệ thống dùng `underthesea`. Cách này chạy được ngay nhưng chất lượng entity thấp hơn.

Train NER tùy chọn:

```powershell
.\.venv\Scripts\python.exe scripts\train_ner.py `
  --data-csv data\vnexpress_articles.csv `
  --output-dir data\ner_model
```

### 3. Reranker

Runtime dùng BGE reranker train sẵn:

1. Local BGE tại `data/reranker_bge_v2_m3/`, nếu có.
2. Hugging Face model id `BAAI/bge-reranker-v2-m3`, nếu chưa có local.

Nếu muốn ép dùng model train sẵn trên Hugging Face:

```powershell
.\.venv\Scripts\python.exe main.py --load-index --reranker-dir BAAI/bge-reranker-v2-m3
```

Nếu muốn dùng thư mục local:

```powershell
.\.venv\Scripts\python.exe main.py --load-index --reranker-dir data\reranker_bge_v2_m3
```

### 4. Build index

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py --data data\vnexpress_articles.csv
```

Build index sẽ tạo `data/index/`. NER có checkpoint nên nếu bị ngắt có thể chạy lại lệnh build, pipeline tiếp tục từ `ner_checkpoint.json` và `ner_results.jsonl`.

### 5. Chạy CLI

Interactive:

```powershell
.\.venv\Scripts\python.exe main.py --load-index
```

One-shot query:

```powershell
.\.venv\Scripts\python.exe main.py --load-index --query "Samsung đầu tư Việt Nam"
```

Sinh câu trả lời QA có nguồn:

```powershell
.\.venv\Scripts\python.exe main.py --load-index --query "Samsung đầu tư Việt Nam" --answer
```

Chỉ định model:

```powershell
.\.venv\Scripts\python.exe main.py --load-index `
  --ner-model-dir data\ner_model `
  --reranker-dir data\reranker_bge_v2_m3
```

## Chạy Web Demo

Web app mặc định load index tại `data/index/`. Reranker trên web chỉ được load khi bật `USE_RERANKER=1`.

```powershell
$env:USE_RERANKER="1"
$env:RERANKER_DIR="D:\1.HOC\nlp\data\reranker_bge_v2_m3"

.\.venv\Scripts\python.exe -m uvicorn web_app:app --reload --port 8005
```

Mở:

```text
http://127.0.0.1:8005
```

Các mode trong web:

| Mode | Vector | Graph boost | Rerank | Date decay |
|---|---:|---:|---:|---:|
| `vector-only` | Có | Không | Không | Không |
| `vector-graph` | Có | Có | Không | Không |
| `vector-rerank` | Có | Không | Có, nếu reranker đã load | Không |
| `full` | Có | Có | Có, nếu reranker đã load | Có |

Env hữu ích:

| Env | Mặc định | Ý nghĩa |
|---|---|---|
| `INDEX_DIR` | `data/index` | Thư mục index |
| `DATA_PATH` | `data/vnexpress_articles.csv` | File dữ liệu cho lite build |
| `USE_RERANKER` | `0` | Bật load cross-encoder cho web |
| `RERANKER_DIR` | None | Path hoặc Hugging Face model id |
| `ALLOW_LITE_BUILD` | `0` | Nếu không có index, build demo nhỏ không KG/NER |
| `DEMO_MAX_DOCS` | `2000` | Số bài cho lite build |
| `RERANKER_BATCH_SIZE` | CUDA: 8, CPU: 16 | Batch size khi rerank |
| `RERANKER_MAX_LENGTH` | `192` | Max length cho cross-encoder |
| `VIT5_MODEL` | `VietAI/vit5-base-vietnews-summarization` | Model encoder-decoder đã fine-tune cho tóm tắt tin tức |
| `VIT5_CONTEXT_CHARS` | `1200` | Số ký tự tối đa lấy từ mỗi context |
| `VIT5_MIN_CONTEXT_CHARS` | `80` | Bỏ context quá ngắn, thiếu bằng chứng |
| `VIT5_MAX_INPUT_CHARS` | `3500` | Số ký tự tối đa đưa vào prompt ViT5 |
| `VIT5_MAX_LENGTH` | `1024` | Số token tối đa khi tokenize input |
| `VIT5_MAX_NEW_TOKENS` | `220` | Giới hạn output reader |
| `VIT5_NUM_BEAMS` | `4` | Beam search khi sinh tóm tắt |
| `VIT5_DEVICE` | `auto` | `auto`, `cpu` hoặc `cuda` |
| `VIT5_MIN_CONTEXT_SCORE` | `0.30` | Ngưỡng điểm context tối thiểu để Reader dùng |
| `VIT5_MIN_QUERY_OVERLAP` | `0.34` | Ngưỡng overlap query tối thiểu để tránh context yếu |

API web:

```text
GET  /health
GET  /api/status
GET  /api/analysis?query=...
POST /api/graph
POST /api/ask
```

## CLI Options

```powershell
python main.py [options]
```

| Flag | Mặc định | Mô tả |
|---|---|---|
| `--query`, `-q` | None | Chạy một query rồi thoát |
| `--data`, `-d` | `data/vnexpress_articles.csv` | Đường dẫn CSV/JSON |
| `--top-k`, `-k` | `10` | Số kết quả trả về |
| `--answer` | False | Sinh câu trả lời QA từ top kết quả |
| `--reader-context-docs` | `5` | Số kết quả dùng làm context reader |
| `--load-index` | False | Load index từ disk thay vì build lại |
| `--index-dir` | `data/index` | Thư mục index |
| `--ner-model-dir` | None | Thư mục PhoBERT NER; None nghĩa là `data/ner_model` |
| `--reranker-dir` | None | Path hoặc Hugging Face model id; None nghĩa là auto BGE |
| `--viz` | False | Xuất KG visualization sau build/load |

Lệnh trong interactive mode:

| Lệnh | Chức năng |
|---|---|
| `<query>` | Tìm kiếm tin tức |
| `:kg` | Thống kê Knowledge Graph |
| `:top` | Top entity theo PageRank |
| `:suggest` | Gợi ý query từ KG |
| `:viz` | Xuất KG HTML |
| `:help` | Hiển thị trợ giúp |
| `:quit`, `:exit`, `:q` | Thoát |

## Models

| Model | Vai trò | Ghi chú |
|---|---|---|
| `vinai/phobert-base-v2` | Backbone NER | Dùng khi đã fine-tune và đặt ở `data/ner_model/` |
| `underthesea` | Fallback NER | Tự dùng khi chưa có PhoBERT NER |
| `bkai-foundation-models/vietnamese-bi-encoder` | Dense embedding | Dùng cho FAISS, query embedding, entity linking, similarity graph |
| `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker mặc định | Model train sẵn, ưu tiên local `data/reranker_bge_v2_m3/` nếu có |
| `VietAI/vit5-base-vietnews-summarization` | Reader/Summarizer | Chạy local qua Hugging Face Transformers, dùng để tóm tắt top context có citation |

## Đánh Giá Và Benchmark

NER trên ground truth nhỏ:

```powershell
.\.venv\Scripts\python.exe src\evaluation_nlp.py --use-model
```

Dense self-retrieval:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_test_model.py `
  --data data\vnexpress_articles.csv `
  --max-docs 1000 `
  --top-k 10
```

BM25 self-retrieval:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_test_bm25.py `
  --data data\vnexpress_articles.csv `
  --max-docs 1000 `
  --top-k 10
```

Tạo bộ eval weak-labeled khó hơn title self-retrieval:

```powershell
.\.venv\Scripts\python.exe scripts\build_retrieval_eval_set.py `
  --source data\vnexpress_articles.csv `
  --corpus-size 2000 `
  --query-count 200
```

Benchmark nhiều phương pháp, có thể dùng query file:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_retrieval.py `
  --data data\benchmarks\retrieval_eval_200_corpus.csv `
  --query-file data\benchmarks\retrieval_eval_200_queries.jsonl `
  --reranker-dir data\reranker_bge_v2_m3 `
  --top-k 10
```

Các report được ghi vào `data/benchmarks/`.

## Crawler Dữ Liệu

Thu URL:

```powershell
.\.venv\Scripts\python.exe newsurl.py
```

Crawl nội dung bài báo từ URL:

```powershell
.\.venv\Scripts\python.exe news.py
```

Hai script này ghi ra `vnexpress_urls.txt` và `vnexpress_articles.csv` theo config hard-code trong file. Khi dùng cho pipeline chính, chuyển CSV vào `data/vnexpress_articles.csv`.

## Chi Tiết Retrieval Runtime

Mỗi query đi qua các bước chính:

1. `QueryProcessor.process()` normalize query, chạy NER, link entity, lấy keyword/topic/year/intent.
2. `QueryExpander.expand()` tạo multi-query từ KG nếu có seed entity.
3. `EmbeddingManager.encode_query()` encode query bằng bi-encoder, có cache SHA1.
4. FAISS lấy tối đa `FAISS_FETCH_K = 50` chunk gần nhất.
5. Retriever gom chunk về document, giữ tối đa `MAX_CHUNKS_PER_DOC = 2` metadata mỗi doc.
6. Graph boost dùng PPR hoặc global importance, hệ số `GRAPH_BOOST_ALPHA = 0.15`.
7. Cross-encoder BGE rerank candidate bằng cặp `(query, chunk_text)`.
8. Date decay chạy sau rerank, trọng số `DATE_DECAY_WEIGHT = 0.08`.
9. Nếu bật `--answer` hoặc gọi `/api/ask`, `QAReader` dùng top contexts để sinh answer + citations.

Các field nên nhìn khi debug kết quả:

| Field | Ý nghĩa |
|---|---|
| `retrieval_score` | Score dùng để sort cuối cùng |
| `vector_score` | Điểm FAISS của chunk tốt nhất |
| `graph_boost` | Boost từ KG/PPR |
| `cross_encoder_score` | Điểm reranker, chỉ có khi rerank chạy thành công |
| `date_decay_weight` | Hệ số thời gian |
| `chunk_text` | Text thực tế đưa vào reranker |
| `chunk_id` | Chunk tốt nhất |
| `matched_chunk_ids` | Các chunk cùng doc được giữ để debug |

Reader QA trả thêm:

| Field | Ý nghĩa |
|---|---|
| `answer` | Câu trả lời ngắn bằng tiếng Việt |
| `citations` | Danh sách nguồn `[S1]`, `[S2]` đã dùng |
| `used_contexts` | Context thực tế đưa vào reader |
| `selected_sentences` | Câu đã chọn sau bước context compressor |
| `confidence` | Mức tự tin thô: `high`, `medium` hoặc `low` |
| `is_answerable` | Context có đủ bằng chứng để trả lời hay không |
| `provider` | Luôn là `vit5-local` |
| `model` | Model ViT5 đang dùng |
| `reader_mode` | `factoid_extractive`, `vit5_summarization`, `unreliable` hoặc `none` |
| `error` | Lý do lỗi hoặc context yếu, nếu có |

## Lưu Ý Kỹ Thuật

- `data/` rất lớn và đang được ignore. Các model, index và CSV là local artifact.
- `setup.py` là helper cũ, có nhiều giả định cấu hình không còn khớp hoàn toàn với pipeline hiện tại. Nên ưu tiên cài bằng `requirements.txt` và chạy trực tiếp các script.
- Knowledge Graph sẽ có relation semantic nếu document có field `triples`; pipeline hiện tại không chạy relation extraction trong build chính, nên với CSV thường KG chủ yếu dựa trên entity co-occurrence và similarity edge.
- `SimilarityGraphBuilder` tính similarity toàn cặp entity, có thể tốn thời gian nếu số entity lớn.
- Web chỉ load reranker khi `USE_RERANKER=1`; nếu không bật env này thì mode `vector-rerank` và `full` sẽ không có rerank.
- Web gọi `/api/graph` để hiển thị kết quả retrieval/rerank trước, sau đó mới gọi `/api/ask` để cập nhật answer panel. Vì vậy Reader chậm hoặc lỗi không chặn danh sách bài báo.
- Reader chỉ dùng các bài/chunk đã được Retriever trả về; nó không tự tìm thêm nguồn mới.
- Reranker chỉ sắp xếp lại candidate đã được FAISS lấy ra. Nếu bài đúng không vào top chunk ban đầu, reranker không tự tìm thêm bài mới.
- README này phản ánh trạng thái runtime hiện tại: BGE reranker train sẵn và ViT5 local Reader/Summarizer.

## Xử Lý Sự Cố

**Không có `data/index/` hoặc thiếu `vector.index`**

Chạy lại:

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py --data data\vnexpress_articles.csv
```

**NER build bị ngắt**

Chạy lại đúng lệnh build. Pipeline resume từ `data/index/ner_checkpoint.json`.

**Không có `data/ner_model/`**

Hệ thống dùng `underthesea`. Nếu muốn chất lượng tốt hơn, train hoặc đặt PhoBERT NER checkpoint vào `data/ner_model/`.

**Reranker BGE quá nặng hoặc load chậm**

Tắt rerank trên web bằng cách không set `USE_RERANKER=1`, hoặc giảm `RERANKER_BATCH_SIZE`. CLI mặc định sẽ cố load BGE reranker; có thể truyền BGE local/Hugging Face id qua `--reranker-dir`.

**Reader ViT5 load chậm hoặc lỗi model**

Reader không cần API key bên ngoài. Lần đầu chạy `--answer`, Hugging Face Transformers có thể cần tải/cache model ViT5. Đảm bảo môi trường có đúng dependency trong `requirements.txt`, đặc biệt là `sentencepiece` và `transformers==4.41.0`:

```powershell
.\.venv\Scripts\python.exe -m pip install sentencepiece==0.2.1 transformers==4.41.0 tokenizers==0.19.1
```

Nếu máy không có GPU hoặc CUDA lỗi, ép chạy CPU:

```powershell
$env:VIT5_MODEL="VietAI/vit5-base-vietnews-summarization"
$env:VIT5_DEVICE="cpu"
```

**FAISS không import được**

```powershell
pip install faiss-cpu
```

**Output tiếng Việt bị lỗi encoding trên Windows**

```powershell
$env:PYTHONIOENCODING="utf-8"
```
