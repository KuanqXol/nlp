"""
=============================================================================
  Vietnamese KG-Enhanced News Search System
  setup.py -- Cai dat moi truong tu dong (chay 1 lan duy nhat)
=============================================================================

TONG QUAN DU AN
---------------
He thong tim kiem tin tuc tieng Viet su dung Knowledge Graph ket hop voi
vector search (FAISS). Pipeline:

1. NER (Named Entity Recognition)
   - PhoBERT fine-tuned tren VLSP2016 nhan dang entity trong bai bao
   - 3 loai entity: PER (nguoi), LOC (dia diem), ORG (to chuc)
   - Word segmentation bang VnCoreNLP TRUOC khi dua vao PhoBERT
     Vi du: "Ho Chi Minh" -> "Ho_Chi_Minh" -> PhoBERT nhan ra 1 entity
     (neu khong segment: "Ho", "Chi", "Minh" la 3 token rieng -> NER sai)

2. Entity Linking
   - Chuan hoa entity co dang viet khac nhau ve cung 1 canonical ID
   - 4 tang match theo thu tu uu tien:
     a) Exact match:       "Viet Nam" == "Viet Nam"
     b) Normalized match:  bo dau, lowercased
     c) Embedding cosine:  vector similarity >= 0.85 (bi-encoder)
     d) Levenshtein fuzzy: edit distance cho loi chinh ta, viet tat
   - Ket qua: "VN", "Viet Nam", "nuoc Viet" -> cung entity "Viet Nam"
   - Dung shared encoder voi EmbeddingManager -> khong load model 2 lan

3. Knowledge Graph (KG)
   - Do thi co huong: node = entity, edge = quan he
   - Edge "co_occurrence": 2 entity cung xuat hien trong 1 bai bao
     weight = tich link_score cua 2 entity (entity tin cay cao -> weight lon)
     Giu top 5 cap co score cao nhat moi bai (MAX_COOCCUR_PER_DOC=5)

4. Similarity Graph
   - Tinh cosine similarity giua moi cap entity bang bi-encoder
   - Neu cosine >= 0.75 -> them edge "similar_to" vao KG
   - Cho phep PPR di qua entity tuong tu du khong co co-occurrence truc tiep

5. PageRank (offline, tinh 1 lan khi build)
   - Diem quan trong cua tung entity tren toan bo KG
   - Entity nhieu ket noi, nhieu bai bao de cap -> diem cao
   - Cong thuc: PR(v) = (1-d) + d * sum(PR(u)/out(u)) voi d=0.85
   - Dung lam base score: bai co entity PageRank cao duoc uu tien

6. PPR - Personalized PageRank (query time, moi lan search)
   - NER trich "seed entities" tu query nguoi dung
   - PPR khoi dau tu seeds, lan truyen 2 hop tren KG
   - Khac PageRank global: tim entity lien quan den QUERY CU THE
   - Vi du: query "Samsung dau tu" -> seed=[Samsung]
     PPR lan ra: Samsung -> Galaxy, Lee Jae-yong, Viet Nam, ban dan tu...
   - Final score = 0.5*pagerank + 0.3*ppr + 0.15*frequency + 0.05*quality
   - PPR con dung de sinh multi-query variants (query expansion)

7. FAISS Vector Search
   - Moi bai bao chia thanh chunks ~800 ky tu (sentence window)
   - Moi chunk embed thanh vector 768 chieu bang bi-encoder
   - FAISS index:
     * <= 50k chunks: FlatIP (brute-force, chinh xac 100%)
     * >  50k chunks: IVFFlat (approximate, nhanh hon)
   - Query embed -> tim top-50 chunk gan nhat -> merge theo doc

8. Cross-Encoder Reranking
   - Top-50 chunk tu FAISS rerank bang cross-encoder (nhin ca query+chunk)
   - Ket hop date decay (bai moi +8%) va PageRank score
   - Tra ve top-10 bai bao cuoi cung

THOI GIAN CHAY (RTX 3050 Ti 4GB)
----------------------------------
  Build index (1 lan):    ~6-10 gio
    - NER 155k bai:        2-4 gio  (co checkpoint, resume duoc)
    - Entity linking:      20-40 phut
    - Build KG + PageRank: 5-10 phut
    - Chunking (~950k):    ~5 phut
    - Embedding + FAISS:   3-5 gio
  Query (moi lan search): 150-300ms

CAU TRUC THU MUC
-----------------
  data/
    ner_model/          -- PhoBERT NER (giai nen thu cong)
    bi_encoder_model/   -- vietnamese-bi-encoder (auto download)
    vncorenlp/          -- VnCoreNLP-1.2.jar + models (auto download)
    vnexpress_articles.csv
    index/              -- FAISS + KG + state (sau khi build)

CACH CHAY
----------
  python setup.py
  set PYTHONIOENCODING=utf-8
  python scripts/build_index.py --data data/vnexpress_articles.csv
  python main.py --load-index

=============================================================================
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SRC_NER = ROOT / "src" / "preprocessing" / "ner.py"
SRC_EMBED = ROOT / "src" / "retrieval" / "embedding.py"
MAIN_PY = ROOT / "main.py"


def run(cmd, check=True, silent=False):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not silent and result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0 and result.stderr.strip():
        print(result.stderr.strip())
    if check and result.returncode != 0:
        print(f"[FAIL] {cmd}")
        sys.exit(1)
    return result


def ok(msg):
    print(f"  [OK] {msg}")


def warn(msg):
    print(f"  [!!] {msg}")


def info(msg):
    print(f"       {msg}")


def section(n, t, title):
    print(f"\n[{n}/{t}] {title}")


# ---------------------------------------------------------------------------
# 1. Kiem tra Python & Java
# ---------------------------------------------------------------------------
def check_requirements():
    section(1, 9, "Kiem tra moi truong...")
    major, minor = sys.version_info[:2]
    if major < 3 or minor < 9:
        warn(f"Python {major}.{minor} -- can Python >= 3.9")
        sys.exit(1)
    ok(f"Python {major}.{minor}")

    r = run("java -version", check=False, silent=True)
    if r.returncode != 0:
        warn("Java chua duoc cai. VnCoreNLP can Java 8+.")
        info("Download: https://www.java.com/en/download/")
        sys.exit(1)
    ver_line = (r.stderr or r.stdout).split("\n")[0]
    ok(f"Java: {ver_line.strip()}")


# ---------------------------------------------------------------------------
# 2. Cai dependencies
# ---------------------------------------------------------------------------
def install_deps():
    section(2, 9, "Cai dependencies...")
    run(f"{sys.executable} -m pip install -q -r requirements.txt")
    # Pin version: sentence-transformers 3.x+ bug tren Windows + torch 2.5
    run(
        f"{sys.executable} -m pip install -q "
        f'"sentence-transformers==2.7.0" "transformers==4.41.0" py_vncorenlp'
    )
    # faiss-gpu chi co Linux
    r = run(f'{sys.executable} -c "import faiss"', check=False, silent=True)
    if r.returncode != 0:
        warn("faiss-gpu khong kha dung tren Windows, cai faiss-cpu...")
        run(f"{sys.executable} -m pip install -q faiss-cpu")
    print()
    info("Luu y tren Windows -- chay truoc khi build/search:")
    info("  set PYTHONIOENCODING=utf-8")
    ok("Dependencies OK")


# ---------------------------------------------------------------------------
# 3. Download VnCoreNLP
# ---------------------------------------------------------------------------
def download_vncorenlp():
    section(3, 9, "Kiem tra VnCoreNLP word segmenter...")
    vncore_dir = DATA_DIR / "vncorenlp"
    jar = vncore_dir / "VnCoreNLP-1.2.jar"
    vocab = vncore_dir / "models" / "wordsegmenter" / "vi-vocab"
    rdr = vncore_dir / "models" / "wordsegmenter" / "wordsegmenter.rdr"

    if (
        jar.exists()
        and jar.stat().st_size > 1_000_000
        and vocab.exists()
        and rdr.exists()
    ):
        ok(f"VnCoreNLP da co san ({jar.stat().st_size // 1024 // 1024}MB).")
        return

    vncore_dir.mkdir(parents=True, exist_ok=True)
    (vncore_dir / "models" / "wordsegmenter").mkdir(parents=True, exist_ok=True)

    base = "https://github.com/vncorenlp/VnCoreNLP/raw/master"
    files = {
        jar: f"{base}/VnCoreNLP-1.2.jar",
        vocab: f"{base}/models/wordsegmenter/vi-vocab",
        rdr: f"{base}/models/wordsegmenter/wordsegmenter.rdr",
    }
    for dest, url in files.items():
        if dest.exists() and dest.stat().st_size > 1000:
            info(f"{dest.name} da co.")
            continue
        print(f"  Dang tai {dest.name} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            size_mb = dest.stat().st_size / 1024 / 1024
            ok(f"{dest.name} ({size_mb:.1f} MB)")
            if dest.name.endswith(".jar") and size_mb < 1:
                warn(f"File JAR qua nho ({size_mb:.1f}MB) -- co the bi corrupt!")
                info("Tai thu cong tu trinh duyet: " + url)
                info(f"Luu vao: {dest}")
        except Exception as e:
            warn(f"Khong tai duoc {dest.name}: {e}")
            info(f"Tai thu cong: {url}")
    ok("VnCoreNLP OK")


# ---------------------------------------------------------------------------
# 4. Download bi-encoder
# ---------------------------------------------------------------------------
def download_bi_encoder():
    section(4, 9, "Kiem tra bi-encoder model...")
    model_dir = DATA_DIR / "bi_encoder_model"
    if (model_dir / "config.json").exists():
        ok("bi-encoder da co san.")
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    print("  Dang download bkai-foundation-models/vietnamese-bi-encoder (~300MB)...")
    run(
        f'{sys.executable} -c "from huggingface_hub import snapshot_download; '
        f"snapshot_download('bkai-foundation-models/vietnamese-bi-encoder', "
        f"local_dir=r'{model_dir}', local_dir_use_symlinks=False)\""
    )
    ok("bi-encoder OK")


# ---------------------------------------------------------------------------
# 5. Download PhoBERT tokenizer files
# ---------------------------------------------------------------------------
def download_phobert_tokenizer():
    section(5, 9, "Kiem tra PhoBERT tokenizer files...")
    ner_dir = DATA_DIR / "ner_model"
    if not ner_dir.exists() or not (ner_dir / "config.json").exists():
        warn("NER model chua co -- bo qua.")
        return
    needed = ["vocab.txt", "bpe.codes", "tokenizer_config.json"]
    missing = [f for f in needed if not (ner_dir / f).exists()]
    if not missing:
        ok("Tokenizer files day du.")
        return
    print(f"  Thieu: {', '.join(missing)} -- download tu vinai/phobert-base-v2...")
    run(
        f'{sys.executable} -c "from transformers import AutoTokenizer; '
        f"t = AutoTokenizer.from_pretrained('vinai/phobert-base-v2'); "
        f"t.save_pretrained(r'{ner_dir}')\""
    )
    ok("PhoBERT tokenizer OK")


# ---------------------------------------------------------------------------
# 6. Kiem tra NER model weights
# ---------------------------------------------------------------------------
def check_ner_model():
    section(6, 9, "Kiem tra NER model weights...")
    ner_dir = DATA_DIR / "ner_model"
    has_weights = (ner_dir / "model.safetensors").exists() or (
        ner_dir / "pytorch_model.bin"
    ).exists()
    if not ner_dir.exists() or not (ner_dir / "config.json").exists():
        warn("NER model chua co tai data/ner_model/")
        info("Giai nen file model zip vao data/ner_model/")
        info("Can co: config.json, tokenizer_config.json, model.safetensors")
        info("He thong se dung underthesea fallback neu khong co model.")
    elif not has_weights:
        warn("Chua co model weights!")
        info("Giai nen lai file zip model.")
    else:
        ok("NER model weights OK")


# ---------------------------------------------------------------------------
# 7. Kiem tra VnCoreNLP hoat dong
# ---------------------------------------------------------------------------
def verify_vncorenlp():
    section(7, 9, "Kiem tra VnCoreNLP hoat dong...")
    vncore_dir = (DATA_DIR / "vncorenlp").resolve()
    jar = vncore_dir / "VnCoreNLP-1.2.jar"
    if not jar.exists() or jar.stat().st_size < 1_000_000:
        warn(f"VnCoreNLP-1.2.jar chua co hoac bi corrupt.")
        info("Tai thu cong tu trinh duyet:")
        info("https://github.com/vncorenlp/VnCoreNLP/raw/master/VnCoreNLP-1.2.jar")
        info(f"Luu vao: {jar}")
        return
    r = run(
        f'{sys.executable} -c "import py_vncorenlp; '
        f"v = py_vncorenlp.VnCoreNLP(annotators=['wseg'], save_dir=r'{vncore_dir}'); "
        f"print('VnCoreNLP OK')\"",
        check=False,
        silent=False,
    )
    if r.returncode == 0:
        ok("VnCoreNLP hoat dong chinh xac.")
    else:
        warn("VnCoreNLP khong hoat dong -- NER se dung whitespace fallback.")
        info("Kiem tra: java -version (can Java 8+)")
        info(f"Kiem tra jar ton tai: {jar}")


# ---------------------------------------------------------------------------
# 8. Patch ner.py
# ---------------------------------------------------------------------------
def patch_ner():
    section(8, 9, "Kiem tra ner.py...")
    content = SRC_NER.read_text(encoding="utf-8")

    checks = {
        "VnCoreNLP absolute path (resolve)": "save_dir = str(_Path(save_dir).resolve())",
        "VnCoreNLP skip download if jar exists": "Chi download neu jar chua co",
        "word_ids SentencePiece prefix": 'tok.startswith("\\u2581")',
        "re.search offset mapping": "re.search(re.escape",
        "batch tokenize that": "Batch tokenize that",
    }
    all_ok = True
    for desc, marker in checks.items():
        if marker in content:
            info(f"[v] {desc}")
        else:
            warn(f"Thieu patch: {desc}")
            all_ok = False

    if all_ok:
        ok("ner.py tat ca patches da co.")
    else:
        warn(
            "Mot so patch chua duoc apply -- thay the ner.py bang file moi nhat tu repo."
        )


# ---------------------------------------------------------------------------
# 9. Patch main.py + embedding.py
# ---------------------------------------------------------------------------
def patch_configs():
    section(9, 9, "Kiem tra config main.py + embedding.py...")
    changed = False

    # main.py: local bi-encoder
    c = MAIN_PY.read_text(encoding="utf-8")
    old = "self.em = EmbeddingManager()"
    new = 'self.em = EmbeddingManager(model_name=str(DATA_DIR / "bi_encoder_model"))'
    if new not in c:
        if old in c:
            c = c.replace(old, new)
            MAIN_PY.write_text(c, encoding="utf-8")
            ok("main.py: EmbeddingManager -> local bi_encoder_model")
            changed = True
        else:
            warn("main.py: khong tim thay EmbeddingManager() -- kiem tra thu cong.")
    else:
        info("main.py: EmbeddingManager local da co.")

    # main.py: max_chars 800 (giam so chunks: 1.9M -> ~950k)
    import re

    c = MAIN_PY.read_text(encoding="utf-8")
    m = re.search(r"max_chars=(\d+)", c)
    if m:
        current = int(m.group(1))
        if current < 800:
            c = re.sub(r"max_chars=\d+", "max_chars=800", c)
            MAIN_PY.write_text(c, encoding="utf-8")
            ok(f"main.py: max_chars {current}->800 (giam chunks ~2x)")
            changed = True
        else:
            info(f"main.py: max_chars={current} da OK.")

    # embedding.py: batch_size 128 cho 4GB VRAM
    c = SRC_EMBED.read_text(encoding="utf-8")
    if '256 if device.startswith("cuda")' in c:
        c = c.replace(
            'self._default_batch_size = 256 if device.startswith("cuda") else 64',
            'self._default_batch_size = 128 if device.startswith("cuda") else 32',
        )
        SRC_EMBED.write_text(c, encoding="utf-8")
        ok("embedding.py: batch_size 256->128 (phu hop VRAM 4GB)")
        changed = True
    else:
        info("embedding.py: batch_size da OK.")

    if not changed:
        ok("Tat ca config da dung.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  Vietnamese KG-Enhanced News Search System -- Setup")
    print("=" * 65)

    check_requirements()
    install_deps()
    download_vncorenlp()
    download_bi_encoder()
    download_phobert_tokenizer()
    check_ner_model()
    verify_vncorenlp()
    patch_ner()
    patch_configs()

    print("\n" + "=" * 65)
    print("[DONE] Setup hoan tat!")
    print()
    print("Buoc tiep theo:")
    print()
    print("  1. Giai nen NER model vao data/ner_model/ (neu chua co)")
    print("     Can co: config.json, model.safetensors, tokenizer_config.json")
    print()
    print("  2. Set encoding (Windows):")
    print("       set PYTHONIOENCODING=utf-8")
    print()
    print("  3. Build index -- chay 1 lan (~6-10 gio tren GPU 4GB):")
    print("       python scripts/build_index.py --data data/vnexpress_articles.csv")
    print("     NER co checkpoint -- neu bi ngat chay lai se tiep tuc tu do.")
    print()
    print("  4. Tim kiem:")
    print("       python main.py --load-index")
    print('       python main.py --load-index --query "Samsung dau tu Viet Nam"')
    print("=" * 65)


if __name__ == "__main__":
    main()
