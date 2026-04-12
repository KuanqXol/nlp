"""
Đọc URL từ file txt (output của 1_url_collector.py),
crawl nội dung từng bài và lưu vào CSV.
"""

import requests
from bs4 import BeautifulSoup
import csv
import os
import time
import logging
import re
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
CONFIG = {
    "input_urls": "vnexpress_urls.txt",  # File URL đầu vào (từ 1_url_collector.py)
    "output_csv": "vnexpress_articles.csv",  # File CSV đầu ra
    "max_workers": 16,  # Số luồng song song
    "request_delay": 0.2,  # Delay giữa các request (giây)
    "request_timeout": 15,
    "max_retries": 3,
}

FIELDS = ["url", "date", "category", "title", "text"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

BASE_URL = "https://vnexpress.net"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("article_crawler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────
_session = requests.Session()
_session.headers.update(HEADERS)

# Tăng connection pool size bằng số luồng để tránh "Connection pool is full"
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=CONFIG["max_workers"],
    pool_maxsize=CONFIG["max_workers"] + 4,
    max_retries=0,
)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def fetch(url: str) -> Optional[str]:
    for attempt in range(CONFIG["max_retries"]):
        try:
            resp = _session.get(url, timeout=CONFIG["request_timeout"])
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text
        except Exception:
            if attempt < CONFIG["max_retries"] - 1:
                time.sleep(1.5 * (attempt + 1))
    return None


# ─────────────────────────────────────────────
# ARTICLE PARSER
# ─────────────────────────────────────────────


def parse_article(url: str) -> Optional[dict]:
    html = fetch(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Category — breadcrumb hoặc URL path
    category = ""
    breadcrumb = soup.find("ul", class_="breadcrumb") or soup.find(
        "div", class_="breadcrumb"
    )
    if breadcrumb:
        items = breadcrumb.find_all("li")
        if items:
            category = items[-1].get_text(strip=True)
    if not category:
        path = url.replace(BASE_URL, "").strip("/").split("/")
        if path:
            category = path[0].replace("-", " ").title()

    # Title
    title_tag = (
        soup.find("h1", class_="title-detail")
        or soup.find("h1", class_=re.compile(r"title"))
        or soup.find("h1")
    )
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Date
    date = ""
    date_tag = soup.find("span", class_="date")
    if date_tag:
        date = date_tag.get_text(strip=True)
    else:
        meta = soup.find("meta", {"property": "article:published_time"})
        if meta:
            date = meta.get("content", "")

    # Body text
    body = soup.find("article", class_="fck_detail") or soup.find(
        "div", class_="fck_detail"
    )
    if body:
        for tag in body.find_all(["figure", "script", "style", "aside", "table"]):
            tag.decompose()
        text = "\n".join(
            p.get_text(strip=True) for p in body.find_all("p") if p.get_text(strip=True)
        )
    else:
        text = ""

    if not title or not text:
        return None

    return {
        "url": url,
        "date": date,
        "category": category,
        "title": title,
        "text": text,
    }


# ─────────────────────────────────────────────
# THREAD-SAFE CSV WRITER
# ─────────────────────────────────────────────
_write_lock = threading.Lock()
_csv_writer = None
_csv_handle = None
_crawled_urls: set[str] = set()


def init_csv(csv_path: str):
    global _csv_writer, _csv_handle, _crawled_urls

    # Resume: đọc URL đã crawl
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("url"):
                    _crawled_urls.add(row["url"])
        logger.info(f"🔄 Resume: {len(_crawled_urls):,} bài đã crawl trong {csv_path}")

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    _csv_handle = open(csv_path, "a", encoding="utf-8-sig", newline="")
    _csv_writer = csv.DictWriter(_csv_handle, fieldnames=FIELDS)
    if not file_exists:
        _csv_writer.writeheader()
        _csv_handle.flush()


def write_article(article: dict):
    with _write_lock:
        _crawled_urls.add(article["url"])
        _csv_writer.writerow(article)
        _csv_handle.flush()  # flush ngay — không mất dữ liệu nếu dừng giữa chừng


# ─────────────────────────────────────────────
# WORKER
# ─────────────────────────────────────────────


def worker(url: str) -> Optional[dict]:
    if url in _crawled_urls:
        return None
    time.sleep(CONFIG["request_delay"])
    article = parse_article(url)
    if article:
        write_article(article)
    return article


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def main():
    url_path = CONFIG["input_urls"]
    csv_path = CONFIG["output_csv"]

    logger.info("=" * 55)
    logger.info("  VnExpress Article Crawler")
    logger.info(f"  Input URLs : {url_path}")
    logger.info(f"  Output CSV : {csv_path}")
    logger.info(f"  Luồng      : {CONFIG['max_workers']}")
    logger.info("=" * 55)

    # Đọc danh sách URL
    if not os.path.exists(url_path):
        logger.error(
            f"❌ Không tìm thấy {url_path} — hãy chạy 1_url_collector.py trước!"
        )
        return

    with open(url_path, encoding="utf-8") as f:
        all_urls = [line.strip() for line in f if line.strip()]

    logger.info(f"📋 Đọc được {len(all_urls):,} URL từ {url_path}")

    # Khởi tạo CSV (load resume state)
    init_csv(csv_path)

    # Lọc URL chưa crawl
    pending = [u for u in all_urls if u not in _crawled_urls]
    skipped = len(all_urls) - len(pending)
    if skipped:
        logger.info(f"⏭️  Bỏ qua {skipped:,} bài đã crawl trước đó")
    logger.info(
        f"\n🚀 Bắt đầu crawl {len(pending):,} bài với {CONFIG['max_workers']} luồng...\n"
    )

    done = failed = 0
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = {executor.submit(worker, url): url for url in pending}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    done += 1
                    if done % 100 == 0:
                        logger.info(
                            f"   ✅ {done:,}/{len(pending):,} bài | ❌ {failed} lỗi"
                        )
                else:
                    failed += 1
            except Exception:
                failed += 1

    _csv_handle.close()
    logger.info(f"\n✅ Hoàn tất! {done:,} bài thành công | {failed} thất bại")
    logger.info(f"📊 Tổng cộng {len(_crawled_urls):,} bài trong {csv_path}")


if __name__ == "__main__":
    main()
