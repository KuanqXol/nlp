"""
VnExpress URL Collector
=======================
Chỉ thu thập URL bài báo và lưu vào file txt.
Chạy file này trước, sau đó chạy 2_article_crawler.py.

Cài đặt:
    pip install requests beautifulsoup4

Chạy:
    python 1_url_collector.py
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import logging
import re
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urljoin

# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
CONFIG = {
    "start_date":      datetime(2001, 1, 1),  # Từ ngày
    "end_date":        datetime.now(),         # Đến ngày
    "output_urls":     "vnexpress_urls.txt",   # File lưu URL
    "request_timeout": 15,
    "max_retries":     3,
    # Chuyên mục muốn lấy — để None để lấy tất cả
    "categories":      None,
}

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

CATEGORY_IDS = {
    "thoi-su":    1001005,
    "the-gioi":   1001002,
    "kinh-doanh": 1003159,
    "giai-tri":   1002691,
    "the-thao":   1002565,
    "phap-luat":  1001007,
    "giao-duc":   1003497,
    "suc-khoe":   1003750,
    "du-lich":    1003231,
    "khoa-hoc":   1001009,
    "so-hoa":     1002592,
    "xe":         1001006,
    "y-kien":     1001012,
    "tam-su":     1001014,
}

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("url_collector.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────
_session = requests.Session()
_session.headers.update(HEADERS)


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
# URL EXTRACTION
# ─────────────────────────────────────────────
_ARTICLE_URL_RE = re.compile(r"https://vnexpress\.net/[^\"'\s]+\.html")


def _is_article_url(url: str) -> bool:
    if not url.endswith(".html"):
        return False
    if "vnexpress.net" not in url:
        return False
    skip = ["/topic/", "/tag/", "/category/", "/author/",
            "/search/", "/photo/", "/video/", "page="]
    return not any(p in url for p in skip)


def extract_urls_from_html(html: str) -> list[str]:
    found, seen = [], set()

    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.startswith("http"):
            href = urljoin(BASE_URL, href)
        if _is_article_url(href) and href not in seen:
            found.append(href)
            seen.add(href)

    if len(found) < 5:  # fallback
        for href in _ARTICLE_URL_RE.findall(html):
            if _is_article_url(href) and href not in seen:
                found.append(href)
                seen.add(href)

    return found


# ─────────────────────────────────────────────
# SEARCH API — chia theo tháng
# ─────────────────────────────────────────────

def month_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    chunks, cur = [], start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        nxt = (cur.replace(year=cur.year + 1, month=1, day=1)
               if cur.month == 12
               else cur.replace(month=cur.month + 1, day=1))
        chunks.append((cur, min(nxt - timedelta(seconds=1), end)))
        cur = nxt
    return chunks


def fetch_urls_one_month(cat_id: int, from_dt: datetime, to_dt: datetime) -> list[str]:
    urls, seen = [], set()
    from_ts, to_ts = int(from_dt.timestamp()), int(to_dt.timestamp())

    for page in range(1, 51):
        api = (f"https://vnexpress.net/category/day/cateid/{cat_id}"
               f"/fromdate/{from_ts}/todate/{to_ts}/allcate/0/page/{page}")
        html = fetch(api)
        if not html:
            break
        found = 0
        for u in extract_urls_from_html(html):
            if u not in seen:
                urls.append(u)
                seen.add(u)
                found += 1
        if found == 0:
            break
        time.sleep(0.3)

    return urls


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    start      = CONFIG["start_date"]
    end        = CONFIG["end_date"]
    url_path   = CONFIG["output_urls"]
    cats       = CONFIG["categories"] or list(CATEGORY_IDS.keys())
    chunks     = month_chunks(start, end)

    logger.info("=" * 55)
    logger.info("  VnExpress URL Collector")
    logger.info(f"  Thời gian  : {start:%d/%m/%Y} → {end:%d/%m/%Y}")
    logger.info(f"  Chuyên mục : {len(cats)}")
    logger.info(f"  Số tháng   : {len(chunks)}")
    logger.info(f"  Output     : {url_path}")
    logger.info("=" * 55)

    # Resume: load URL đã có
    known: set[str] = set()
    if os.path.exists(url_path):
        with open(url_path, encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u:
                    known.add(u)
        logger.info(f"🔄 Resume: {len(known):,} URL đã có trong {url_path}")

    # Mở file append
    out = open(url_path, "a", encoding="utf-8")
    total_new = 0

    for slug in cats:
        cat_id = CATEGORY_IDS.get(slug)
        if not cat_id:
            continue

        cat_new = 0
        for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
            urls = fetch_urls_one_month(cat_id, chunk_start, chunk_end)
            new  = [u for u in urls if u not in known]

            for u in new:
                known.add(u)
                out.write(u + "\n")
            if new:
                out.flush()

            cat_new   += len(new)
            total_new += len(new)

            logger.info(
                f"  [{slug}] {chunk_start:%m/%Y} "
                f"+{len(new):>4} URL | tháng {i:>3}/{len(chunks)} | "
                f"tổng: {len(known):,}"
            )

    out.close()
    logger.info(f"\n✅ Hoàn tất! Thêm {total_new:,} URL mới → tổng {len(known):,} URL trong {url_path}")


if __name__ == "__main__":
    main()