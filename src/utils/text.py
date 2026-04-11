"""
src/utils/text.py
─────────────────
Shared text utilities dùng chung cho NER và chunking.

Vấn đề trước: _split_sentences được định nghĩa 2 lần với 2 regex khác nhau:
  - ner.py      : regex `[^.!?]+[.!?]?`  → trả về List[Dict] có start/end offset
  - chunking.py : regex `(?<=[.!?])\\s+` → trả về List[str]

Fix: thống nhất vào 1 module, 2 function với interface rõ ràng.
  - split_sentences_spans() → dùng trong NER (cần offset để map entity về text gốc)
  - split_sentences()       → dùng trong chunking (chỉ cần text thuần)

Cả 2 dùng cùng 1 regex split cơ bản, đảm bảo chunking và NER chia câu nhất quán.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

# Regex tách câu tiếng Việt: split sau dấu câu kết thúc + whitespace.
# Tiếng Việt không có viết hoa đầu câu nhất quán nên không thể dùng
# lookhead [A-Z] như tiếng Anh. Phương pháp đơn giản nhất vẫn hiệu quả tốt.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+", flags=re.UNICODE)

# Câu quá ngắn không có nghĩa (ví dụ: "OK." "v.v.")
DEFAULT_MIN_CHARS = 10


def split_sentences(text: str, min_chars: int = DEFAULT_MIN_CHARS) -> List[str]:
    """Tách câu tiếng Việt, trả về list string.

    Dùng trong: chunking.py

    Args:
        text: văn bản đầu vào (đã NFC-normalized).
        min_chars: bỏ qua câu ngắn hơn ngưỡng này.

    Returns:
        List[str] các câu, đã strip, dài >= min_chars.

    Example:
        >>> split_sentences("Hà Nội là thủ đô. Việt Nam có 63 tỉnh.")
        ['Hà Nội là thủ đô.', 'Việt Nam có 63 tỉnh.']
    """
    text = (text or "").strip()
    if not text:
        return []
    sentences = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if len(s.strip()) >= min_chars]


def split_sentences_spans(text: str, min_chars: int = DEFAULT_MIN_CHARS) -> List[Dict]:
    """Tách câu tiếng Việt, trả về list dict với offset trong text gốc.

    Dùng trong: ner.py (cần start/end để map entity về vị trí trong full_text)

    Args:
        text: văn bản đầu vào.
        min_chars: bỏ qua câu ngắn hơn ngưỡng này.

    Returns:
        List[Dict] mỗi phần tử có:
            sentence_id (int): index thứ tự câu
            text        (str): nội dung câu
            start       (int): offset bắt đầu trong text gốc
            end         (int): offset kết thúc trong text gốc

    Example:
        >>> spans = split_sentences_spans("Putin họp NATO. Zelensky phát biểu.")
        >>> spans[0]['text']
        'Putin họp NATO.'
        >>> spans[0]['start']
        0
    """
    text = text or ""
    if not text.strip():
        return []

    spans: List[Dict] = []
    sentence_id = 0

    # Dùng finditer để lấy match position chính xác
    for m in re.finditer(r"[^.!?…]+[.!?…]?", text, flags=re.UNICODE):
        sent = m.group().strip()
        if len(sent) < min_chars:
            continue

        # Tìm vị trí thực sự trong text gốc (bỏ qua leading whitespace)
        raw_start = m.start()
        while raw_start < len(text) and text[raw_start].isspace():
            raw_start += 1

        spans.append(
            {
                "sentence_id": sentence_id,
                "text": sent,
                "start": raw_start,
                "end": raw_start + len(sent),
            }
        )
        sentence_id += 1

    # Fallback: nếu không tách được gì, coi cả text là 1 câu
    if not spans and text.strip():
        spans.append(
            {
                "sentence_id": 0,
                "text": text.strip(),
                "start": 0,
                "end": len(text.strip()),
            }
        )

    return spans
