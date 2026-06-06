"""Gemini reader layer for retrieval-augmented Vietnamese QA."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import requests


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_MAX_CONTEXT_DOCS = 5
DEFAULT_MAX_CONTEXT_CHARS = 1200
NO_EVIDENCE_ANSWER = "Không đủ thông tin trong các bài báo tìm được."
MISSING_KEY_ANSWER = "Chưa cấu hình GEMINI_API_KEY nên Reader chưa thể sinh câu trả lời QA."
GEMINI_ERROR_ANSWER = "Gemini Reader đang lỗi, vui lòng kiểm tra GEMINI_API_KEY hoặc GEMINI_MODEL."


class QAReader:
    """Generate a concise Gemini answer from retrieved news snippets."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_context_docs: int = DEFAULT_MAX_CONTEXT_DOCS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ):
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except Exception:
            pass

        self.model = (
            model if model is not None else os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        )
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self.max_context_docs = int(max_context_docs)
        self.max_context_chars = int(max_context_chars)

    def answer(
        self,
        query: str,
        retrieval_results: List[Dict[str, Any]],
        max_context_docs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return answer, citations, used_contexts, confidence, is_answerable."""
        contexts = self.build_contexts(
            retrieval_results,
            max_context_docs=max_context_docs or self.max_context_docs,
        )
        if not contexts:
            return self._empty_answer(reason="no_context")

        if not self.api_key:
            return self._error_answer("missing_gemini_api_key", contexts)

        try:
            llm_answer = self._call_gemini(query, contexts)
        except Exception as exc:
            return self._error_answer(f"gemini_error:{type(exc).__name__}", contexts)

        if not llm_answer or not llm_answer.strip():
            return self._error_answer("empty_gemini_answer", contexts)

        is_answerable = NO_EVIDENCE_ANSWER.lower() not in llm_answer.lower()
        return {
            "answer": llm_answer.strip(),
            "citations": self._citations_from_answer(llm_answer, contexts),
            "used_contexts": contexts,
            "confidence": "medium" if is_answerable else "low",
            "is_answerable": is_answerable,
            "provider": "gemini",
            "model": self.model,
            "error": None,
        }

    def build_contexts(
        self,
        retrieval_results: List[Dict[str, Any]],
        max_context_docs: int = DEFAULT_MAX_CONTEXT_DOCS,
    ) -> List[Dict[str, Any]]:
        contexts: List[Dict[str, Any]] = []
        seen_docs = set()

        for doc in retrieval_results or []:
            doc_id = str(doc.get("id") or doc.get("url") or doc.get("title") or "")
            if doc_id and doc_id in seen_docs:
                continue
            if doc_id:
                seen_docs.add(doc_id)

            text = self._context_text(doc)
            if not text:
                continue

            source_id = f"S{len(contexts) + 1}"
            contexts.append(
                {
                    "source_id": source_id,
                    "rank": len(contexts) + 1,
                    "doc_id": doc.get("id", ""),
                    "chunk_id": doc.get("chunk_id", ""),
                    "title": doc.get("title", "(không có tiêu đề)"),
                    "url": doc.get("url", ""),
                    "date": doc.get("date", ""),
                    "category": doc.get("category", ""),
                    "score": float(doc.get("retrieval_score", 0.0) or 0.0),
                    "snippet": text[:360],
                    "text": text,
                }
            )
            if len(contexts) >= max_context_docs:
                break
        return contexts

    def _call_gemini(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": self._build_prompt(query, contexts)}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": int(os.getenv("READER_MAX_TOKENS", "320")),
                "temperature": float(os.getenv("READER_TEMPERATURE", "0")),
            },
        }
        response = requests.post(
            url,
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=float(os.getenv("READER_TIMEOUT_SECONDS", "30")),
        )
        response.raise_for_status()
        data = response.json()

        parts = []
        for candidate in data.get("candidates", []) or []:
            content = candidate.get("content", {}) or {}
            for part in content.get("parts", []) or []:
                text = part.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()

    def _build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        source_blocks = []
        for ctx in contexts:
            meta = " | ".join(
                part
                for part in [
                    ctx.get("title", ""),
                    ctx.get("date", ""),
                    ctx.get("category", ""),
                    ctx.get("url", ""),
                ]
                if part
            )
            source_blocks.append(f"[{ctx['source_id']}] {meta}\n{ctx['text']}")

        return (
            "Bạn là reader QA cho hệ thống tìm kiếm tin tức tiếng Việt.\n"
            "Chỉ dùng thông tin trong SOURCES, không bịa ngoài nguồn.\n"
            "Trả lời ngắn gọn trong 2-4 câu, bằng tiếng Việt.\n"
            "Mỗi ý quan trọng phải có citation dạng [S1], [S2].\n"
            f"Nếu SOURCES không đủ bằng chứng, trả đúng câu: {NO_EVIDENCE_ANSWER}\n\n"
            f"QUESTION:\n{query.strip()}\n\n"
            "SOURCES:\n"
            + "\n\n".join(source_blocks)
        )

    def _empty_answer(self, reason: str) -> Dict[str, Any]:
        return {
            "answer": NO_EVIDENCE_ANSWER,
            "citations": [],
            "used_contexts": [],
            "confidence": "low",
            "is_answerable": False,
            "provider": "gemini",
            "model": self.model,
            "error": reason,
        }

    def _error_answer(
        self,
        reason: str,
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        answer = (
            MISSING_KEY_ANSWER
            if reason == "missing_gemini_api_key"
            else GEMINI_ERROR_ANSWER
        )
        return {
            "answer": answer,
            "citations": [],
            "used_contexts": contexts or [],
            "confidence": "low",
            "is_answerable": False,
            "provider": "gemini",
            "model": self.model,
            "error": reason,
        }

    def _citations_from_answer(
        self,
        answer: str,
        contexts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        cited_ids = set(re.findall(r"\[(S\d+)\]", answer or ""))
        if not cited_ids and answer and answer != NO_EVIDENCE_ANSWER:
            cited_ids = {ctx["source_id"] for ctx in contexts[:2]}

        by_id = {ctx["source_id"]: ctx for ctx in contexts}
        citations = []
        for source_id in sorted(cited_ids, key=lambda s: int(s[1:])):
            ctx = by_id.get(source_id)
            if not ctx:
                continue
            citations.append(
                {
                    "source_id": source_id,
                    "rank": ctx.get("rank"),
                    "title": ctx.get("title", ""),
                    "url": ctx.get("url", ""),
                    "date": ctx.get("date", ""),
                    "category": ctx.get("category", ""),
                    "doc_id": ctx.get("doc_id", ""),
                    "chunk_id": ctx.get("chunk_id", ""),
                    "snippet": ctx.get("snippet", ""),
                }
            )
        return citations

    def _context_text(self, doc: Dict[str, Any]) -> str:
        text = (
            doc.get("chunk_text")
            or doc.get("snippet")
            or doc.get("full_text")
            or doc.get("content")
            or ""
        )
        text = re.sub(r"\s+", " ", str(text)).strip()
        return text[: self.max_context_chars]
