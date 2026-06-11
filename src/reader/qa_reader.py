"""Local ViT5 reader/summarizer for retrieval-augmented Vietnamese news."""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


BASE_VIT5_MODEL = "VietAI/vit5-base"
DEFAULT_VIT5_MODEL = "VietAI/vit5-base-vietnews-summarization"
DEFAULT_MAX_CONTEXT_DOCS = 5
DEFAULT_MAX_CONTEXT_CHARS = 1200
DEFAULT_MIN_CONTEXT_CHARS = 80
DEFAULT_MIN_CONTEXT_SCORE = 0.30
DEFAULT_MIN_QUERY_OVERLAP = 0.34
NO_EVIDENCE_ANSWER = "Không đủ thông tin trong các bài báo tìm được."
VIT5_ERROR_ANSWER = "ViT5 Reader đang lỗi, vui lòng kiểm tra cấu hình model local."

INVESTMENT_SIGNAL_PATTERNS = [
    r"\bdau tu\b",
    r"\bvon dau tu\b",
    r"\btong von\b",
    r"\brot von\b",
    r"\bdu an\b",
    r"\bnha may\b",
    r"\bkhu cong nghiep\b",
    r"\bfdi\b",
    r"\bty usd\b",
    r"\btrieu usd\b",
]

INDIRECT_INVESTMENT_PATTERNS = [
    r"\bho tro dao tao\b",
    r"\bdao tao\b",
    r"\bky su\b",
    r"\bban dan\b",
    r"\bhop tac\b",
]

STOPWORDS = {
    "ai",
    "an",
    "cua",
    "của",
    "co",
    "có",
    "cho",
    "da",
    "đã",
    "de",
    "để",
    "di",
    "đi",
    "do",
    "vao",
    "vào",
    "va",
    "và",
    "ve",
    "về",
    "tai",
    "tại",
    "the",
    "thế",
    "nhu",
    "như",
    "nao",
    "nào",
    "bao",
    "nhieu",
    "nhiêu",
    "mot",
    "một",
    "cac",
    "các",
    "nhung",
    "những",
    "duoc",
    "được",
    "la",
    "là",
    "se",
    "sẽ",
    "tu",
    "từ",
    "trong",
    "ngoai",
    "ngoài",
    "voi",
    "với",
}


class QAReader:
    """Summarize retrieved Vietnamese news contexts with a local ViT5 model."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_context_docs: int = DEFAULT_MAX_CONTEXT_DOCS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        min_context_chars: int = DEFAULT_MIN_CONTEXT_CHARS,
        min_context_score: float = DEFAULT_MIN_CONTEXT_SCORE,
        min_query_overlap: float = DEFAULT_MIN_QUERY_OVERLAP,
    ):
        # api_key is intentionally ignored; kept only so older call sites/tests do not break.
        _ = api_key
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except Exception:
            pass

        env_model = os.getenv("VIT5_MODEL")
        if model is not None:
            self.model_name = model
        elif env_model and env_model.strip() and env_model.strip() != BASE_VIT5_MODEL:
            self.model_name = env_model.strip()
        else:
            self.model_name = DEFAULT_VIT5_MODEL
        self.max_context_docs = int(max_context_docs)
        self.max_context_chars = self._env_int("VIT5_CONTEXT_CHARS", max_context_chars)
        self.min_context_chars = self._env_int("VIT5_MIN_CONTEXT_CHARS", min_context_chars)
        self.min_context_score = self._env_float("VIT5_MIN_CONTEXT_SCORE", min_context_score)
        self.min_query_overlap = self._env_float("VIT5_MIN_QUERY_OVERLAP", min_query_overlap)
        self.max_input_chars = self._env_int("VIT5_MAX_INPUT_CHARS", 3500)
        self.max_length = self._env_int("VIT5_MAX_LENGTH", 1024)
        self.max_new_tokens = self._env_int("VIT5_MAX_NEW_TOKENS", 220)
        self.num_beams = self._env_int("VIT5_NUM_BEAMS", 4)
        self.device_setting = os.getenv("VIT5_DEVICE", "auto").lower()

        self._tokenizer = None
        self._model = None
        self._device = None

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.getenv(name)
        try:
            return int(value) if value not in (None, "") else int(default)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.getenv(name)
        try:
            return float(value) if value not in (None, "") else float(default)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _format_error(prefix: str, exc: Exception) -> str:
        message = re.sub(r"\s+", " ", str(exc or "")).strip()
        if message:
            return f"{prefix}:{type(exc).__name__}:{message[:120]}"
        return f"{prefix}:{type(exc).__name__}"

    def answer(
        self,
        query: str,
        retrieval_results: List[Dict[str, Any]],
        max_context_docs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return answer, citations, used_contexts, confidence, is_answerable."""
        contexts = self.build_contexts(
            query,
            retrieval_results,
            max_context_docs=max_context_docs or self.max_context_docs,
        )
        if not contexts:
            return self._empty_answer(reason="no_context")
        contexts = [ctx for ctx in contexts if self._is_context_relevant(ctx)]
        if not contexts:
            return self._empty_answer(reason="weak_context")

        prompt, used_contexts = self._build_vit5_prompt(query, contexts)
        if not used_contexts:
            return self._empty_answer(reason="no_context")

        factoid_answer = self._answer_factoid(query, used_contexts)
        if factoid_answer:
            answer = factoid_answer
            citations = self._citations_from_answer(answer, used_contexts)
            confidence = self._estimate_confidence(
                answer,
                citations,
                used_contexts,
                force_low_confidence=False,
            )
            return self._build_payload(
                answer=answer,
                citations=citations,
                used_contexts=used_contexts,
                confidence=confidence,
                is_answerable=bool(citations),
                error=None,
                reader_mode="factoid_extractive",
            )

        indirect_answer = self._answer_indirect_intent(query, used_contexts)
        if indirect_answer:
            citations = self._citations_from_answer(indirect_answer, used_contexts)
            return self._build_payload(
                answer=indirect_answer,
                citations=citations,
                used_contexts=used_contexts,
                confidence="low",
                is_answerable=False,
                error="indirect_context",
                reader_mode="intent_indirect",
            )

        error = None
        try:
            raw_answer = self._generate_vit5(prompt)
            answer = self._postprocess_answer(raw_answer)
        except Exception as exc:
            error = self._format_error("vit5_error", exc)
            answer = ""
        if not answer:
            error = error or "vit5_empty_output"
        else:
            answer = self._ensure_citation(answer, used_contexts)
            answer = self._clean_invalid_citations(answer, used_contexts)

        valid, validation_reason = self._validate_answer(query, answer, used_contexts)
        if not valid:
            return self._unreliable_answer(error or validation_reason, used_contexts)

        citations = self._citations_from_answer(answer, used_contexts)
        confidence = self._estimate_confidence(
            answer,
            citations,
            used_contexts,
            force_low_confidence=False,
        )
        payload_error = error if error and error.startswith("vit5_error:") else None
        return self._build_payload(
            answer=answer,
            citations=citations,
            used_contexts=used_contexts,
            confidence=confidence,
            is_answerable=bool(citations),
            error=payload_error,
            reader_mode="vit5_summarization",
        )

    def _build_payload(
        self,
        answer: str,
        citations: List[Dict[str, Any]],
        used_contexts: List[Dict[str, Any]],
        confidence: str,
        is_answerable: bool,
        error: Optional[str],
        reader_mode: str,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "citations": citations,
            "used_contexts": used_contexts,
            "selected_sentences": [
                {
                    "source_id": ctx["source_id"],
                    "text": ctx.get("selected_text", ""),
                    "context_score": ctx.get("context_score", 0.0),
                }
                for ctx in used_contexts
            ],
            "confidence": confidence,
            "is_answerable": is_answerable,
            "provider": "vit5-local",
            "model": self.model_name,
            "error": error,
            "reader_mode": reader_mode,
        }

    def build_contexts(
        self,
        query: str,
        retrieval_results: List[Dict[str, Any]],
        max_context_docs: int = DEFAULT_MAX_CONTEXT_DOCS,
    ) -> List[Dict[str, Any]]:
        raw_contexts: List[Dict[str, Any]] = []
        seen_docs = set()

        for doc in retrieval_results or []:
            doc_id = str(doc.get("id") or doc.get("url") or doc.get("title") or "")
            if doc_id and doc_id in seen_docs:
                continue
            if doc_id:
                seen_docs.add(doc_id)

            text = self._context_text(doc)
            if len(text) < self.min_context_chars:
                continue

            raw_contexts.append(
                {
                    "doc_id": doc.get("id", ""),
                    "chunk_id": doc.get("chunk_id", ""),
                    "title": doc.get("title", "(không có tiêu đề)"),
                    "url": doc.get("url", ""),
                    "date": doc.get("date", ""),
                    "category": doc.get("category", ""),
                    "retrieval_score": float(doc.get("retrieval_score", 0.0) or 0.0),
                    "snippet": text[:360],
                    "text": text,
                }
            )

        if not raw_contexts:
            return []

        max_retrieval_score = max(ctx["retrieval_score"] for ctx in raw_contexts) or 1.0
        scored_contexts = []
        for ctx in raw_contexts:
            context_score, score_parts = self._score_context(query, ctx, max_retrieval_score)
            if context_score <= 0.0:
                continue
            ctx.update(score_parts)
            ctx["context_score"] = round(context_score, 4)
            scored_contexts.append(ctx)

        scored_contexts.sort(
            key=lambda item: (item.get("context_score", 0.0), item.get("retrieval_score", 0.0)),
            reverse=True,
        )

        contexts = scored_contexts[:max_context_docs]
        for idx, ctx in enumerate(contexts, start=1):
            ctx["source_id"] = f"S{idx}"
            ctx["rank"] = idx
        return contexts

    def _score_context(
        self,
        query: str,
        context: Dict[str, Any],
        max_retrieval_score: float,
    ) -> Tuple[float, Dict[str, float]]:
        intent = self._detect_query_intent(query)
        query_terms = self._content_terms(query)
        text = f"{context.get('title', '')} {context.get('text', '')}"
        text_terms = set(self._content_terms(text))
        title_terms = set(self._content_terms(context.get("title", "")))

        retrieval_norm = min(max(context.get("retrieval_score", 0.0) / max_retrieval_score, 0.0), 1.0)
        query_overlap = self._overlap_ratio(query_terms, text_terms)
        entity_terms = self._entity_terms(query_terms)
        entity_overlap = self._overlap_ratio(entity_terms, text_terms)
        number_bonus = 0.0 if intent == "who_is" else 1.0 if self._has_number_signal(text) else 0.0
        title_overlap = self._overlap_ratio(query_terms, title_terms)
        intent_bonus = self._intent_context_bonus(query, context, intent)
        intent_directness = self._intent_directness_score(query, text, intent)

        score = (
            0.45 * retrieval_norm
            + 0.20 * query_overlap
            + 0.15 * entity_overlap
            + 0.10 * number_bonus
            + 0.10 * title_overlap
            + intent_bonus
        )
        score_parts = {
            "retrieval_score_norm": round(retrieval_norm, 4),
            "query_overlap": round(query_overlap, 4),
            "entity_overlap": round(entity_overlap, 4),
            "number_bonus": round(number_bonus, 4),
            "title_overlap": round(title_overlap, 4),
            "intent_bonus": round(intent_bonus, 4),
            "intent_directness": round(intent_directness, 4),
        }
        return score, score_parts

    def _build_vit5_prompt(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        parts = []
        used_contexts = []
        remaining_chars = self.max_input_chars

        for ctx in contexts:
            selected = self._compress_context(query, ctx)
            if not selected:
                continue
            title = ctx.get("title", "")
            block = f"{title}. {selected}" if title else selected
            if len(block) > remaining_chars and used_contexts:
                break
            if len(block) > remaining_chars:
                block = block[: max(0, remaining_chars)]
            ctx = dict(ctx)
            ctx["selected_text"] = selected
            used_contexts.append(ctx)
            parts.append(block)
            parts.append("")
            remaining_chars -= len(block)
            if remaining_chars <= 0:
                break

        return "\n".join(parts)[: self.max_input_chars], used_contexts

    def _compress_context(self, query: str, context: Dict[str, Any], max_sentences: int = 2) -> str:
        sentences = self._split_sentences(context.get("text", ""))
        if not sentences:
            return context.get("text", "")[:360]

        query_terms = set(self._content_terms(query))
        scored = []
        for idx, sentence in enumerate(sentences):
            terms = set(self._content_terms(sentence))
            overlap = self._overlap_ratio(query_terms, terms)
            number_bonus = 0.25 if self._has_number_signal(sentence) else 0.0
            length_bonus = 0.15 if 60 <= len(sentence) <= 260 else 0.0
            score = overlap + number_bonus + length_bonus
            scored.append((score, idx, sentence))

        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected = [item for item in scored[:max_sentences] if item[0] > 0.0]
        selected.sort(key=lambda item: item[1])
        if not selected:
            selected = scored[:1]
        return " ".join(sentence for _, _, sentence in selected).strip()

    def _generate_vit5(self, prompt: str) -> str:
        tokenizer, model, device = self._load_vit5()
        import torch

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        sequences = getattr(outputs, "sequences", outputs)
        if hasattr(sequences, "dim") and sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return (decoded[0] if decoded else "").strip()

    def _load_vit5(self):
        if self._tokenizer is not None and self._model is not None and self._device is not None:
            return self._tokenizer, self._model, self._device

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.device_setting == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device_setting == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("VIT5_DEVICE=cuda nhưng CUDA không khả dụng")
        else:
            device = self.device_setting

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model.to(device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        return tokenizer, model, device

    def _postprocess_answer(self, answer: str) -> str:
        text = re.sub(r"\*\*", "", answer or "")
        text = re.sub(r"^\s*(câu trả lời|tóm tắt)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<!\[)\bS(\d+)\b", r"[S\1]", text)
        text = re.sub(r"\s+", " ", text).strip(" '\"\n\t")
        sentences = self._split_sentences(text)
        if len(sentences) > 3:
            text = " ".join(sentences[:3])
        return text

    def _ensure_citation(self, answer: str, contexts: List[Dict[str, Any]]) -> str:
        if re.search(r"\[S\d+\]", answer or "") or not contexts:
            return answer
        return f"{answer.rstrip()} [{contexts[0]['source_id']}]"

    def _validate_answer(
        self,
        query: str,
        answer: str,
        contexts: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        if not answer or len(answer.strip()) < 20:
            return False, "answer_too_short"
        lower = answer.lower()
        if any(bad in lower for bad in ["mentions", "source mentions", "article says"]):
            return False, "answer_is_source_note"
        if self._leaks_prompt_or_source_format(answer):
            return False, "answer_leaks_prompt"
        if self._looks_like_gibberish(answer):
            return False, "answer_is_gibberish"
        if self._looks_like_english_note(answer):
            return False, "answer_is_english_note"
        if self._looks_like_title_only(answer, contexts):
            return False, "answer_is_title_only"
        if not any(self._is_context_relevant(ctx) for ctx in contexts):
            return False, "weak_context"
        if not re.search(r"\[S\d+\]", answer):
            return False, "missing_citation"
        query_terms = set(self._content_terms(query))
        answer_terms = set(self._content_terms(answer))
        if query_terms and self._overlap_ratio(query_terms, answer_terms) < 0.10:
            return False, "low_query_overlap"
        valid_source_ids = {ctx["source_id"] for ctx in contexts}
        cited_ids = set(re.findall(r"\[(S\d+)\]", answer))
        if not cited_ids.intersection(valid_source_ids):
            return False, "invalid_citation"
        return True, ""

    def _leaks_prompt_or_source_format(self, answer: str) -> bool:
        folded = self._fold_text(answer)
        leaked_phrases = [
            "tra dung",
            "tra sai",
            "nguon:",
            "yeu cau:",
            "khong du thong tin trong nguon",
            "source:",
        ]
        if any(phrase in folded for phrase in leaked_phrases):
            return True
        if len(re.findall(r"\bnguon\b", folded)) >= 2:
            return True
        if re.search(r"[a-z0-9]+(?:-[a-z0-9]+){3,}", folded):
            return True
        return False

    def _looks_like_gibberish(self, answer: str) -> bool:
        text = answer or ""
        if not text:
            return True

        allowed = re.findall(r"[0-9A-Za-zÀ-ỹĐđ\s.,;:!?%/\-\[\]\(\)\"'“”]", text)
        strange_ratio = 1.0 - (len(allowed) / max(1, len(text)))
        if strange_ratio > 0.08:
            return True

        compact = re.sub(r"\s+", "", text)
        punctuation = re.findall(r"[^0-9A-Za-zÀ-ỹĐđ\s]", compact)
        if len(punctuation) / max(1, len(compact)) > 0.22:
            return True

        if re.search(r"[\]\[_&+*<>}{|\\]{3,}", text):
            return True
        if re.search(r"([À-ỹĐđA-Za-z])\1{4,}", text):
            return True
        if len(re.findall(r"\[[^\]]{0,3}$|^[^\[]*\]", text)) > 0:
            return True

        words = re.findall(r"[A-Za-zÀ-ỹĐđ]{2,}", text)
        if len(words) >= 8:
            short_or_noise = [word for word in words if len(word) <= 2]
            if len(short_or_noise) / len(words) > 0.45:
                return True
        return False

    def _is_context_relevant(self, context: Dict[str, Any]) -> bool:
        score = float(context.get("context_score", 0.0) or 0.0)
        query_overlap = float(context.get("query_overlap", 0.0) or 0.0)
        entity_overlap = float(context.get("entity_overlap", 0.0) or 0.0)
        has_number = float(context.get("number_bonus", 0.0) or 0.0) > 0.0

        if score < self.min_context_score:
            return False
        if query_overlap >= self.min_query_overlap:
            return True
        if entity_overlap >= 0.50:
            return True
        return has_number and query_overlap >= max(0.20, self.min_query_overlap - 0.12)

    def _looks_like_english_note(self, answer: str) -> bool:
        lower = (answer or "").lower()
        english_phrases = [
            "according to",
            "answer:",
            "summary:",
            "source:",
            "article",
            "mentions",
        ]
        if any(phrase in lower for phrase in english_phrases):
            return True

        ascii_words = re.findall(r"\b[a-zA-Z]{3,}\b", answer or "")
        vietnamese_marked = re.findall(r"[À-ỹĐđ]", answer or "")
        vietnamese_common = {
            "theo",
            "các",
            "bài",
            "báo",
            "tìm",
            "được",
            "việt",
            "nam",
            "đầu",
            "tư",
            "nguồn",
        }
        answer_terms = set(self._content_terms(answer))
        has_vietnamese_common = bool(answer_terms & vietnamese_common)
        return len(ascii_words) >= 4 and not vietnamese_marked and not has_vietnamese_common

    def _looks_like_title_only(self, answer: str, contexts: List[Dict[str, Any]]) -> bool:
        answer_norm = self._normalize_for_compare(re.sub(r"\[S\d+\]", "", answer or ""))
        if not answer_norm:
            return False
        for ctx in contexts:
            title_norm = self._normalize_for_compare(ctx.get("title", ""))
            if title_norm and answer_norm == title_norm:
                return True
        return False

    def _answer_factoid(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        intent = self._detect_query_intent(query)
        if intent == "who_is":
            return self._answer_who_is(query, contexts)
        return ""

    def _answer_indirect_intent(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
    ) -> str:
        intent = self._detect_query_intent(query)
        if intent != "investment" or not contexts:
            return ""

        direct_contexts = [
            ctx
            for ctx in contexts
            if self._intent_directness_score(
                query,
                f"{ctx.get('title', '')} {ctx.get('selected_text', '')} {ctx.get('text', '')}",
                intent,
            )
            >= 0.45
        ]
        if direct_contexts:
            return ""

        top = contexts[0]
        source_id = top.get("source_id", "S1")
        evidence = self._short_evidence_sentence(
            top.get("selected_text") or top.get("text") or top.get("title", ""),
            query=query,
            title=top.get("title", ""),
        )
        scope = self._investment_scope_text(query)
        if evidence:
            return (
                "Nguồn hiện tại chỉ liên quan gián tiếp đến intent đầu tư của truy vấn: "
                f"nguồn này nêu rằng {self._lower_first(evidence)} [{source_id}]. "
                f"Chưa thấy nguồn trong top kết quả nêu trực tiếp khoản vốn, dự án hoặc nhà máy đầu tư {scope}, "
                "nên đây chưa phải câu trả lời đầy đủ."
            )
        return (
            "Nguồn hiện tại chỉ liên quan gián tiếp đến intent đầu tư của truy vấn "
            f"và chưa có bằng chứng trực tiếp về khoản vốn, dự án hoặc nhà máy đầu tư {scope} [{source_id}]."
        )

    def _detect_query_intent(self, query: str) -> str:
        if self._is_who_is_query(query):
            return "who_is"
        if self._is_investment_query(query):
            return "investment"
        return "summarize"

    def _intent_context_bonus(self, query: str, context: Dict[str, Any], intent: str) -> float:
        if intent == "investment":
            text = f"{context.get('title', '')} {context.get('text', '')}"
            directness = self._intent_directness_score(query, text, intent)
            return 0.20 * directness if directness >= 0.45 else -0.08

        if intent != "who_is":
            return 0.0

        query_norm = self._fold_text(query)
        title_norm = self._fold_text(context.get("title", ""))
        context_norm = self._fold_text(f"{context.get('title', '')} {context.get('text', '')}")
        bonus = 0.0
        if self._role_matches_query(query_norm, title_norm):
            bonus += 0.20
        if self._geo_matches_query(query_norm, context_norm):
            bonus += 0.15
        if self._has_who_is_title_clue(title_norm):
            bonus += 0.45
        if any(bad in title_norm for bad in ["xep hang", "lich su", "cuu tong thong", "khao sat"]):
            bonus -= 0.35
        return bonus

    def _intent_directness_score(self, query: str, text: str, intent: str) -> float:
        if intent != "investment":
            return 1.0

        folded_text = self._fold_text(text)
        folded_query = self._fold_text(query)
        score = 0.0
        if "samsung" in folded_query and "samsung" in folded_text:
            score += 0.20
        if "viet nam" in folded_query and "viet nam" in folded_text:
            score += 0.15
        if self._has_investment_signal(folded_text):
            score += 0.55
        if self._has_number_signal(text) and self._has_investment_signal(folded_text):
            score += 0.15
        if self._has_indirect_investment_signal(folded_text) and not self._has_investment_signal(folded_text):
            score -= 0.20
        return min(max(score, 0.0), 1.0)

    def _is_investment_query(self, query: str) -> bool:
        folded = self._fold_text(query)
        return self._has_investment_signal(folded)

    @staticmethod
    def _has_investment_signal(folded_text: str) -> bool:
        return any(re.search(pattern, folded_text or "") for pattern in INVESTMENT_SIGNAL_PATTERNS)

    @staticmethod
    def _has_indirect_investment_signal(folded_text: str) -> bool:
        return any(re.search(pattern, folded_text or "") for pattern in INDIRECT_INVESTMENT_PATTERNS)

    def _short_evidence_sentence(
        self,
        text: str,
        query: str = "",
        title: str = "",
    ) -> str:
        sentences = self._split_sentences(re.sub(r"\[[Ss]\d+\]", "", text or ""))
        if sentences:
            query_terms = set(self._content_terms(query))
            title_norm = self._normalize_for_compare(title)
            candidates = []
            for idx, sentence in enumerate(sentences):
                sentence_norm = self._normalize_for_compare(sentence)
                overlap = self._overlap_ratio(query_terms, set(self._content_terms(sentence)))
                number_bonus = 0.20 if self._has_number_signal(sentence) else 0.0
                title_penalty = 0.35 if title_norm and sentence_norm == title_norm else 0.0
                candidates.append((overlap + number_bonus - title_penalty, -idx, sentence))
            candidates.sort(reverse=True)
            sentence = candidates[0][2]
        else:
            sentence = text or ""
        sentence = re.sub(r"\s+", " ", sentence).strip(" .")
        if len(sentence) > 240:
            sentence = sentence[:237].rstrip(" ,;:") + "..."
        return sentence

    def _investment_scope_text(self, query: str) -> str:
        folded = self._fold_text(query)
        if "samsung" in folded and "viet nam" in folded:
            return "của Samsung tại Việt Nam"
        if "samsung" in folded:
            return "của Samsung"
        if "viet nam" in folded:
            return "tại Việt Nam"
        return "được hỏi trong truy vấn"

    def _is_who_is_query(self, query: str) -> bool:
        folded = self._fold_text(query)
        if not re.search(r"\bai\b", folded):
            return False
        role_terms = [
            "tong thong",
            "thu tuong",
            "chu tich",
            "bo truong",
            "giam doc",
            "ceo",
            "lanh dao",
        ]
        return any(role in folded for role in role_terms)

    def _answer_who_is(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        query_norm = self._fold_text(query)
        role_label = self._query_role_label(query)
        candidates = []

        for ctx in contexts:
            title = ctx.get("title", "")
            person = self._extract_person_from_title(title)
            if not person:
                continue

            title_norm = self._fold_text(title)
            context_norm = self._fold_text(f"{title} {ctx.get('selected_text', '')} {ctx.get('text', '')}")
            role_bonus = 0.40 if self._role_matches_query(query_norm, title_norm) else 0.0
            geo_bonus = 0.20 if self._geo_matches_query(query_norm, context_norm) else 0.0
            clue_bonus = 0.45 if self._has_who_is_title_clue(title_norm) else 0.20
            stale_penalty = 0.35 if any(bad in title_norm for bad in ["xep hang", "lich su", "cuu tong thong"]) else 0.0
            score = float(ctx.get("context_score", 0.0) or 0.0) + role_bonus + geo_bonus + clue_bonus - stale_penalty
            candidates.append((score, person, ctx))

        if not candidates:
            return ""

        candidates.sort(key=lambda item: item[0], reverse=True)
        score, person, ctx = candidates[0]
        if score < 0.85:
            return ""

        role_text = self._lower_first(role_label)
        source_id = ctx["source_id"]
        title = ctx.get("title", "")
        evidence_phrase = self._who_is_evidence_phrase(self._fold_text(title))
        if evidence_phrase:
            return f"Theo nguồn tìm được, {person} {evidence_phrase} {role_text}; câu trả lời ngắn là {person} [{source_id}]."
        return f"Theo nguồn tìm được, {role_text} là {person} [{source_id}]."

    def _query_role_label(self, query: str) -> str:
        folded = self._fold_text(query)
        has_us = "hoa ky" in folded or re.search(r"\bmy\b", folded)
        if "tong thong" in folded:
            return "Tổng thống Mỹ" if has_us else "Tổng thống"
        if "thu tuong" in folded:
            return "Thủ tướng"
        if "chu tich" in folded:
            return "Chủ tịch"
        if "bo truong" in folded:
            return "Bộ trưởng"
        if "ceo" in folded:
            return "CEO"
        if "giam doc" in folded:
            return "Giám đốc"
        return "Người được hỏi"

    def _role_matches_query(self, query_norm: str, text_norm: str) -> bool:
        role_pairs = [
            "tong thong",
            "thu tuong",
            "chu tich",
            "bo truong",
            "ceo",
            "giam doc",
            "lanh dao",
        ]
        return any(role in query_norm and role in text_norm for role in role_pairs)

    @staticmethod
    def _geo_matches_query(query_norm: str, text_norm: str) -> bool:
        asks_us = "hoa ky" in query_norm or re.search(r"\bmy\b", query_norm)
        if asks_us:
            return "hoa ky" in text_norm or re.search(r"\bmy\b", text_norm) is not None
        return True

    @staticmethod
    def _has_who_is_title_clue(title_norm: str) -> bool:
        clues = [
            "nham chuc",
            "dac cu",
            "tuyen the",
            "tro thanh",
            "duoc bau",
            "la tan",
            "la tong thong",
            "la thu tuong",
            "la chu tich",
        ]
        return any(clue in title_norm for clue in clues)

    @staticmethod
    def _who_is_evidence_phrase(title_norm: str) -> str:
        if "nham chuc" in title_norm:
            return "nhậm chức"
        if "dac cu" in title_norm or "duoc bau" in title_norm:
            return "được bầu làm"
        if "tuyen the" in title_norm:
            return "tuyên thệ làm"
        if "tro thanh" in title_norm:
            return "trở thành"
        if "la tan" in title_norm:
            return "được nêu là"
        return ""

    def _extract_person_from_title(self, title: str) -> str:
        title = re.sub(r"\s+", " ", title or "").strip()
        if not title:
            return ""

        clue_idx = self._first_title_clue_index(title)
        if clue_idx > 0:
            person = self._extract_name_from_prefix(title[:clue_idx])
            if person:
                return person

        role_then_name = re.search(
            r"(?:Tổng thống|Thủ tướng|Chủ tịch|Bộ trưởng|CEO|Giám đốc)"
            r"(?:\s+(?:Mỹ|Hoa Kỳ|Việt Nam|Nga|Trung Quốc|Pháp|Đức|Anh|Nhật Bản))?"
            r"\s+([A-ZÀ-ỸĐ][A-Za-zÀ-ỹĐđ'.-]*(?:\s+[A-ZÀ-ỸĐ][A-Za-zÀ-ỹĐđ'.-]*){0,3})",
            title,
        )
        if role_then_name:
            return self._clean_person_name(role_then_name.group(1))
        return ""

    def _first_title_clue_index(self, title: str) -> int:
        folded = self._fold_text(title)
        clue_positions = []
        for clue in ["nham chuc", "dac cu", "tuyen the", "tro thanh", "duoc bau", "la tan"]:
            idx = folded.find(clue)
            if idx > 0:
                clue_positions.append(idx)
        return min(clue_positions) if clue_positions else -1

    def _extract_name_from_prefix(self, prefix: str) -> str:
        honorific_match = re.match(r"\s*(ông|bà|ong|ba)\s+", prefix, flags=re.IGNORECASE)
        honorific = honorific_match.group(1).lower() if honorific_match else ""
        tokens = re.findall(r"[A-ZÀ-ỸĐ][A-Za-zÀ-ỹĐđ'.-]*", prefix or "")
        ignored = {
            "ong",
            "ba",
            "tan",
            "cuu",
            "tong",
            "thong",
            "thu",
            "tuong",
            "chu",
            "tich",
            "bo",
            "truong",
            "my",
            "hoa",
            "ky",
            "viet",
            "nam",
        }
        kept = [token for token in tokens if self._fold_text(token) not in ignored]
        if not kept:
            return ""
        name = self._clean_person_name(" ".join(kept[-3:]))
        if honorific in {"ông", "ong"}:
            return f"ông {name}"
        if honorific in {"bà", "ba"}:
            return f"bà {name}"
        return name

    @staticmethod
    def _clean_person_name(name: str) -> str:
        return re.sub(r"\s+", " ", (name or "").strip(" .,:;\"'"))

    @staticmethod
    def _lower_first(text: str) -> str:
        if not text:
            return text
        return text[:1].lower() + text[1:]

    def _clean_invalid_citations(self, answer: str, contexts: List[Dict[str, Any]]) -> str:
        valid_source_ids = {ctx["source_id"] for ctx in contexts}

        def replace(match: re.Match) -> str:
            source_id = match.group(1)
            return match.group(0) if source_id in valid_source_ids else ""

        text = re.sub(r"\[(S\d+)\]", replace, answer or "")
        return re.sub(r"\s+", " ", text).strip()

    def _citations_from_answer(
        self,
        answer: str,
        contexts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        cited_ids = set(re.findall(r"\[(S\d+)\]", answer or ""))
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

    def _estimate_confidence(
        self,
        answer: str,
        citations: List[Dict[str, Any]],
        contexts: List[Dict[str, Any]],
        force_low_confidence: bool,
    ) -> str:
        if not answer or not citations:
            return "low"
        max_score = max((ctx.get("context_score", 0.0) for ctx in contexts), default=0.0)
        if force_low_confidence:
            return "low"
        if len(citations) >= 2 and max_score >= 0.70:
            return "high"
        if max_score >= 0.45:
            return "medium"
        return "low"

    def _empty_answer(self, reason: str) -> Dict[str, Any]:
        return {
            "answer": NO_EVIDENCE_ANSWER,
            "citations": [],
            "used_contexts": [],
            "selected_sentences": [],
            "confidence": "low",
            "is_answerable": False,
            "provider": "vit5-local",
            "model": self.model_name,
            "error": reason,
            "reader_mode": "none",
        }

    def _unreliable_answer(
        self,
        reason: str,
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "answer": NO_EVIDENCE_ANSWER,
            "citations": [],
            "used_contexts": contexts or [],
            "selected_sentences": [
                {
                    "source_id": ctx["source_id"],
                    "text": ctx.get("selected_text", ""),
                    "context_score": ctx.get("context_score", 0.0),
                }
                for ctx in (contexts or [])
            ],
            "confidence": "low",
            "is_answerable": False,
            "provider": "vit5-local",
            "model": self.model_name,
            "error": reason,
            "reader_mode": "unreliable",
        }

    def _context_text(self, doc: Dict[str, Any]) -> str:
        title = re.sub(r"\s+", " ", str(doc.get("title") or "")).strip()
        text = (
            doc.get("chunk_text")
            or doc.get("snippet")
            or doc.get("content")
            or doc.get("full_text")
            or ""
        )
        text = re.sub(r"\s+", " ", str(text)).strip()
        if title and title.lower() not in text[: max(40, len(title) + 20)].lower():
            text = f"{title}. {text}" if text else title
        return text[: self.max_context_chars]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?…])\s+", text or "")
        return [s.strip() for s in sentences if len(s.strip()) >= 20]

    @staticmethod
    def _tokenize_terms(text: str) -> List[str]:
        return [term.lower() for term in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", text or "")]

    @classmethod
    def _content_terms(cls, text: str) -> List[str]:
        return [
            term
            for term in cls._tokenize_terms(text)
            if len(term) >= 2 and term not in STOPWORDS
        ]

    @classmethod
    def _normalize_for_compare(cls, text: str) -> str:
        terms = cls._content_terms(text)
        return " ".join(terms)

    @staticmethod
    def _fold_text(text: str) -> str:
        normalized = unicodedata.normalize("NFD", text or "")
        without_marks = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
        without_marks = without_marks.replace("Đ", "D").replace("đ", "d")
        return re.sub(r"\s+", " ", without_marks.lower()).strip()

    @staticmethod
    def _entity_terms(query_terms: List[str]) -> List[str]:
        return [term for term in query_terms if len(term) >= 4][:6]

    @staticmethod
    def _overlap_ratio(source_terms: List[str] | set, target_terms: List[str] | set) -> float:
        source = set(source_terms)
        target = set(target_terms)
        if not source or not target:
            return 0.0
        return len(source & target) / max(1, len(source))

    @staticmethod
    def _has_number_signal(text: str) -> bool:
        return bool(
            re.search(
                r"\d|tỷ|triệu|usd|vnd|đồng|%|năm|tháng",
                text or "",
                flags=re.IGNORECASE,
            )
        )
