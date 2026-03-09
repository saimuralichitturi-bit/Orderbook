"""
LLMLingua-2 Text Compressor
============================
Compresses clean PDF text 3x before sending to Groq/Gemini.
Downloads model from HuggingFace on first run (~500MB, cached after).

Pipeline:
  raw PDF → text_cleaner → LLMLingua-2 → Groq 70B

No HuggingFace API key needed — inference runs fully locally.
"""

from loguru import logger
from pathlib import Path

_lingua = None   # lazy-loaded singleton
_LINGUA_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
_FALLBACK_RATIO = 0.4   # target 60% compression


def _load_lingua():
    """Lazy load LLMLingua-2 — only on first call."""
    global _lingua
    if _lingua is not None:
        return _lingua
    try:
        from llmlingua import PromptCompressor
        logger.info("Loading LLMLingua-2 model (first run downloads ~500MB)...")
        _lingua = PromptCompressor(
            model_name=_LINGUA_MODEL,
            use_llmlingua2=True,
            device_map="cpu",         # CPU mode — works without GPU
        )
        logger.info("✅ LLMLingua-2 loaded")
    except ImportError:
        logger.warning("llmlingua not installed — run: pip install llmlingua")
        _lingua = None
    except Exception as e:
        logger.warning(f"LLMLingua-2 load failed: {e}")
        _lingua = None
    return _lingua


def compress_text(text: str, ratio: float = _FALLBACK_RATIO,
                  target_token: int = 2000) -> tuple[str, float]:
    """
    Compress text using LLMLingua-2.
    Returns (compressed_text, actual_compression_ratio).

    Falls back to simple truncation if LLMLingua-2 unavailable.
    """
    if not text:
        return text, 1.0

    lingua = _load_lingua()

    if lingua is None:
        # Fallback — simple smart truncation (keep first + last 30%)
        logger.debug("LLMLingua-2 unavailable — using truncation fallback")
        max_len = int(len(text) * (1 - _FALLBACK_RATIO))
        compressed = text[:max_len]
        actual_ratio = len(compressed) / max(len(text), 1)
        return compressed, actual_ratio

    try:
        result = lingua.compress_prompt(
            text,
            rate=ratio,
            target_token=target_token,
            force_tokens=["\n", ".", "!", "?"],   # preserve sentence boundaries
        )
        compressed = result.get("compressed_prompt", text)
        actual_ratio = len(compressed) / max(len(text), 1)
        logger.debug(f"LLMLingua-2: {len(text):,} → {len(compressed):,} chars ({(1-actual_ratio)*100:.0f}% compressed)")
        return compressed, actual_ratio

    except Exception as e:
        logger.warning(f"LLMLingua-2 compression failed: {e} — using original text")
        return text[:int(len(text) * ratio * 2)], ratio


def prepare_for_llm(raw_text: str, sector: str = "unknown",
                    use_compression: bool = True) -> dict:
    """
    Full pipeline: clean → compress → ready for LLM.

    Returns dict with:
      text         : final text to send to LLM
      original_len : chars before processing
      final_len    : chars after processing
      compression  : ratio achieved
      method       : 'llmlingua2' | 'truncation' | 'clean_only'
    """
    from processors.text_cleaner import clean_pdf_text

    original_len = len(raw_text)

    # Step 1 — Clean
    cleaned = clean_pdf_text(raw_text, max_chars=20000)
    after_clean = len(cleaned)

    if not use_compression or after_clean < 3000:
        # Small enough — no compression needed
        return {
            "text":         cleaned,
            "original_len": original_len,
            "final_len":    after_clean,
            "compression":  after_clean / max(original_len, 1),
            "method":       "clean_only",
        }

    # Step 2 — Compress
    compressed, ratio = compress_text(cleaned, ratio=0.4, target_token=2500)

    method = "llmlingua2" if _lingua is not None else "truncation"

    logger.info(
        f"PDF prep [{sector}]: {original_len:,} → {after_clean:,} (clean) "
        f"→ {len(compressed):,} chars ({method})"
    )

    return {
        "text":         compressed,
        "original_len": original_len,
        "final_len":    len(compressed),
        "compression":  len(compressed) / max(original_len, 1),
        "method":       method,
    }
