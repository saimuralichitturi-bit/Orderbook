"""
NSE Filing Chat Agent — Multi-provider, batch processing, memory-backed.

Pipeline per question:
  1. Recall relevant past Q&A from Supermemory + Mem0
  2. AI relevance-filter all filings to the most relevant ones for the question
  3. Batch-process selected filings (Groq — fast extractor)
  4. Synthesize final answer (DeepSeek / Gemini / OpenRouter)
  5. Store Q&A in Supermemory + Mem0 for future recall
"""

import json
import re
import requests
from typing import Callable
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_API_URL,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_API_URL,
    GEMINI_API_KEY, GEMINI_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_API_URL,
    SUPERMEMORY_API_KEY, MEM0_API_KEY,
    ANALYSIS_DIR,
)


# ─── Supermemory ──────────────────────────────────────────────────

_SM_BASE = "https://api.supermemory.ai/v3"
_SM_HEADERS = lambda: {"Authorization": f"Bearer {SUPERMEMORY_API_KEY}", "Content-Type": "application/json"}


def _sm_add(content: str, symbol: str, tags: list[str] = None):
    if not SUPERMEMORY_API_KEY:
        return
    try:
        payload = {
            "content": content,
            "metadata": {"symbol": symbol, "tags": tags or []},
        }
        r = requests.post(f"{_SM_BASE}/memories", headers=_SM_HEADERS(), json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        logger.warning(f"Supermemory add failed: {e}")


def _sm_search(query: str, symbol: str, limit: int = 5) -> str:
    if not SUPERMEMORY_API_KEY:
        return ""
    try:
        payload = {"query": query, "limit": limit}
        r = requests.post(f"{_SM_BASE}/memories/search", headers=_SM_HEADERS(), json=payload, timeout=10)
        r.raise_for_status()
        items = r.json().get("results", [])
        if not items:
            return ""
        parts = [f"[Memory {i+1}] {item.get('content','')}" for i, item in enumerate(items)]
        return "\n".join(parts)
    except Exception as e:
        logger.warning(f"Supermemory search failed: {e}")
        return ""


# ─── Mem0 ─────────────────────────────────────────────────────────

def _get_mem0_client():
    if not MEM0_API_KEY:
        return None
    try:
        from mem0 import MemoryClient
        return MemoryClient(api_key=MEM0_API_KEY)
    except Exception:
        return None


def _mem0_add(content: str, user_id: str):
    client = _get_mem0_client()
    if not client:
        return
    try:
        client.add(content, user_id=user_id)
    except Exception as e:
        logger.warning(f"Mem0 add failed: {e}")


def _mem0_search(query: str, user_id: str, limit: int = 5) -> str:
    client = _get_mem0_client()
    if not client:
        return ""
    try:
        results = client.search(query, user_id=user_id, limit=limit)
        if not results:
            return ""
        return "\n".join(
            f"[Mem0 {i+1}] {r.get('memory', r.get('content', ''))}"
            for i, r in enumerate(results)
        )
    except Exception as e:
        logger.warning(f"Mem0 search failed: {e}")
        return ""


# ─── Helpers ──────────────────────────────────────────────────────

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _filter_relevant_filings(filings: list[dict], question: str, max_filings: int = 60) -> list[dict]:
    """
    Use Groq to score all filings by relevance to the question.
    Returns the top max_filings most relevant ones (no arbitrary cap on total).
    Processes in chunks of 80 to stay within token limits.
    """
    if len(filings) <= max_filings:
        return filings

    all_scores: list[float] = []
    chunk_size = 80

    for chunk_start in range(0, len(filings), chunk_size):
        chunk = filings[chunk_start:chunk_start + chunk_size]
        lines = []
        for i, f in enumerate(chunk):
            lines.append(
                f"[{chunk_start + i}] "
                f"Subject: {str(f.get('subject', ''))[:80]} | "
                f"Details: {str(f.get('details', ''))[:80]} | "
                f"Date: {str(f.get('broadcast_dt', ''))[:10]}"
            )

        prompt = (
            f"Question: {question}\n\n"
            f"Rate each filing's relevance to answer this question. Score 0–10 (0=irrelevant, 10=directly answers).\n"
            f"Return ONLY a JSON array of numbers, one per filing, in order.\n"
            f"Example for 5 filings: [2, 9, 0, 7, 4]\n\n"
            f"Filings:\n" + "\n".join(lines)
        )

        try:
            resp = _call_groq(
                "You rate NSE filing relevance. Return only a JSON array of integers.",
                prompt,
                max_tokens=500,
            )
            match = re.search(r'\[[\d.,\s]+\]', resp)
            if match:
                scores = json.loads(match.group())
                all_scores.extend(float(s) for s in scores[:len(chunk)])
            else:
                all_scores.extend([5.0] * len(chunk))
        except Exception:
            all_scores.extend([5.0] * len(chunk))

    # Pad if needed
    if len(all_scores) < len(filings):
        all_scores.extend([5.0] * (len(filings) - len(all_scores)))

    # Sort by score descending, keep top max_filings
    scored = sorted(zip(all_scores, range(len(filings))), reverse=True)
    top_indices = sorted(idx for _, idx in scored[:max_filings])
    return [filings[i] for i in top_indices]


def _load_analysis_json(filing_id: str) -> dict:
    path = ANALYSIS_DIR / f"{filing_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _build_filing_context(filing: dict, analysis: dict) -> str:
    """Build a compact text block for one filing."""
    lines = [
        f"Filing: {filing.get('subject', '?')}",
        f"Date: {str(filing.get('broadcast_dt', '?'))[:10]}",
        f"Details: {str(filing.get('details', ''))[:300]}",
    ]
    if analysis:
        lines += [
            f"AI Summary: {analysis.get('summary', '')}",
            f"Sentiment: {analysis.get('sentiment', '')} ({analysis.get('sentiment_score', '')})",
            f"Signal: {analysis.get('action_signal', '')}",
            f"Highlights: {'; '.join(analysis.get('key_highlights', []))}",
            f"Risks: {'; '.join(analysis.get('risk_factors', []))}",
        ]
        fin = analysis.get("financial_data", {})
        if fin and any(fin.get(k) for k in ["revenue", "net_profit", "eps"]):
            lines.append(
                f"Financials: Revenue={fin.get('revenue')} Cr, "
                f"Net Profit={fin.get('net_profit')} Cr, EPS={fin.get('eps')}"
            )
    return "\n".join(lines)


# ─── LLM callers ──────────────────────────────────────────────────

EXTRACTOR_SYSTEM = (
    "You are an expert at extracting relevant information from NSE corporate filings. "
    "Given a set of filings and a question, extract ONLY the information relevant to the question. "
    "Be concise. Return plain text."
)

SYNTHESIZER_SYSTEM = (
    "You are a senior equity analyst at an Indian hedge fund. "
    "Using the provided filing extracts and past context, give a thorough, structured answer. "
    "Include: direct answer, supporting evidence from filings, risks, recommendation. "
    "Use ₹ for amounts and mention specific filings where relevant."
)


def _call_groq(system: str, user: str, max_tokens: int = 1500) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_URL)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _call_deepseek(system: str, user: str, max_tokens: int = 2048) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _call_gemini(system: str, user: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{system}\n\n{user}",
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return resp.text.strip()


def _call_openrouter(system: str, user: str, max_tokens: int = 2048) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_API_URL,
        default_headers={"HTTP-Referer": "https://github.com/nse-filings-pipeline"},
    )
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _synthesize(system: str, user: str) -> tuple[str, str]:
    """Try DeepSeek → Gemini → OpenRouter for synthesis. Returns (answer, model_name)."""
    if DEEPSEEK_API_KEY:
        try:
            return _call_deepseek(system, user), "DeepSeek Chat"
        except Exception as e:
            logger.warning(f"DeepSeek synth failed: {e}")
    if GEMINI_API_KEY:
        try:
            return _call_gemini(system, user), "Gemini 2.0 Flash"
        except Exception as e:
            logger.warning(f"Gemini synth failed: {e}")
    if OPENROUTER_API_KEY:
        try:
            return _call_openrouter(system, user), OPENROUTER_MODEL
        except Exception as e:
            logger.warning(f"OpenRouter synth failed: {e}")
    return "All AI providers failed. Please check API keys.", "none"


# ─── Main Chat Agent ──────────────────────────────────────────────

class NSEChatAgent:
    """
    Multi-agent filing chat with Supermemory + Mem0 persistence.
    """

    def __init__(self, symbol: str):
        self.symbol  = symbol
        self.user_id = f"nse_{symbol.lower()}"

    def answer(
        self,
        question: str,
        filings: list[dict],
        progress: Callable[[int, int, str], None] | None = None,
        batch_size: int = 5,
    ) -> dict:
        """
        Process the question against the provided filings.

        Args:
            question:  User's question
            filings:   List of filing dicts (rows from parquet)
            progress:  Callback(current, total, message) for Streamlit progress
            batch_size: Filings per extractor batch

        Returns:
            dict with keys: answer, model, sources, memory_hits
        """
        def _prog(cur, tot, msg):
            if progress:
                progress(cur, tot, msg)

        # ── Step 0: AI relevance filtering ────────────────────────
        # No hard cap — use Groq to pick most relevant filings for the question
        if len(filings) > 60:
            _prog(0, len(filings) + 3, f"Filtering {len(filings)} filings by relevance to your question...")
            filings = _filter_relevant_filings(filings, question, max_filings=60)

        total_steps = len(filings) + 2  # filings + memory recall + synthesis

        # ── Step 1: Recall memory ──────────────────────────────────
        _prog(0, total_steps, "Recalling past context from memory...")
        sm_context  = _sm_search(question, self.symbol)
        m0_context  = _mem0_search(question, self.user_id)
        memory_hits = bool(sm_context or m0_context)
        past_ctx    = "\n\n".join(filter(None, [sm_context, m0_context]))

        # ── Step 2: Batch-extract relevant info from each filing ───
        extracts   = []
        batch_num  = 0
        all_batches = list(_chunks(filings, batch_size))

        for batch in all_batches:
            batch_num += 1
            start_idx = (batch_num - 1) * batch_size
            end_idx   = min(start_idx + batch_size, len(filings))
            _prog(start_idx, total_steps, f"Extracting from filings {start_idx+1}–{end_idx} of {len(filings)}...")

            # Build context block for this batch
            filing_blocks = []
            for i, f in enumerate(batch):
                analysis = _load_analysis_json(str(f.get("filing_id", "")))
                block    = _build_filing_context(f, analysis)
                filing_blocks.append(f"--- Filing {start_idx + i + 1} ---\n{block}")

            batch_text = "\n\n".join(filing_blocks)
            user_prompt = (
                f"Company: {self.symbol}\n"
                f"Question: {question}\n\n"
                f"Filings:\n{batch_text}\n\n"
                f"Extract all information relevant to the question. Be concise."
            )

            try:
                extract = _call_groq(EXTRACTOR_SYSTEM, user_prompt, max_tokens=800)
                extracts.append(extract)
            except Exception as e:
                logger.warning(f"Extractor batch {batch_num} failed: {e}")
                extracts.append(f"[Batch {batch_num} extraction failed: {e}]")

        # ── Step 3: Synthesize final answer ────────────────────────
        _prog(len(filings), total_steps, "Synthesizing final answer...")

        combined_extracts = "\n\n".join(
            f"=== Extract {i+1} ===\n{e}" for i, e in enumerate(extracts)
        )

        synth_user = (
            f"Company: {self.symbol}\n"
            f"Analyst Question: {question}\n\n"
            + (f"Relevant Past Context:\n{past_ctx}\n\n" if past_ctx else "")
            + f"Filing Extracts ({len(filings)} filings analyzed):\n{combined_extracts}"
        )

        answer_text, model_used = _synthesize(SYNTHESIZER_SYSTEM, synth_user)

        # ── Step 4: Store in memory ────────────────────────────────
        _prog(total_steps - 1, total_steps, "Saving to memory...")
        memory_entry = (
            f"Q: {question}\n"
            f"A: {answer_text[:1000]}\n"
            f"Symbol: {self.symbol}"
        )
        _sm_add(memory_entry, self.symbol, tags=["qa", self.symbol])
        _mem0_add(memory_entry, self.user_id)

        _prog(total_steps, total_steps, "Done")

        # Build source list
        sources = [
            {
                "subject": f.get("subject", "?"),
                "date":    str(f.get("broadcast_dt", "?"))[:10],
                "pdf_url": f.get("pdf_url", ""),
            }
            for f in filings[:10]  # show first 10 as sources
        ]

        return {
            "answer":      answer_text,
            "model":       model_used,
            "sources":     sources,
            "memory_hits": memory_hits,
            "filings_used": len(filings),
        }
