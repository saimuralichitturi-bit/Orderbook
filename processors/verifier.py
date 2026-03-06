"""
Cross-model verification layer.
Rule: Groq extracts → Gemini verifies. Never same model self-reviewing.
Reference: Huang et al. ICLR 2024 (LLM self-verification bias)

Verification tiers:
  confidence >= 0.8  → TRUSTED (no AI call needed)
  confidence 0.5-0.8 → VERIFY  (Gemini cross-checks value + description)
  confidence < 0.5   → FLAG    (mark as low confidence, skip from totals)
"""

import json
import sys
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL

VERIFY_THRESHOLD  = 0.8   # below this → send to Gemini
FLAG_THRESHOLD    = 0.5   # below this → auto-flag, don't include in totals

_VERIFY_PROMPT = """You are a financial fact-checker reviewing an AI-extracted orderbook entry from an NSE corporate filing.

Original filing excerpt (source of truth):
---
{context}
---

AI-extracted entry to verify:
  Description : {description}
  Value       : {value_numeric} {value_unit}
  Value (INR Cr): {value_inr_cr}
  Counterparty: {counterparty}
  Is Positive : {is_positive}

Tasks:
1. Does the extracted value ({value_numeric} {value_unit}) appear in the filing excerpt? (yes/no)
2. Is the INR Cr conversion correct? (yes/no/not_applicable)
3. Is the description accurate? (yes/no/partially)
4. Your confidence in this entry: 0.0 to 1.0

Return ONLY this JSON (no markdown):
{{
  "value_confirmed": true,
  "conversion_correct": true,
  "description_accurate": true,
  "verified_confidence": 0.92,
  "correction": null,
  "corrected_value_inr_cr": null,
  "note": "optional short note"
}}

If the value does NOT appear in the text, set verified_confidence to 0.1 and explain in note."""


def _call_gemini_verify(prompt: str) -> dict:
    """Call Gemini for verification (cross-model, not same as Groq)."""
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        raw = resp.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Gemini verify failed: {e}")
        return {}


def verify_entries(entries: list, pdf_text: str) -> list:
    """
    Verify a list of extracted entries against the source PDF text.
    Low-confidence entries get Gemini cross-check.
    Returns enriched entries with verification_status field.

    verification_status values:
      TRUSTED      — confidence >= 0.8, no check needed
      VERIFIED ✅  — Gemini confirmed
      CORRECTED 🔧 — Gemini found and fixed an error
      UNVERIFIED ⚠️ — Gemini check inconclusive
      FLAGGED ❌   — confidence < 0.5 OR Gemini rejected
    """
    if not GEMINI_API_KEY:
        # No Gemini key — mark all as unverified but keep them
        for e in entries:
            e["verification_status"] = "UNVERIFIED ⚠️"
        return entries

    verified = []
    gemini_calls = 0

    for entry in entries:
        conf = float(entry.get("confidence", 0.5) or 0.5)

        # Tier 1 — trusted, no API call
        if conf >= VERIFY_THRESHOLD:
            entry["verification_status"] = "TRUSTED"
            verified.append(entry)
            continue

        # Tier 3 — too low, flag immediately
        if conf < FLAG_THRESHOLD:
            entry["verification_status"] = "FLAGGED ❌"
            entry["include_in_totals"]   = False
            verified.append(entry)
            continue

        # Tier 2 — send to Gemini
        if gemini_calls >= 20:
            # Rate limit guard — don't burn quota
            entry["verification_status"] = "UNVERIFIED ⚠️"
            verified.append(entry)
            continue

        context_start = max(0, pdf_text.find(str(entry.get("value_numeric", ""))[:8]) - 200)
        context_snip  = pdf_text[context_start:context_start + 600] if context_start >= 0 else pdf_text[:600]

        prompt = _VERIFY_PROMPT.format(
            context       = context_snip,
            description   = entry.get("description", ""),
            value_numeric = entry.get("value_numeric", ""),
            value_unit    = entry.get("value_unit", ""),
            value_inr_cr  = entry.get("value_inr_cr", ""),
            counterparty  = entry.get("counterparty", ""),
            is_positive   = entry.get("is_positive", True),
        )

        result = _call_gemini_verify(prompt)
        gemini_calls += 1

        if not result:
            entry["verification_status"] = "UNVERIFIED ⚠️"
            verified.append(entry)
            continue

        gem_conf = float(result.get("verified_confidence", 0.5) or 0.5)
        confirmed = result.get("value_confirmed", False)

        if gem_conf >= 0.8 and confirmed:
            entry["verification_status"] = "VERIFIED ✅"
            # Apply correction if provided
            if result.get("corrected_value_inr_cr"):
                entry["value_inr_cr"]        = result["corrected_value_inr_cr"]
                entry["verification_status"] = "CORRECTED 🔧"
        elif gem_conf < 0.4 or not confirmed:
            entry["verification_status"]  = "FLAGGED ❌"
            entry["include_in_totals"]    = False
        else:
            entry["verification_status"]  = "UNVERIFIED ⚠️"

        if result.get("note"):
            entry["verification_note"] = result["note"]

        verified.append(entry)
        logger.info(f"Verified entry: {entry.get('description','')[:40]} → {entry['verification_status']} (Gemini calls: {gemini_calls})")

    trusted  = sum(1 for e in verified if e.get("verification_status") == "TRUSTED")
    verified_ = sum(1 for e in verified if "VERIFIED" in e.get("verification_status",""))
    flagged  = sum(1 for e in verified if "FLAGGED" in e.get("verification_status",""))
    logger.info(f"Verification complete: {trusted} trusted, {verified_} verified, {flagged} flagged. Gemini calls used: {gemini_calls}")

    return verified
