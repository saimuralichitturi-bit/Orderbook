"""
PDF Text Pre-Processor — strips NSE boilerplate, keeps financial signals.
"""
import re
from loguru import logger

# Full phrases to remove (case-insensitive, full line match)
_LINE_REMOVE = re.compile(
    r"(?i)^("
    r"national stock exchange.*|bse limited.*|bombay stock exchange.*|"
    r"listing compliance.*|exchange plaza.*|dalal street.*|bandra.kurla.*|"
    r"phiroze jeejeebhoy.*|"
    r"pursuant to regulation.*|pursuant to sebi.*|"
    r"kindly (?:take|bring|acknowledge).*|please (?:take|find|note|acknowledge).*|"
    r"thanking you.*|yours (?:faithfully|sincerely|truly).*|"
    r"for and on behalf.*|authoris[e|z]d signatory.*|company secretary.*|"
    r"(?:sub|subject)\s*:?\s*(?:outcome of board|board meeting intimation).*|"
    r"sebi \(listing obligations.*|"
    r"page \d+ of \d+|^\d+$"
    r")$",
    re.MULTILINE
)

_MULTI_BLANK = re.compile(r"\n{2,}")
_INLINE_JUNK = re.compile(
    r"(?i)(national stock exchange|bombay stock exchange|"
    r"bandra.kurla complex|dalal street|phiroze jeejeebhoy|"
    r"listing compliance department|exchange plaza)"
)


def clean_pdf_text(raw_text: str, max_chars: int = 12000) -> str:
    """Clean NSE filing text — remove boilerplate, deduplicate, keep signals."""
    if not raw_text:
        return ""

    # Remove full boilerplate lines
    text = _LINE_REMOVE.sub("", raw_text)

    # Remove inline junk phrases
    text = _INLINE_JUNK.sub("", text)

    # Deduplicate consecutive identical lines
    lines = text.splitlines()
    deduped = []
    seen = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            deduped.append("")
            continue
        if stripped in seen:
            continue
        seen.add(stripped)
        deduped.append(line)

    text = "\n".join(deduped)

    # Collapse blank lines
    text = _MULTI_BLANK.sub("\n\n", text).strip()

    reduction = (1 - len(text) / max(len(raw_text), 1)) * 100
    logger.debug(f"Cleaner: {len(raw_text):,} → {len(text):,} chars ({reduction:.0f}% reduction)")

    return text[:max_chars]
