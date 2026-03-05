"""
Sector-aware orderbook framework.
Based on: memory/ORDERBOOK_FRAMEWORK.md
"""

import yfinance as yf
from loguru import logger

# ── Sector classification for watchlist ──────────────────────────
SECTOR_MAP = {
    # Orderbook FULLY applicable
    "LT":          "epc_infra",
    "TATAMOTORS":  "auto_mfg",

    # Partial / Segment rule
    "ONGC":        "psu_energy",
    "RELIANCE":    "conglomerate",
    "ITC":         "conglomerate",
    "MARUTI":      "auto_consumer",

    # IT Services — use TCV / deal win rate
    "TCS":         "it_services",
    "INFY":        "it_services",
    "WIPRO":       "it_services",

    # Banking / NBFC — use NIM, NPA, credit growth
    "HDFCBANK":    "banking",
    "ICICIBANK":   "banking",
    "SBIN":        "banking",
    "AXISBANK":    "banking",
    "KOTAKBANK":   "banking",
    "BAJFINANCE":  "nbfc",

    # Others — wrong tool
    "HINDUNILVR":  "fmcg",
    "NESTLEIND":   "fmcg",
    "BHARTIARTL":  "telecom",
    "TITAN":       "consumer_retail",
    "SUNPHARMA":   "pharma",
}

SECTOR_LABELS = {
    "epc_infra":      "EPC / Infrastructure",
    "auto_mfg":       "Auto / Manufacturing",
    "psu_energy":     "PSU Energy",
    "conglomerate":   "Conglomerate",
    "auto_consumer":  "Auto (Consumer)",
    "it_services":    "IT Services",
    "banking":        "Banking",
    "nbfc":           "NBFC",
    "fmcg":           "FMCG",
    "telecom":        "Telecom",
    "consumer_retail":"Consumer / Retail",
    "pharma":         "Pharma",
}

# Sectors where orderbook framework is meaningful
ORDERBOOK_SECTORS = {"epc_infra", "auto_mfg"}
PARTIAL_SECTORS   = {"psu_energy", "conglomerate", "auto_consumer"}

# What metric to use instead
ALTERNATIVE_METRICS = {
    "it_services":     "TCV (Total Contract Value) trend, deal win rate, utilisation",
    "banking":         "NIM, NPA ratio, credit growth, CASA ratio",
    "nbfc":            "AUM growth, NPA, credit cost, ROE",
    "fmcg":            "Volume growth, gross margin, distribution reach",
    "telecom":         "ARPU growth, 5G subscriber adds, enterprise revenue",
    "consumer_retail": "Same-store sales growth, wedding demand cycle, gold sensitivity",
    "pharma":          "ANDA pipeline, US generics revenue, R&D spend, specialty growth",
    "psu_energy":      "Crude price sensitivity, capex execution, dividend yield",
    "conglomerate":    "Segment-level analysis — apply framework to each division separately",
    "auto_consumer":   "Booking-to-delivery ratio, model mix, export momentum",
}


def get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol.upper(), "unknown")


def fetch_fundamentals(symbol: str) -> dict:
    """Fetch live MCap + Revenue from yfinance."""
    try:
        info = yf.Ticker(f"{symbol}.NS").info
        rev    = round((info.get("totalRevenue",    0) or 0) / 1e7, 0)
        mcap   = round((info.get("marketCap",       0) or 0) / 1e7, 0)
        ocf    = round((info.get("operatingCashflow",0) or 0) / 1e7, 0)
        ebitda = round((info.get("ebitda",          0) or 0) / 1e7, 0)
        logger.info(f"{symbol}: Rev=₹{rev:,.0f}Cr MCap=₹{mcap:,.0f}Cr")
        return {
            "annual_revenue_cr": rev,
            "market_cap_cr":     mcap,
            "ocf_cr":            ocf,
            "ebitda_cr":         ebitda,
        }
    except Exception as e:
        logger.warning(f"Fundamentals fetch failed for {symbol}: {e}")
        return {}


def compute_framework(symbol: str, orderbook_inr_cr: float, fundamentals: dict) -> dict:
    """
    Pure financial framework — Coverage Ratio + MCap Coverage.
    Returns structured result with verdict.
    """
    sector = get_sector(symbol)
    rev    = fundamentals.get("annual_revenue_cr", 0)
    mcap   = fundamentals.get("market_cap_cr", 0)
    ocf    = fundamentals.get("ocf_cr", 0)
    ebitda = fundamentals.get("ebitda_cr", 0)

    result = {
        "symbol":          symbol,
        "sector":          sector,
        "sector_label":    SECTOR_LABELS.get(sector, sector),
        "framework_applicable": sector in ORDERBOOK_SECTORS,
        "framework_partial":    sector in PARTIAL_SECTORS,
        "annual_revenue_cr":    rev,
        "market_cap_cr":        mcap,
        "orderbook_inr_cr":     orderbook_inr_cr,
    }

    # Not applicable
    if sector not in ORDERBOOK_SECTORS and sector not in PARTIAL_SECTORS:
        result["verdict"]            = "FRAMEWORK_NOT_APPLICABLE"
        result["alternative_metric"] = ALTERNATIVE_METRICS.get(sector, "N/A")
        return result

    # Formula 1 — Coverage Ratio
    if rev > 0 and orderbook_inr_cr > 0:
        cov = orderbook_inr_cr / rev
        result["coverage_ratio"] = round(cov, 2)
        if cov > 6:   cov_sig, cov_color = "INVESTIGATE", "orange"
        elif cov >= 4: cov_sig, cov_color = "STRONG", "green"
        elif cov >= 2: cov_sig, cov_color = "GOOD", "green"
        elif cov >= 1: cov_sig, cov_color = "STABLE", "orange"
        else:          cov_sig, cov_color = "WEAK", "red"
        result["coverage_signal"] = cov_sig
        result["coverage_color"]  = cov_color
    else:
        result["coverage_ratio"]  = None
        result["coverage_signal"] = "NO_DATA"
        result["coverage_color"]  = "grey"

    # Formula 2 — Market Cap Coverage %
    if mcap > 0 and orderbook_inr_cr > 0:
        mcp = (orderbook_inr_cr / mcap) * 100
        result["market_cap_pct"] = round(mcp, 1)
        if mcp > 100:  mcp_sig, mcp_color = "DEEP VALUE", "green"
        elif mcp >= 30: mcp_sig, mcp_color = "SIGNIFICANT", "green"
        elif mcp >= 10: mcp_sig, mcp_color = "MODERATE", "orange"
        else:           mcp_sig, mcp_color = "NOISE", "red"
        result["market_signal"] = mcp_sig
        result["market_color"]  = mcp_color
    else:
        result["market_cap_pct"] = None
        result["market_signal"]  = "NO_DATA"
        result["market_color"]   = "grey"

    # Cash conversion ratio
    if ocf > 0 and ebitda > 0:
        ccr = ocf / ebitda
        result["cash_conversion_ratio"] = round(ccr, 2)
        result["cash_signal"] = "REAL" if ccr >= 0.8 else "MODERATE" if ccr >= 0.5 else "ACCOUNTING_ONLY"
    else:
        result["cash_conversion_ratio"] = None
        result["cash_signal"] = "NO_DATA"

    # Final verdict
    cov_ok = result.get("coverage_signal") in ("GOOD", "STRONG", "INVESTIGATE")
    mcp_ok = result.get("market_signal")   in ("SIGNIFICANT", "DEEP VALUE")

    if cov_ok and mcp_ok:
        result["verdict"] = "RE_RATING_CANDIDATE"
    elif mcp_ok:
        result["verdict"] = "WATCH_FOR_RE_RATING"
    elif cov_ok:
        result["verdict"] = "HEALTHY_PIPELINE"
    else:
        result["verdict"] = "NOISE_OR_WEAK"

    # Segment rule note for conglomerates
    if sector == "conglomerate":
        result["segment_note"] = (
            "Apply framework to SEGMENT revenue, not total. "
            "Find supplier ecosystem for the relevant segment."
        )

    return result
