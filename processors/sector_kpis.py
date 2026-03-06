"""
Sector-specific KPI engine.
Pulls real data from yfinance and shows the RIGHT metrics per sector.
No orderbook framework for non-applicable sectors — correct tool, correct signal.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import yfinance as yf
from loguru import logger


def fetch_sector_kpis(symbol: str, sector: str) -> dict:
    """Fetch yfinance data and return sector-appropriate KPIs."""
    try:
        info = yf.Ticker(f"{symbol}.NS").info
    except Exception as e:
        logger.warning(f"yfinance failed for {symbol}: {e}")
        return {}

    def pct(v):  return f"{v*100:.1f}%" if v else "N/A"
    def cr(v):   return f"₹{v/1e7:,.0f} Cr" if v else "N/A"
    def x(v):    return f"{v:.2f}x" if v else "N/A"
    def raw(v):  return f"{v:.2f}" if v else "N/A"

    base = {
        "price":         info.get("currentPrice"),
        "mcap_cr":       round((info.get("marketCap", 0) or 0) / 1e7, 0),
        "pe_ttm":        info.get("trailingPE"),
        "pe_fwd":        info.get("forwardPE"),
        "52w_high":      info.get("fiftyTwoWeekHigh"),
        "52w_low":       info.get("fiftyTwoWeekLow"),
        "52w_change":    info.get("52WeekChange"),
        "analyst_target":info.get("targetMeanPrice"),
        "analyst_rating":info.get("recommendationKey", "").replace("_", " ").title(),
        "dividend_yield":info.get("dividendYield"),
    }

    if sector == "banking":
        base.update({
            "kpi_label": "Banking KPIs",
            "kpis": [
                ("ROE",             pct(info.get("returnOnEquity")),
                 "Return on Equity. >15% = good bank. <10% = weak.",
                 _signal(info.get("returnOnEquity"), 0.15, 0.10)),

                ("ROA",             pct(info.get("returnOnAssets")),
                 "Return on Assets. >1.5% = efficient. <0.8% = concern.",
                 _signal(info.get("returnOnAssets"), 0.015, 0.008)),

                ("P/B Ratio",       x(info.get("priceToBook")),
                 "Price-to-Book. High P/B = premium franchise. Check vs ROE.",
                 "neutral"),

                ("Earnings Growth", pct(info.get("earningsGrowth")),
                 "YoY earnings growth.",
                 _signal(info.get("earningsGrowth"), 0.12, 0.05)),

                ("Revenue Growth",  pct(info.get("revenueGrowth")),
                 "Net interest income + fee income growth YoY.",
                 _signal(info.get("revenueGrowth"), 0.15, 0.08)),

                ("Profit Margin",   pct(info.get("profitMargins")),
                 "Net profit margin. Banks: >20% is strong.",
                 _signal(info.get("profitMargins"), 0.20, 0.12)),

                ("Dividend Yield",  pct(info.get("dividendYield")),
                 "Dividend yield.",
                 "neutral"),
            ],
            "note": "NIM and NPA not in yfinance — check quarterly results for those numbers.",
            "watch": ["NIM (Net Interest Margin) — target >4%",
                      "GNPA ratio — target <1.5%",
                      "CASA ratio — target >40%",
                      "Credit growth YoY — target >15%"],
        })

    elif sector == "nbfc":
        base.update({
            "kpi_label": "NBFC KPIs",
            "kpis": [
                ("ROE",             pct(info.get("returnOnEquity")),
                 "ROE. NBFC should be >18% to justify premium.",
                 _signal(info.get("returnOnEquity"), 0.18, 0.12)),

                ("P/B Ratio",       x(info.get("priceToBook")),
                 "NBFCs trade at high P/B if growth + quality is strong.",
                 "neutral"),

                ("Earnings Growth", pct(info.get("earningsGrowth")),
                 "AUM growth drives earnings.",
                 _signal(info.get("earningsGrowth"), 0.20, 0.10)),

                ("Revenue Growth",  pct(info.get("revenueGrowth")),
                 "Interest income growth.",
                 _signal(info.get("revenueGrowth"), 0.20, 0.10)),

                ("Profit Margin",   pct(info.get("profitMargins")),
                 "Net margin.",
                 _signal(info.get("profitMargins"), 0.22, 0.15)),
            ],
            "note": "AUM size and NPA not in yfinance — check quarterly results.",
            "watch": ["AUM growth >25% YoY",
                      "GNPA <2%",
                      "Credit cost <1.5%",
                      "Tier-1 CAR >15%"],
        })

    elif sector == "it_services":
        base.update({
            "kpi_label": "IT Services KPIs",
            "kpis": [
                ("EBITDA Margin",   pct(info.get("ebitdaMargins")),
                 "IT margins: >25% = strong, <20% = pressure.",
                 _signal(info.get("ebitdaMargins"), 0.25, 0.20)),

                ("Revenue Growth",  pct(info.get("revenueGrowth")),
                 "Organic CC revenue growth. >10% = good cycle.",
                 _signal(info.get("revenueGrowth"), 0.10, 0.05)),

                ("Earnings Growth", pct(info.get("earningsGrowth")),
                 "EPS growth.",
                 _signal(info.get("earningsGrowth"), 0.10, 0.05)),

                ("ROE",             pct(info.get("returnOnEquity")),
                 "ROE. IT should be >25% (asset-light).",
                 _signal(info.get("returnOnEquity"), 0.25, 0.18)),

                ("P/E (TTM)",       raw(info.get("trailingPE")),
                 "Valuation. IT at >30x needs strong growth.",
                 "neutral"),

                ("Dividend Yield",  pct(info.get("dividendYield")),
                 "IT companies return cash via dividends + buybacks.",
                 "neutral"),
            ],
            "note": "TCV (Total Contract Value) and deal wins not in yfinance — check quarterly earnings.",
            "watch": ["Large deal TCV (>$500Mn/quarter = healthy)",
                      "Attrition rate (target <15%)",
                      "Utilisation rate (target >82%)",
                      "US/Europe revenue mix (macro sensitivity)"],
        })

    elif sector == "fmcg":
        base.update({
            "kpi_label": "FMCG KPIs",
            "kpis": [
                ("EBITDA Margin",   pct(info.get("ebitdaMargins")),
                 "FMCG margins: >20% = pricing power. <15% = commodity squeeze.",
                 _signal(info.get("ebitdaMargins"), 0.20, 0.15)),

                ("Revenue Growth",  pct(info.get("revenueGrowth")),
                 "Volume + price growth. Ideally volume-led.",
                 _signal(info.get("revenueGrowth"), 0.08, 0.04)),

                ("Earnings Growth", pct(info.get("earningsGrowth")),
                 "PAT growth.",
                 _signal(info.get("earningsGrowth"), 0.10, 0.05)),

                ("ROE",             pct(info.get("returnOnEquity")),
                 "FMCG ROE should be >30% (asset-light brand model).",
                 _signal(info.get("returnOnEquity"), 0.30, 0.20)),

                ("P/E (TTM)",       raw(info.get("trailingPE")),
                 "FMCG commands premium PE. Justify with growth.",
                 "neutral"),

                ("Dividend Yield",  pct(info.get("dividendYield")),
                 "Steady dividend. FMCG = bond-like income.",
                 "neutral"),
            ],
            "note": "Volume growth vs price growth split — check quarterly results.",
            "watch": ["Volume growth (separate from price)",
                      "Rural vs urban mix",
                      "Gross margin (input cost impact)",
                      "Distribution reach (# outlets)"],
        })

    elif sector == "pharma":
        base.update({
            "kpi_label": "Pharma KPIs",
            "kpis": [
                ("EBITDA Margin",   pct(info.get("ebitdaMargins")),
                 "Pharma margins: >25% = strong franchise. <18% = generic pressure.",
                 _signal(info.get("ebitdaMargins"), 0.25, 0.18)),

                ("Revenue Growth",  pct(info.get("revenueGrowth")),
                 "US + India formulations growth.",
                 _signal(info.get("revenueGrowth"), 0.12, 0.06)),

                ("Earnings Growth", pct(info.get("earningsGrowth")),
                 "PAT growth.",
                 _signal(info.get("earningsGrowth"), 0.15, 0.08)),

                ("ROE",             pct(info.get("returnOnEquity")),
                 "ROE. Pharma: >20% is healthy.",
                 _signal(info.get("returnOnEquity"), 0.20, 0.12)),

                ("P/E (TTM)",       raw(info.get("trailingPE")),
                 "Pharma PE. Specialty = premium, generics = discount.",
                 "neutral"),
            ],
            "note": "ANDA pipeline and US FDA status not in yfinance — check annual report + FDA website.",
            "watch": ["# pending ANDAs (US FDA)",
                      "US generics revenue % of total",
                      "Specialty revenue growth",
                      "USFDA plant observations (OAI/VAI)"],
        })

    elif sector == "telecom":
        base.update({
            "kpi_label": "Telecom KPIs",
            "kpis": [
                ("EBITDA Margin",   pct(info.get("ebitdaMargins")),
                 "Telecom margins: >40% = efficient. <35% = capex drag.",
                 _signal(info.get("ebitdaMargins"), 0.40, 0.32)),

                ("Revenue Growth",  pct(info.get("revenueGrowth")),
                 "Telecom revenue growth. ARPU uplift drives this.",
                 _signal(info.get("revenueGrowth"), 0.10, 0.04)),

                ("Earnings Growth", pct(info.get("earningsGrowth")),
                 "PAT growth (post heavy depreciation).",
                 _signal(info.get("earningsGrowth"), 0.15, 0.05)),

                ("ROE",             pct(info.get("returnOnEquity")),
                 "Telecom ROE. Capex-heavy — target >15%.",
                 _signal(info.get("returnOnEquity"), 0.15, 0.08)),

                ("P/E (TTM)",       raw(info.get("trailingPE")),
                 "Telecom PE. Justified by ARPU growth + 5G optionality.",
                 "neutral"),
            ],
            "note": "ARPU and subscriber data not in yfinance — check TRAI monthly reports.",
            "watch": ["ARPU (target >₹200 for Airtel)",
                      "5G subscriber adds",
                      "Net subscriber adds (monthly TRAI data)",
                      "Capex/Revenue ratio (target declining)"],
        })

    elif sector == "auto_consumer":
        base.update({
            "kpi_label": "Auto (Consumer) KPIs",
            "kpis": [
                ("EBITDA Margin",   pct(info.get("ebitdaMargins")),
                 "Auto margins: >12% = healthy. <8% = commodity squeeze.",
                 _signal(info.get("ebitdaMargins"), 0.12, 0.08)),

                ("Revenue Growth",  pct(info.get("revenueGrowth")),
                 "Volume + realization growth.",
                 _signal(info.get("revenueGrowth"), 0.12, 0.06)),

                ("Earnings Growth", pct(info.get("earningsGrowth")),
                 "PAT growth.",
                 _signal(info.get("earningsGrowth"), 0.15, 0.08)),

                ("ROE",             pct(info.get("returnOnEquity")),
                 "Auto ROE. >20% = capital-efficient.",
                 _signal(info.get("returnOnEquity"), 0.20, 0.12)),

                ("P/E (TTM)",       raw(info.get("trailingPE")),
                 "Auto PE. Cyclical — buy low PE in downcycle.",
                 "neutral"),
            ],
            "note": "Monthly volume data from SIAM — check for booking backlog.",
            "watch": ["Monthly wholesale volumes (SIAM data)",
                      "Booking-to-delivery ratio (pending orders)",
                      "Export volume growth",
                      "EV mix %"],
        })

    elif sector == "consumer_retail":
        base.update({
            "kpi_label": "Consumer / Retail KPIs",
            "kpis": [
                ("EBITDA Margin",   pct(info.get("ebitdaMargins")),
                 "Retail margin depends on product mix.",
                 "neutral"),

                ("Revenue Growth",  pct(info.get("revenueGrowth")),
                 "Same-store sales growth + new store additions.",
                 _signal(info.get("revenueGrowth"), 0.15, 0.08)),

                ("Earnings Growth", pct(info.get("earningsGrowth")),
                 "PAT growth.",
                 _signal(info.get("earningsGrowth"), 0.15, 0.08)),

                ("ROE",             pct(info.get("returnOnEquity")),
                 "ROE for retail / consumer.",
                 _signal(info.get("returnOnEquity"), 0.25, 0.15)),

                ("P/E (TTM)",       raw(info.get("trailingPE")),
                 "Premium valuations for strong brands.",
                 "neutral"),
            ],
            "note": "SSS (Same-Store Sales) growth is the #1 metric — not in yfinance.",
            "watch": ["Same-store sales growth (SSSG)",
                      "Store addition pace",
                      "Wedding / festive demand cycle (Titan)",
                      "Gold price impact on margins"],
        })

    elif sector == "psu_energy":
        base.update({
            "kpi_label": "PSU Energy KPIs",
            "kpis": [
                ("P/E (TTM)",       raw(info.get("trailingPE")),
                 "PSU energy PE is low. Value trap risk if crude falls.",
                 "neutral"),

                ("Dividend Yield",  pct(info.get("dividendYield")),
                 "PSU pays high dividend. Yield >5% = income play.",
                 _signal(info.get("dividendYield"), 0.05, 0.03)),

                ("Revenue Growth",  pct(info.get("revenueGrowth")),
                 "Crude price × production volume drives this.",
                 "neutral"),

                ("ROE",             pct(info.get("returnOnEquity")),
                 "PSU ROE. Often depressed by capex.",
                 "neutral"),

                ("ROA",             pct(info.get("returnOnAssets")),
                 "Asset efficiency.",
                 "neutral"),
            ],
            "note": "Crude oil price is the biggest driver — not a company-specific metric.",
            "watch": ["Brent crude price ($65–80 = sweet spot)",
                      "Production volume (MMT/MMSCMD)",
                      "Capex execution rate",
                      "Govt divestment news"],
        })

    elif sector == "conglomerate":
        base.update({
            "kpi_label": "Conglomerate — Segment Analysis Required",
            "kpis": [
                ("P/E (TTM)",       raw(info.get("trailingPE")),
                 "Conglomerate PE is blended — often hides undervalued segments.",
                 "neutral"),

                ("Revenue Growth",  pct(info.get("revenueGrowth")),
                 "Blended revenue growth across all segments.",
                 _signal(info.get("revenueGrowth"), 0.10, 0.05)),

                ("ROE",             pct(info.get("returnOnEquity")),
                 "Blended ROE.",
                 "neutral"),

                ("EBITDA Margin",   pct(info.get("ebitdaMargins")),
                 "Blended EBITDA margin.",
                 "neutral"),
            ],
            "note": "⚠️ Segment rule applies. Analyze each division independently.",
            "watch": ["Retail (Jio Mart) — GMV growth",
                      "Green Energy (Reliance) — GW target vs delivery",
                      "Telecom (Jio) — ARPU + 5G subs",
                      "Supplier ecosystem — small caps that supply to the segment"],
        })

    else:
        base.update({
            "kpi_label": "General KPIs",
            "kpis": [
                ("P/E (TTM)",       raw(info.get("trailingPE")), "", "neutral"),
                ("Revenue Growth",  pct(info.get("revenueGrowth")), "", "neutral"),
                ("Earnings Growth", pct(info.get("earningsGrowth")), "", "neutral"),
                ("EBITDA Margin",   pct(info.get("ebitdaMargins")), "", "neutral"),
                ("ROE",             pct(info.get("returnOnEquity")), "", "neutral"),
                ("Dividend Yield",  pct(info.get("dividendYield")), "", "neutral"),
            ],
            "watch": [],
            "note": "",
        })

    return base


def _signal(val, good_thresh, warn_thresh) -> str:
    if val is None:
        return "neutral"
    if val >= good_thresh:
        return "green"
    elif val >= warn_thresh:
        return "orange"
    else:
        return "red"
