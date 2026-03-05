"""
NSE Corporate Filings Intelligence Dashboard
Hedge Fund grade — DeepSeek · Gemini · Groq · OpenRouter
"""

import json
import io
import re
import sys
from pathlib import Path
from datetime import date, timedelta
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import FILING_CATEGORIES, PARQUET_DIR, ANALYSIS_DIR, PDF_DIR, DATA_DIR

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INDUSTRY_MAP = {
    "RELIANCE":   "Energy",
    "TCS":        "IT",
    "HDFCBANK":   "Banking",
    "INFY":       "IT",
    "ICICIBANK":  "Banking",
    "HINDUNILVR": "FMCG",
    "ITC":        "FMCG",
    "SBIN":       "Banking",
    "BHARTIARTL": "Telecom",
    "KOTAKBANK":  "Banking",
    "LT":         "Infrastructure",
    "AXISBANK":   "Banking",
    "BAJFINANCE": "Financial Services",
    "MARUTI":     "Automobile",
    "TITAN":      "Consumer Goods",
    "NESTLEIND":  "FMCG",
    "WIPRO":      "IT",
    "SUNPHARMA":  "Pharma",
    "ONGC":       "Energy",
    "TATAMOTORS": "Automobile",
}

# Primary categories from CATEGORY.txt (first keyword per line)
PRIMARY_CATEGORIES = [
    "Acquisition", "Battery", "Bonus", "Buyback", "Capacity Expansion",
    "Capital Structure", "CDMO", "Conference Call", "Credit Rating",
    "Delisting", "Dividend", "Equity Listing", "EV", "Exchange Migration",
    "Incident", "Investor Meet", "Investor Presentation", "IPO",
    "Joint Venture", "Management Update", "Merger/Demerger", "Name Change",
    "New Business", "New Order", "Offer For Sale", "Oil Discovery",
    "One Time Settlement", "Open Offer", "Order Win", "Patent", "PLI",
    "Press Release", "Regulatory", "Resolution", "Results",
    "Rights Issue", "Sale or Transfer of Assets", "Smart Meters",
    "Solar", "Takeover", "Tariffs", "Tax Related", "Update",
]

# Map NSE subject strings → primary category (order matters — more specific first)
_NSE_SUBJECT_MAP = {
    # Results
    "financial result": "Results",
    "quarterly result": "Results",
    "annual result": "Results",
    "financial statement": "Results",
    "standalone result": "Results",
    "consolidated result": "Results",
    "half year result": "Results",
    "unaudited result": "Results",
    "audited result": "Results",
    "earnings": "Results",
    # Conference Call (before analyst to be specific)
    "con. call": "Conference Call",
    "conference call": "Conference Call",
    "concall": "Conference Call",
    "earnings call": "Conference Call",
    "institutional investor meet": "Conference Call",
    "analyst meet": "Conference Call",
    "analyst": "Conference Call",
    "transcript": "Conference Call",
    # Investor Presentation
    "investor presentation": "Investor Presentation",
    "investor meet": "Investor Meet",
    # Acquisition / Takeover
    "takeover": "Takeover",
    "acquisition": "Acquisition",
    # Merger/Demerger
    "amalgamation": "Merger/Demerger",
    "scheme of arrangement": "Merger/Demerger",
    "composite scheme": "Merger/Demerger",
    "demerger": "Merger/Demerger",
    "merger": "Merger/Demerger",
    "nclt": "Merger/Demerger",
    "spin off": "Merger/Demerger",
    "spin-off": "Merger/Demerger",
    # Dividend
    "interim dividend": "Dividend",
    "final dividend": "Dividend",
    "special dividend": "Dividend",
    "dividend": "Dividend",
    # Buyback
    "buy-back": "Buyback",
    "buyback": "Buyback",
    "share repurchase": "Buyback",
    # Bonus
    "bonus issue": "Bonus",
    "bonus": "Bonus",
    # Rights Issue
    "rights issue": "Rights Issue",
    "rights entitlement": "Rights Issue",
    # Capital Structure (specific before generic)
    "preferential allotment": "Capital Structure",
    "preferential issue": "Capital Structure",
    "qualified institutional placement": "Capital Structure",
    "qip": "Capital Structure",
    "non-convertible debenture": "Capital Structure",
    "debenture": "Capital Structure",
    "ncd": "Capital Structure",
    "allotment": "Capital Structure",
    "capital structure": "Capital Structure",
    "esop": "Capital Structure",
    "esos": "Capital Structure",
    "warrant": "Capital Structure",
    "pledge": "Capital Structure",
    "encumbrance": "Capital Structure",
    "gdr": "Capital Structure",
    "adr": "Capital Structure",
    "fpo": "Capital Structure",
    "fund raise": "Capital Structure",
    "fundraise": "Capital Structure",
    "fundraising": "Capital Structure",
    "borrowing": "Capital Structure",
    # Order Win / New Order
    "order win": "Order Win",
    "new order": "New Order",
    "order receipt": "New Order",
    # Capacity Expansion
    "capacity expansion": "Capacity Expansion",
    "capacity addition": "Capacity Expansion",
    "greenfield": "Capacity Expansion",
    "brownfield": "Capacity Expansion",
    # Joint Venture
    "joint venture": "Joint Venture",
    "memorandum of understanding": "Joint Venture",
    "mou": "Joint Venture",
    "collaboration": "Joint Venture",
    "partnership": "Joint Venture",
    # Press Release
    "press release": "Press Release",
    "newspaper publication": "Press Release",
    "newspaper advertisement": "Press Release",
    "public notice": "Press Release",
    # Credit Rating
    "credit rating": "Credit Rating",
    "rating": "Credit Rating",
    # Resolution
    "annual general": "Resolution",
    "extraordinary general": "Resolution",
    "postal ballot": "Resolution",
    "scrutinizer": "Resolution",
    "shareholders meeting": "Resolution",
    "voting results": "Resolution",
    "e-voting": "Resolution",
    "resolution": "Resolution",
    "agm": "Resolution",
    "egm": "Resolution",
    "proceedings": "Resolution",
    # Management Update
    "outcome of board": "Management Update",
    "intimation of board": "Management Update",
    "board meeting": "Management Update",
    "change in directorate": "Management Update",
    "change of director": "Management Update",
    "managing director": "Management Update",
    "key managerial": "Management Update",
    "appointment": "Management Update",
    "cessation": "Management Update",
    "auditor": "Management Update",
    "company secretary": "Management Update",
    "management": "Management Update",
    # Regulatory
    "insider trading": "Regulatory",
    "trading window": "Regulatory",
    "sast": "Regulatory",
    "sebi": "Regulatory",
    "rera": "Regulatory",
    "penalty": "Regulatory",
    "show cause": "Regulatory",
    "compliance": "Regulatory",
    "regulatory": "Regulatory",
    # IPO
    "initial public offer": "IPO",
    "sme ipo": "IPO",
    "ipo": "IPO",
    # Offer For Sale
    "offer for sale": "Offer For Sale",
    "ofs": "Offer For Sale",
    # Sale or Transfer of Assets
    "sale or transfer": "Sale or Transfer of Assets",
    "divestment": "Sale or Transfer of Assets",
    "divestiture": "Sale or Transfer of Assets",
    # Solar / Renewable
    "renewable energy": "Solar",
    "wind energy": "Solar",
    "green energy": "Solar",
    "solar": "Solar",
    # EV
    "electric vehicle": "EV",
    # Battery
    "energy storage": "Battery",
    "battery": "Battery",
    # Oil Discovery
    "oil discovery": "Oil Discovery",
    "gas discovery": "Oil Discovery",
    "hydrocarbon": "Oil Discovery",
    # Tax Related
    "income tax": "Tax Related",
    "gst": "Tax Related",
    "customs duty": "Tax Related",
    "tax": "Tax Related",
    # Incident
    "incident": "Incident",
    # Smart Meters
    "smart meter": "Smart Meters",
    # PLI
    "production linked incentive": "PLI",
    "pli": "PLI",
    # Patent
    "intellectual property": "Patent",
    "patent": "Patent",
    # CDMO
    "contract research": "CDMO",
    "cdmo": "CDMO",
    # Delisting
    "delisting": "Delisting",
    "delist": "Delisting",
    # Equity Listing / Exchange Migration
    "exchange migration": "Exchange Migration",
    "listing": "Equity Listing",
    # Open Offer
    "open offer": "Open Offer",
    # One Time Settlement
    "one time settlement": "One Time Settlement",
    # Name Change
    "name change": "Name Change",
    "change of name": "Name Change",
    # Update (generic — keep near end)
    "general update": "Update",
    "update": "Update",
    "intimation": "Update",
    # Other — should stay as Other
    "loss of share": "Other",
    "duplicate share": "Other",
    "share certificate": "Other",
}

# ── AI category cache (persists across sessions) ──────────────────
_AI_CAT_CACHE_PATH = DATA_DIR / "category_cache.json"


def _load_category_cache() -> dict:
    if _AI_CAT_CACHE_PATH.exists():
        try:
            return json.loads(_AI_CAT_CACHE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_category_cache(cache: dict):
    try:
        _AI_CAT_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass


def categorise_subject(subject: str, ai_cache: dict = None) -> str:
    """Keyword map first; fall back to AI cache if available."""
    s = str(subject).lower()
    for keyword, cat in _NSE_SUBJECT_MAP.items():
        if keyword in s:
            return cat
    if ai_cache and subject in ai_cache:
        return ai_cache[subject]
    return "Other"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Badge helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SENT_STYLE = {
    "bullish": "background:#1a3a2a;color:#3fb950;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:600",
    "bearish": "background:#3a1a1a;color:#f85149;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:600",
    "neutral": "background:#1c2333;color:#8b949e;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:600",
}
_SIG_COLOR = {"buy": "#3fb950", "sell": "#f85149", "hold": "#d29922", "watch": "#58a6ff"}


def _sent_badge(s: str) -> str:
    return f'<span style="{_SENT_STYLE.get(s, _SENT_STYLE["neutral"])}">{s.upper()}</span>'


def _sig_badge(s: str) -> str:
    return f'<span style="color:{_SIG_COLOR.get(s,"#8b949e")};font-weight:700">▶ {s.upper()}</span>'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data loaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=600, show_spinner="Loading all filings data...")
def load_all_parquets() -> pd.DataFrame:
    """Load every *_All.parquet and combine."""
    ai_cache = _load_category_cache()
    frames = []
    for path in sorted(PARQUET_DIR.glob("*_All.parquet")):
        symbol = path.stem.replace("_All", "")
        try:
            df = pd.read_parquet(path)
            df["symbol"] = symbol
            df["industry"] = INDUSTRY_MAP.get(symbol, "Other")
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["broadcast_dt"] = pd.to_datetime(df["broadcast_dt"], errors="coerce")
    df["primary_category"] = df["subject"].apply(lambda s: categorise_subject(s, ai_cache))
    return df


@st.cache_data(ttl=600)
def load_symbol_parquet(symbol: str, category: str) -> pd.DataFrame:
    ai_cache = _load_category_cache()
    safe_cat = category.replace(" ", "_").replace("/", "_")
    path = PARQUET_DIR / f"{symbol}_{safe_cat}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["broadcast_dt"] = pd.to_datetime(df["broadcast_dt"], errors="coerce")
        df["primary_category"] = df["subject"].apply(lambda s: categorise_subject(s, ai_cache))
        return df
    return pd.DataFrame()


@st.cache_resource(ttl=300)
def get_scraper():
    from scrapers.nse_filings import NSEFilingsScraper
    return NSEFilingsScraper()


@st.cache_data(ttl=300)
def search_company(query: str):
    return get_scraper().search_company(query)


def load_analysis(filing_id: str) -> dict:
    path = ANALYSIS_DIR / f"{filing_id}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def fetch_pdf_for_analysis(filing_id: str, pdf_url: str, local_pdf: str = "") -> Path | None:
    """
    Resolve a PDF for AI analysis. Priority: local file → Google Drive → NSE direct URL.
    Returns local Path if found/downloaded, else None.
    """
    # 1. Check local PDF_DIR
    local = PDF_DIR / f"{filing_id}.pdf"
    if local.exists():
        return local
    # 2. Check local_pdf column path
    if local_pdf and local_pdf not in ["nan", "None", ""]:
        p = Path(local_pdf)
        if p.exists():
            return p
    # 3. Try Google Drive
    try:
        from storage.drive_handler import DriveHandler
        dh = DriveHandler()
        buf = dh.download_pdf_by_name(f"{filing_id}.pdf")
        if buf:
            local.write_bytes(buf.read())
            return local
    except Exception:
        pass
    # 4. Try direct NSE URL
    if pdf_url and isinstance(pdf_url, str) and pdf_url.startswith("http"):
        try:
            from scrapers.nse_filings import NSEFilingsScraper
            scraper = NSEFilingsScraper()
            path = scraper.download_pdf(pdf_url, filing_id)
            if path:
                return path
        except Exception:
            pass
    return None


def make_wordcloud(text_series: pd.Series):
    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text = " ".join(text_series.dropna().astype(str).tolist())
    wc = WordCloud(
        width=700, height=340,
        background_color="#0e1117",
        colormap="Blues",
        max_words=120,
        prefer_horizontal=0.8,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor="#0e1117")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0e1117", dpi=120)
    plt.close()
    buf.seek(0)
    return buf


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AI analysis renderer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _safe_float(val, default: float = 0.0) -> float:
    """Safe float conversion — returns default on any error."""
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _safe_json(val, default):
    """Safe json.loads — returns default on any error."""
    if not val or str(val) in ("nan", "None", ""):
        return default
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return default


def _render_analysis(a: dict, _key: str = ""):
    sentiment = a.get("sentiment", "neutral")
    score     = _safe_float(a.get("sentiment_score", 0))
    signal    = a.get("action_signal", "watch")
    conf      = _safe_float(a.get("confidence", 0))
    model     = a.get("model_used", "")
    horizon   = a.get("time_horizon", "")

    horizon_tag = f" · ⏱ {horizon}" if horizon else ""
    st.markdown(
        f'{_sent_badge(sentiment)} &nbsp; {_sig_badge(signal)} &nbsp;'
        f'<span style="color:#8b949e;font-size:12px">Conf: {conf:.0%}{horizon_tag} · {model}</span>',
        unsafe_allow_html=True,
    )
    if a.get("summary"):
        st.info(a["summary"])

    # ── Investment Thesis (new) ────────────────────────────────────
    thesis = a.get("investment_thesis", "")
    catalysts = a.get("key_catalysts", [])
    if thesis or catalysts:
        st.markdown("**💡 Investment Thesis**")
        if thesis:
            st.success(thesis)
        if catalysts:
            st.markdown("**⚡ Key Catalysts**")
            for c in catalysts:
                st.markdown(f"🚀 {c}")
        st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        hl = a.get("key_highlights", [])
        if hl:
            st.markdown("**Key Highlights**")
            for h in hl: st.markdown(f"✅ {h}")
    with c2:
        rk = a.get("risk_factors", [])
        if rk:
            st.markdown("**Risk Factors**")
            for r in rk: st.markdown(f"⚠️ {r}")

    fin = a.get("financial_data", {})
    if fin and any(fin.get(k) for k in ["revenue", "net_profit", "eps"]):
        st.markdown("**Extracted Financials**")
        fc1, fc2, fc3, fc4 = st.columns(4)
        def _fmt(v):
            if v is None: return "—"
            try: return f"₹{float(v):,.0f} Cr"
            except: return str(v)
        fc1.metric("Revenue",    _fmt(fin.get("revenue")))
        fc2.metric("Net Profit", _fmt(fin.get("net_profit")))
        fc3.metric("EPS",        f"₹{fin.get('eps')}" if fin.get("eps") else "—")
        fc4.metric("DPS",        f"₹{fin.get('dividend_per_share')}" if fin.get("dividend_per_share") else "—")

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain={"x": [0,1], "y": [0,1]},
        title={"text": "Sentiment Score", "font": {"color": "#e6edf3"}},
        gauge={
            "axis": {"range": [-1,1], "tickcolor": "#8b949e"},
            "bar": {"color": "#58a6ff"},
            "steps": [
                {"range": [-1,-0.3], "color": "#3a1a1a"},
                {"range": [-0.3,0.3],"color": "#21262d"},
                {"range": [0.3,1],   "color": "#1a3a2a"},
            ],
        },
        number={"font": {"color": "#e6edf3"}},
    ))
    fig.update_layout(height=180, paper_bgcolor="#161b22",
                      font_color="#e6edf3", margin=dict(l=10,r=10,t=30,b=5))
    st.plotly_chart(fig, use_container_width=True, key=f"sentiment_gauge_{_key}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page config + CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="NSE Filings Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background-color: #0e1117; }
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#161b22 0%,#0d1117 100%);
    border-right: 1px solid #30363d;
  }
  h1,h2,h3 { color:#e6edf3 !important; }
  div[data-testid="stMetricValue"] { color:#e6edf3 !important; font-weight:700; }
  div[data-testid="stMetricLabel"] { color:#8b949e !important; }
  .stat-card {
    background:#161b22; border:1px solid #30363d; border-radius:12px;
    padding:20px 24px; text-align:center;
  }
  .stat-card .val { font-size:36px; font-weight:800; color:#ffa726; }
  .stat-card .lbl { font-size:13px; color:#8b949e; margin-top:4px; }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown("### 📊 NSE Filings Intel")
    st.markdown("---")

    # Industry filter
    industries = ["All"] + sorted(set(INDUSTRY_MAP.values()))
    industry_filter = st.selectbox("🏭 Industry", industries)

    st.markdown("---")

    # Company search
    st.markdown("**Search Company**")
    company_query = st.text_input("Symbol or name", placeholder="RELIANCE, TCS, HDFC...",
                                   label_visibility="collapsed")

    selected_symbol = None
    selected_name   = ""

    if company_query and len(company_query) >= 2:
        with st.spinner("Searching..."):
            results = search_company(company_query.upper())
        if results:
            options = {f"{r['symbol']} — {r['name']}": r["symbol"] for r in results[:8]}
            chosen = st.selectbox("Pick", list(options.keys()), label_visibility="collapsed")
            selected_symbol = options[chosen]
            selected_name   = chosen.split(" — ")[1]
        else:
            st.warning("No results found.")

    st.markdown("---")
    st.markdown("**Filing Type**")
    category = st.selectbox("Category", list(FILING_CATEGORIES.keys()),
                             label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Date Range**")
    c1, c2 = st.columns(2)
    with c1: from_date = st.date_input("From", value=date.today()-timedelta(days=365*5))
    with c2: to_date   = st.date_input("To",   value=date.today())

    if selected_symbol:
        st.markdown("---")
        fetch_btn = st.button("🔍 Fetch Filings", type="primary", use_container_width=True)
    else:
        fetch_btn = False

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#8b949e;line-height:1.9">
    <b style="color:#e6edf3">AI Provider Split</b><br>
    🟢 Gemini 2.0 Flash — 40% · Free<br>
    🟡 Groq Llama-3.3 — 30% · Free<br>
    🔵 OpenRouter — 20% · Free<br>
    🔴 DeepSeek — 10% · ~$0.002/filing
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OVERVIEW DASHBOARD (no company selected)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fix_creds_json(raw: str) -> str:
    """
    Re-escape literal newlines that appear inside JSON string values.
    TOML triple-quoted secrets convert \\n → actual newline, which breaks
    JSON parsing (control characters are not allowed in JSON strings).
    """
    try:
        return json.dumps(json.loads(raw))   # already valid — just re-serialize
    except json.JSONDecodeError:
        pass
    # Walk char-by-char; escape bare newlines/carriage-returns inside strings
    fixed, in_str, prev = [], False, ""
    for ch in raw:
        if ch == '"' and prev != "\\":
            in_str = not in_str
        if in_str and ch == "\n":
            fixed.append("\\n")
        elif in_str and ch == "\r":
            fixed.append("\\r")
        else:
            fixed.append(ch)
        prev = ch
    return json.dumps(json.loads("".join(fixed)))


def _prepare_credentials() -> "str | None":
    """
    Return a path to a valid credentials JSON file.
    Checks (in order):
      1. Physical credentials.json on disk (local dev)
      2. st.secrets["GDRIVE_CREDENTIALS_JSON"]  (Streamlit Cloud — JSON string)
      3. st.secrets["GOOGLE_CREDENTIALS_JSON"]   (alternate key name)
      4. st.secrets["gcp_service_account"]       (Streamlit Cloud — TOML table)
    """
    from config import GDRIVE_CREDENTIALS
    if Path(GDRIVE_CREDENTIALS).exists():
        return GDRIVE_CREDENTIALS
    tmp = Path("/tmp/gcp_credentials.json")
    for key in ("GDRIVE_CREDENTIALS_JSON", "GOOGLE_CREDENTIALS_JSON"):
        try:
            raw = st.secrets.get(key)
            if raw:
                text = _fix_creds_json(raw) if isinstance(raw, str) else json.dumps(dict(raw))
                tmp.write_text(text)
                return str(tmp)
        except Exception:
            pass
    try:
        sa = st.secrets.get("gcp_service_account")
        if sa:
            tmp.write_text(json.dumps(dict(sa)))
            return str(tmp)
    except Exception:
        pass
    return None


def _get_drive_folder_id() -> str:
    """Return GOOGLE_DRIVE_FOLDER_ID from env or st.secrets."""
    from config import GDRIVE_FOLDER_ID
    if GDRIVE_FOLDER_ID:
        return GDRIVE_FOLDER_ID
    try:
        return st.secrets.get("GOOGLE_DRIVE_FOLDER_ID", "")
    except Exception:
        return ""


def _try_drive_sync(show_status: bool = True) -> bool:
    """
    Attempt to pull parquets + analysis JSONs from Google Drive to local.
    Returns True if any files were downloaded.
    """
    folder_id = _get_drive_folder_id()
    creds_path = _prepare_credentials()
    if not folder_id or not creds_path:
        return False
    try:
        from storage.drive_handler import DriveHandler
        drive = DriveHandler(folder_id=folder_id, credentials_path=creds_path)
        # Force correct Shared Drive mode regardless of cached config
        drive.shared_drive = folder_id.startswith("0A")
        pulled_parquets = drive.sync_parquets_from_drive()
        pulled_analysis = drive.sync_analysis_from_drive()
        return bool(pulled_parquets or pulled_analysis)
    except Exception as e:
        if show_status:
            st.warning(f"Drive sync failed: {e}")
        return False


if not selected_symbol:
    st.markdown("## 📊 NSE Corporate Filings Intelligence")
    st.caption(f"Last updated: {date.today().strftime('%d %B %Y')} · Updates Daily")
    st.markdown("---")

    all_df = load_all_parquets()

    # ── Auto-sync from Drive if no local data ────────────────────
    if all_df.empty and not st.session_state.get("_drive_sync_tried"):
        st.session_state["_drive_sync_tried"] = True
        with st.spinner("☁️ No local data — fetching from Google Drive..."):
            synced = _try_drive_sync(show_status=False)
        if synced:
            st.cache_data.clear()
            st.rerun()

    # Re-check after possible sync
    if all_df.empty:
        all_df = load_all_parquets()

    # Apply industry filter
    if industry_filter != "All" and not all_df.empty:
        all_df = all_df[all_df["industry"] == industry_filter]

    # Apply date filter
    if not all_df.empty and "broadcast_dt" in all_df.columns:
        all_df = all_df[
            (all_df["broadcast_dt"].dt.date >= from_date) &
            (all_df["broadcast_dt"].dt.date <= to_date)
        ]

    total_filings  = len(all_df) if not all_df.empty else 0
    total_symbols  = all_df["symbol"].nunique() if not all_df.empty else 0
    analyzed_count = int(all_df["ai_summary"].notna().sum()) if ("ai_summary" in all_df.columns and not all_df.empty) else 0

    # ── Summary cards
    hdr_left, hdr_right = st.columns([5, 2])
    with hdr_right:
        if st.button("☁️ Sync from Drive", help="Pull latest parquets & analysis from Google Drive"):
            with st.spinner("Syncing from Google Drive..."):
                synced = _try_drive_sync(show_status=True)
            if synced:
                st.cache_data.clear()
                st.success("Sync complete!")
                st.rerun()
            else:
                st.info("Nothing new on Drive (or Drive not configured).")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(f"""<div class="stat-card"><div class="val">{total_filings:,}</div><div class="lbl">Total Filings</div></div>""", unsafe_allow_html=True)
    mc2.markdown(f"""<div class="stat-card"><div class="val">{total_symbols}</div><div class="lbl">Companies</div></div>""", unsafe_allow_html=True)
    mc3.markdown(f"""<div class="stat-card"><div class="val">{analyzed_count:,}</div><div class="lbl">AI Analyzed</div></div>""", unsafe_allow_html=True)
    pct = f"{analyzed_count/total_filings*100:.1f}%" if total_filings > 0 else "—"
    mc4.markdown(f"""<div class="stat-card"><div class="val" style="color:#3fb950">{pct}</div><div class="lbl">Analysis Coverage</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    if all_df.empty:
        drive_ok = bool(_get_drive_folder_id() and _prepare_credentials())
        if drive_ok:
            st.warning("No data found locally or on Drive. Has the pipeline been run yet?")
            st.markdown(
                "**Run the pipeline first** — it will scrape NSE, then upload to Google Drive:\n"
                "```\npython run_pipeline.py\n```"
            )
        else:
            st.warning("No local data and Google Drive is not configured.")
            st.markdown(
                "**Option 1 — Run pipeline locally:**\n```\npython run_pipeline.py\n```\n\n"
                "**Option 2 — Configure Drive** — add `GOOGLE_DRIVE_FOLDER_ID` and `credentials.json`."
            )
        st.stop()

    # ── Word cloud + Category table
    left_col, right_col = st.columns([4, 3])

    with left_col:
        st.markdown("#### Filing Subject Word Cloud")
        try:
            buf = make_wordcloud(all_df["subject"])
            st.image(buf, use_container_width=True)
        except Exception as e:
            st.caption(f"Word cloud unavailable: {e}")

    with right_col:
        st.markdown("#### Category Breakdown")
        cat_counts = all_df["primary_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Total Filings"]
        cat_counts = cat_counts[cat_counts["Category"] != "Other"].head(15)

        fig_cat = px.bar(
            cat_counts, x="Total Filings", y="Category", orientation="h",
            color="Total Filings",
            color_continuous_scale=[[0,"#21262d"],[1,"#58a6ff"]],
            template="plotly_dark",
        )
        fig_cat.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
            showlegend=False, coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(l=10,r=10,t=10,b=10),
            height=380,
        )
        fig_cat.update_traces(texttemplate="%{x:,}", textposition="outside")
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")

    # ── Cross-stock timeline
    st.markdown("#### Filing Activity — All Stocks (Monthly)")
    monthly = (
        all_df.dropna(subset=["broadcast_dt"])
        .groupby([all_df["broadcast_dt"].dt.to_period("M").astype(str), "industry"])
        .size()
        .reset_index(name="count")
    )
    monthly.columns = ["month", "industry", "count"]

    fig_timeline = px.bar(
        monthly, x="month", y="count", color="industry",
        title="Monthly Filing Frequency by Industry",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_timeline.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
        xaxis_title="Month", yaxis_title="Filings",
        legend_title="Industry",
        height=300,
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ── Per-industry category breakdown
    st.markdown("#### Filings by Industry")
    ind_data = all_df.groupby("industry").size().reset_index(name="count").sort_values("count", ascending=False)
    fig_ind = px.pie(ind_data, values="count", names="industry",
                     hole=0.45, template="plotly_dark",
                     color_discrete_sequence=px.colors.qualitative.Bold)
    fig_ind.update_layout(paper_bgcolor="#0e1117", height=320,
                          legend=dict(font=dict(color="#8b949e")))
    st.plotly_chart(fig_ind, use_container_width=True)

    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPANY DETAIL VIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

industry = INDUSTRY_MAP.get(selected_symbol, "")
industry_tag = f" · {industry}" if industry else ""
st.markdown(f"## {selected_symbol}")
st.markdown(f"*{selected_name}*{industry_tag} &nbsp;·&nbsp; `{category}` &nbsp;·&nbsp; {from_date} → {to_date}")
st.markdown("---")

# Load data from parquet (no live fetch unless explicitly clicked)
df = load_symbol_parquet(selected_symbol, category)
if df.empty or fetch_btn:
    with st.spinner("Fetching from NSE..."):
        from scrapers.nse_filings import NSEFilingsScraper
        scraper = NSEFilingsScraper()
        df = scraper.fetch_filings(selected_symbol, category, from_date, to_date)
        if not df.empty:
            df["broadcast_dt"] = pd.to_datetime(df["broadcast_dt"], errors="coerce")
            df["primary_category"] = df["subject"].apply(categorise_subject)

if not df.empty and "broadcast_dt" in df.columns:
    df = df[
        (df["broadcast_dt"].dt.date >= from_date) &
        (df["broadcast_dt"].dt.date <= to_date)
    ].sort_values("broadcast_dt", ascending=False)

if df is None or df.empty:
    st.warning(f"No filings for **{selected_symbol}** / **{category}** in the selected range.")
    st.stop()

# ── Metrics
total    = len(df)
with_pdf = int(df["pdf_url"].notna().sum()) if "pdf_url" in df.columns else 0
analyzed = int(df["ai_summary"].notna().sum()) if "ai_summary" in df.columns else 0
latest   = df["broadcast_dt"].max()
latest_s = latest.strftime("%d %b %Y") if pd.notna(latest) else "—"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Filings", f"{total:,}")
m2.metric("With PDF",      f"{with_pdf:,}")
m3.metric("AI Analyzed",   f"{analyzed:,}")
m4.metric("Latest Filing", latest_s)
st.markdown("---")

# ── Tabs
tab_filings, tab_charts, tab_ai, tab_ob, tab_chat = st.tabs(["📋 Filings", "📈 Charts", "🤖 AI Insights", "📦 Orderbook", "💬 Chat"])


# ──────────────────────────────────────────────────────────────────
# TAB 1 — FILINGS
# ──────────────────────────────────────────────────────────────────

with tab_filings:
    # Subject category filter (from CATEGORY.txt mapping)
    cats_in_data = ["All"] + sorted(df["primary_category"].dropna().unique().tolist())
    cat_filter = st.selectbox("📂 Subject Category", cats_in_data, label_visibility="visible")

    search_text = st.text_input("🔎 Filter", placeholder="Search subject / details...",
                                 label_visibility="collapsed")

    display_df = df.copy()
    if cat_filter != "All":
        display_df = display_df[display_df["primary_category"] == cat_filter]
    if search_text:
        mask = (
            display_df.get("subject", pd.Series(dtype=str)).str.contains(search_text, case=False, na=False) |
            display_df.get("details", pd.Series(dtype=str)).str.contains(search_text, case=False, na=False)
        )
        display_df = display_df[mask]

    st.caption(f"Showing {len(display_df):,} filings")

    for idx, row in display_df.head(100).iterrows():
        filing_id = str(row.get("filing_id", ""))
        subject   = str(row.get("subject", "—"))
        prim_cat  = str(row.get("primary_category", ""))
        details   = str(row.get("details", ""))[:200]
        dt_str    = row["broadcast_dt"].strftime("%d %b %Y, %I:%M %p") if pd.notna(row.get("broadcast_dt")) else "—"
        has_pdf   = bool(row.get("pdf_url")) and str(row.get("pdf_url","")) not in ["nan","None",""]
        has_ai    = bool(row.get("ai_summary")) and str(row.get("ai_summary","")) not in ["nan","None",""]
        sentiment = str(row.get("ai_sentiment", ""))
        signal    = str(row.get("ai_signal", ""))

        with st.container(border=True):
            lc, rc = st.columns([5, 2])
            with lc:
                badges = ""
                if sentiment in _SENT_STYLE:
                    badges += _sent_badge(sentiment) + " &nbsp; "
                if signal in _SIG_COLOR:
                    badges += _sig_badge(signal)
                cat_pill = f'<span style="background:#21262d;color:#79c0ff;padding:1px 8px;border-radius:8px;font-size:11px;margin-right:6px">{prim_cat}</span>' if prim_cat and prim_cat != "Other" else ""
                st.markdown(
                    f'{cat_pill}<span style="color:#e6edf3;font-weight:600;font-size:14px">{subject[:70]}</span>'
                    + (" &nbsp; " + badges if badges else ""),
                    unsafe_allow_html=True,
                )
                if details and details not in ["nan","None"]:
                    st.caption(details)
            with rc:
                st.caption(dt_str)
                if has_pdf: st.caption("📄 PDF available")
                if has_ai:  st.caption("🤖 Analyzed")

            b1, b2, _ = st.columns([1,1,5])
            if has_pdf and b1.button("📄 PDF", key=f"pdf_{filing_id}_{idx}"):
                st.markdown(f"[🔗 Open PDF ↗]({row['pdf_url']})")

            if filing_id and b2.button("🤖 Analyze", key=f"ai_{filing_id}_{idx}"):
                try:
                    analysis = load_analysis(filing_id)
                    if not analysis:
                        # Auto-fetch PDF: local → Drive → NSE URL
                        with st.spinner("Fetching PDF (local → Drive → NSE)..."):
                            pdf_path = fetch_pdf_for_analysis(
                                filing_id,
                                str(row.get("pdf_url", "")),
                                str(row.get("local_pdf", "")),
                            )
                        if pdf_path:
                            with st.spinner("Running AI analysis..."):
                                from processors.ai_analyzer import analyze_filing
                                analysis = analyze_filing(
                                    pdf_path  = pdf_path,
                                    company   = str(row.get("company", selected_symbol)),
                                    subject   = subject,
                                    date_str  = str(row.get("broadcast_dt", ""))[:10],
                                    filing_id = filing_id,
                                )
                        else:
                            st.warning("PDF not available locally, on Drive, or via NSE URL.")
                    if analysis:
                        with st.expander(f"AI Analysis — {subject[:55]}", expanded=True):
                            _render_analysis(analysis, _key=filing_id)
                    elif analysis is not None:
                        st.error("AI analysis failed — all providers returned empty results. Check API keys.")
                except Exception as _e:
                    st.error(f"❌ Analysis error: {_e}")


# ──────────────────────────────────────────────────────────────────
# TAB 2 — CHARTS  (use broadcast_dt across all timeframes)
# ──────────────────────────────────────────────────────────────────

with tab_charts:
    if df.empty or "broadcast_dt" not in df.columns:
        st.info("No data to chart.")
    else:
        df_c = df.dropna(subset=["broadcast_dt"]).copy()
        df_c["year_month"] = df_c["broadcast_dt"].dt.to_period("M").astype(str)
        df_c["year"]       = df_c["broadcast_dt"].dt.year.astype(str)

        # ── AI fix for "Other" categories ────────────────────────
        other_mask = df_c["primary_category"] == "Other"
        other_count = int(other_mask.sum())
        if other_count > 0:
            col_fix, col_info = st.columns([2, 5])
            with col_fix:
                fix_btn = st.button(
                    f"🤖 Fix {other_count} 'Other' Categories with AI",
                    help="Uses DeepSeek/Groq to classify miscategorised filings",
                )
            with col_info:
                st.caption(f"{other_count} filings currently in 'Other' category. AI will reclassify them.")
            if fix_btn:
                other_subjects = df_c.loc[other_mask, "subject"].dropna().unique().tolist()
                with st.spinner(f"Classifying {len(other_subjects)} subjects with AI..."):
                    from processors.ai_analyzer import ai_categorise_subjects
                    cache = _load_category_cache()
                    uncached = [s for s in other_subjects if s not in cache]
                    if uncached:
                        new_cats = ai_categorise_subjects(uncached, PRIMARY_CATEGORIES)
                        cache.update(new_cats)
                        _save_category_cache(cache)
                    # Apply cache to df_c
                    df_c.loc[other_mask, "primary_category"] = df_c.loc[other_mask, "subject"].apply(
                        lambda s: cache.get(s, "Other")
                    )
                fixed = int((df_c["primary_category"] != "Other").sum() - int((~other_mask).sum()))
                st.success(f"Reclassified {fixed} filings. Reload the app to persist across sessions.")
                st.cache_data.clear()

        # 1. Filing frequency — monthly bar
        freq = df_c.groupby("year_month").size().reset_index(name="count")
        fig1 = px.bar(freq, x="year_month", y="count",
                      title=f"{selected_symbol} — Monthly Filing Frequency",
                      color_discrete_sequence=["#58a6ff"], template="plotly_dark")
        fig1.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                           xaxis_title="Month", yaxis_title="Filings",
                           showlegend=False, height=280)
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Subject category breakdown (primary_category)
        cat_c = df_c["primary_category"].value_counts().head(20).reset_index()
        cat_c.columns = ["category", "count"]
        fig2 = px.bar(cat_c, x="count", y="category", orientation="h",
                      title="Filings by Subject Category",
                      color="count",
                      color_continuous_scale=[[0,"#21262d"],[1,"#3fb950"]],
                      template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                           yaxis=dict(categoryorder="total ascending"),
                           coloraxis_showscale=False, height=420)
        fig2.update_traces(texttemplate="%{x}", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Sentiment over time (using broadcast_dt)
        if "ai_sentiment_score" in df_c.columns:
            sent_df = df_c[df_c["ai_sentiment_score"].notna()].copy()
            sent_df["ai_sentiment_score"] = pd.to_numeric(sent_df["ai_sentiment_score"], errors="coerce")
            if not sent_df.empty:
                fig3 = px.scatter(
                    sent_df.sort_values("broadcast_dt"),
                    x="broadcast_dt", y="ai_sentiment_score",
                    color="ai_sentiment",
                    color_discrete_map={"bullish":"#3fb950","bearish":"#f85149","neutral":"#8b949e"},
                    title="AI Sentiment Over Time (filing date)",
                    hover_data=["subject","primary_category"],
                    template="plotly_dark",
                    size_max=10,
                )
                fig3.add_hline(y=0, line_dash="dash", line_color="#30363d")
                fig3.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                   xaxis_title="Filing Date", height=320)
                st.plotly_chart(fig3, use_container_width=True)

        # 4. Annual filing trend
        yr = df_c.groupby(["year","primary_category"]).size().reset_index(name="count")
        yr_top = df_c["primary_category"].value_counts().head(8).index.tolist()
        yr = yr[yr["primary_category"].isin(yr_top)]
        fig4 = px.bar(yr, x="year", y="count", color="primary_category",
                      title="Annual Filing Trend by Category",
                      template="plotly_dark",
                      color_discrete_sequence=px.colors.qualitative.Plotly)
        fig4.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                           xaxis_title="Year", height=320,
                           legend_title="Category")
        st.plotly_chart(fig4, use_container_width=True)

        # 5. Financial data from AI (using broadcast_dt as x-axis)
        fin_records = []
        for path in ANALYSIS_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                if data.get("filing_id") and data.get("financial_data"):
                    fin = data["financial_data"]
                    if any(fin.get(k) for k in ["revenue","net_profit","eps"]):
                        # Match back to filing date via filing_id
                        match = df_c[df_c["filing_id"].astype(str) == data["filing_id"]]
                        filing_dt = match["broadcast_dt"].iloc[0] if not match.empty else None
                        fin_records.append({
                            "filing_id":   data["filing_id"],
                            "filing_date": filing_dt,
                            "revenue":     fin.get("revenue"),
                            "net_profit":  fin.get("net_profit"),
                            "eps":         fin.get("eps"),
                        })
            except Exception:
                pass

        if fin_records:
            fin_df = pd.DataFrame(fin_records).dropna(subset=["filing_date"])
            fin_df = fin_df.sort_values("filing_date")
            st.markdown("#### Financial Data (AI-extracted · Filing Date)")
            f1, f2 = st.columns(2)
            with f1:
                if fin_df["revenue"].notna().any():
                    fig_r = px.line(fin_df.dropna(subset=["revenue"]),
                                    x="filing_date", y="revenue",
                                    title="Revenue (₹ Cr)", template="plotly_dark",
                                    color_discrete_sequence=["#58a6ff"])
                    fig_r.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                        xaxis_title="Filing Date", height=260)
                    st.plotly_chart(fig_r, use_container_width=True)
            with f2:
                if fin_df["net_profit"].notna().any():
                    fig_p = px.line(fin_df.dropna(subset=["net_profit"]),
                                    x="filing_date", y="net_profit",
                                    title="Net Profit (₹ Cr)", template="plotly_dark",
                                    color_discrete_sequence=["#3fb950"])
                    fig_p.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                        xaxis_title="Filing Date", height=260)
                    st.plotly_chart(fig_p, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 3 — AI INSIGHTS  (Saved Analysis | Live Analysis)
# ──────────────────────────────────────────────────────────────────

with tab_ai:
    mode_col, _ = st.columns([3, 5])
    with mode_col:
        analysis_mode = st.radio(
            "Analysis Mode",
            ["📁 Saved Analysis", "🔴 Live Analysis"],
            horizontal=True,
            label_visibility="collapsed",
        )

    st.markdown("---")

    # ── SAVED ANALYSIS MODE ───────────────────────────────────────
    if analysis_mode == "📁 Saved Analysis":
        # Show filings that have a cached JSON in ANALYSIS_DIR
        if not df.empty and "filing_id" in df.columns:
            has_analysis = df["filing_id"].apply(
                lambda x: (ANALYSIS_DIR / f"{x}.json").exists()
            )
            saved_df = df[has_analysis]
        else:
            saved_df = pd.DataFrame()

        # Also check ai_summary column as fallback
        if saved_df.empty and "ai_summary" in df.columns:
            saved_df = df[df["ai_summary"].notna()]

        if saved_df.empty:
            st.info("No saved analyses yet for this company/category.")
            st.markdown("**Options to generate analyses:**")
            st.markdown("- Click **🤖 Analyze** on any filing in the Filings tab")
            st.markdown("- Switch to **🔴 Live Analysis** tab to run batch analysis")
            st.code("python run_pipeline.py " + selected_symbol, language="bash")
        else:
            st.caption(f"Showing {len(saved_df)} pre-analyzed filings")
            # Sentiment distribution
            sent_col = "ai_sentiment" if "ai_sentiment" in saved_df.columns else None
            if sent_col:
                sent_dist = saved_df[sent_col].value_counts().reset_index()
                sent_dist.columns = ["sentiment", "count"]
                fig_s = px.bar(
                    sent_dist, x="sentiment", y="count", color="sentiment",
                    color_discrete_map={"bullish": "#3fb950", "bearish": "#f85149", "neutral": "#8b949e"},
                    template="plotly_dark", title="Sentiment Distribution",
                )
                fig_s.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                    showlegend=False, height=220)
                st.plotly_chart(fig_s, use_container_width=True)

            for _, row in saved_df.iterrows():
                fid  = str(row.get("filing_id", ""))
                # Load from JSON file first (has investment_thesis etc.)
                cached = load_analysis(fid) if fid else {}
                if cached:
                    a = cached
                else:
                    a = {
                        "summary":         row.get("ai_summary", ""),
                        "sentiment":       row.get("ai_sentiment", ""),
                        "sentiment_score": row.get("ai_sentiment_score", 0),
                        "action_signal":   row.get("ai_signal", ""),
                        "key_highlights":  _safe_json(row.get("ai_highlights"), []),
                        "risk_factors":    _safe_json(row.get("ai_risks"), []),
                        "financial_data":  _safe_json(row.get("ai_financial"), {}),
                        "confidence":      row.get("ai_confidence", 0),
                        "model_used":      row.get("ai_model", ""),
                    }
                icon  = "📈" if a.get("sentiment") == "bullish" else "📉" if a.get("sentiment") == "bearish" else "📋"
                dt_s  = str(row.get("broadcast_dt", ""))[:10]
                label = f"{icon} {row.get('subject', '—')} — {dt_s}"
                with st.expander(label[:80], expanded=False):
                    _render_analysis(a, _key=fid)

    # ── LIVE ANALYSIS MODE ────────────────────────────────────────
    else:
        st.markdown("#### 🔴 Live Analysis")
        st.caption(
            "Select filings with PDFs below. The app will fetch the PDF (local → Drive → NSE) "
            "and run AI analysis in real-time. Results are cached for future use."
        )

        pdfs_df = df[
            df["pdf_url"].notna() &
            df["pdf_url"].astype(str).ne("nan") &
            df["pdf_url"].astype(str).ne("")
        ] if not df.empty else pd.DataFrame()

        if pdfs_df.empty:
            st.info("No filings with PDFs found in the current filter.")
        else:
            filing_opts: dict[str, dict] = {}
            for _, row in pdfs_df.iterrows():
                dt_s  = str(row.get("broadcast_dt", ""))[:10]
                label = f"{dt_s} — {str(row.get('subject', '?'))[:65]}"
                if label not in filing_opts:
                    filing_opts[label] = row.to_dict()

            selected_labels = st.multiselect(
                f"Pick filings to analyze ({len(filing_opts)} with PDFs available)",
                list(filing_opts.keys()),
                default=list(filing_opts.keys())[:3],
                max_selections=10,
                help="Max 10 filings per live run",
            )

            if selected_labels:
                if st.button("🤖 Run Live Analysis", type="primary", use_container_width=False):
                    from processors.ai_analyzer import analyze_filing
                    success_count = 0
                    for label in selected_labels:
                        row  = filing_opts[label]
                        fid  = str(row.get("filing_id", ""))
                        subj = str(row.get("subject", ""))
                        try:
                            with st.spinner(f"Analyzing: {label[:55]}..."):
                                pdf_path = fetch_pdf_for_analysis(
                                    fid,
                                    str(row.get("pdf_url", "")),
                                    str(row.get("local_pdf", "")),
                                )
                                if pdf_path:
                                    result = analyze_filing(
                                        pdf_path  = pdf_path,
                                        company   = str(row.get("company", selected_symbol)),
                                        subject   = subj,
                                        date_str  = str(row.get("broadcast_dt", ""))[:10],
                                        filing_id = fid,
                                    )
                                    if result:
                                        success_count += 1
                                        with st.expander(f"✅ {label[:60]}", expanded=True):
                                            _render_analysis(result, _key=fid)
                                    else:
                                        st.error(f"AI analysis returned no result for: {label[:50]}")
                                else:
                                    st.warning(f"PDF not available (local/Drive/NSE): {label[:50]}")
                        except Exception as _e:
                            st.error(f"❌ Error analyzing '{label[:50]}': {_e}")
                    if success_count:
                        st.success(f"✅ Analyzed {success_count}/{len(selected_labels)} filings successfully.")


# ──────────────────────────────────────────────────────────────────
# TAB 4 — ORDERBOOK INTELLIGENCE
# ──────────────────────────────────────────────────────────────────

_OB_GRADE_COLOR = {"A": "#3fb950", "B": "#58a6ff", "C": "#ffa726", "D": "#f85149"}
_OB_ASSESS_COLOR = {"bullish": "#3fb950", "neutral": "#8b949e", "bearish": "#f85149"}
_OB_ACTION_ICON = {
    "strong_buy": "🚀", "buy": "📈", "hold": "⏸", "reduce": "📉", "sell": "🔴"
}
_ENERGY_COLORS = {
    "solar": "#ffa726", "wind": "#58a6ff", "thermal": "#f85149",
    "hydro": "#3fb950",  "gas": "#d29922",  "oil": "#8b949e", "mixed": "#79c0ff",
}

with tab_ob:
    st.markdown("### 📦 Orderbook Intelligence")
    st.caption(
        "Extracts all contracts, PPAs, and order wins from every PDF — "
        "clusters them by size & type — detects trends & momentum — "
        "AI reasons about contract quality and investment implications."
    )

    from processors.orderbook_engine import (
        batch_extract_orderbook, detect_trends,
        ai_orderbook_reasoning, save_orderbook, load_orderbook,
    )

    # ── Load cached orderbook ──────────────────────────────────────
    ob_df, cached_trends, cached_reasoning = load_orderbook(selected_symbol)
    has_cache = not ob_df.empty

    col_run, col_rerun, col_clear, col_info = st.columns([2, 2, 1, 3])
    with col_run:
        run_ob = st.button(
            "▶ Build Orderbook" if not has_cache else "✅ Cached — Rebuild",
            type="primary" if not has_cache else "secondary",
            help="Processes all PDFs for this company to extract contracts & orders",
        )
    with col_rerun:
        run_reasoning = st.button(
            "🤖 Refresh AI Analysis",
            help="Re-runs the AI reasoning on existing orderbook data",
            disabled=ob_df.empty,
        )
    with col_clear:
        if st.button("🗑", help="Clear cached orderbook and start fresh"):
            from pathlib import Path
            import shutil
            from config import DATA_DIR
            ob_dir = DATA_DIR / "orderbook"
            for f in ob_dir.glob(f"{selected_symbol}_*"):
                f.unlink(missing_ok=True)
            st.session_state.pop("ob_processing", None)
            st.session_state.pop("ob_offset", None)
            st.session_state.pop("ob_symbol", None)
            st.success("Cache cleared! Click ▶ Build Orderbook to rebuild.")
            st.rerun()
    with col_info:
        if has_cache:
            st.caption(
                f"Cached: {len(ob_df)} entries · "
                f"{ob_df['date'].min().date() if ob_df['date'].notna().any() else '?'} → "
                f"{ob_df['date'].max().date() if ob_df['date'].notna().any() else '?'}"
            )

    # ── Run extraction (auto-batched to avoid 60s Streamlit timeout) ───────────
    # Session state tracks which PDF index we are at across reruns
    OB_BATCH = 30  # process 30 PDFs per rerun (~45s, safely under timeout)

    if run_ob:
        # Reset progress counter on fresh button click
        st.session_state["ob_processing"] = True
        st.session_state["ob_offset"]     = 0
        st.session_state["ob_symbol"]     = selected_symbol

    # Auto-continue if a batch run is in progress for this symbol
    if st.session_state.get("ob_processing") and st.session_state.get("ob_symbol") == selected_symbol:
        offset = st.session_state.get("ob_offset", 0)
        total  = len(df)
        end    = min(offset + OB_BATCH, total)
        batch_df = df.iloc[offset:end]

        prog_bar = st.progress(
            int(offset / max(total, 1) * 100),
            text=f"Processing PDFs {offset+1}–{end} of {total}…"
        )
        status = st.empty()

        def _ob_progress(cur, tot, msg):
            overall_pct = int((offset + cur) / max(total, 1) * 100)
            prog_bar.progress(overall_pct, text=msg[:80])
            status.caption(msg)

        with st.spinner(f"Batch {offset+1}–{end} / {total} — extracting orderbook…"):
            batch_ob = batch_extract_orderbook(selected_symbol, batch_df, _ob_progress)

        # Merge batch result with any already-extracted entries
        if not batch_ob.empty:
            if not ob_df.empty:
                ob_df = pd.concat([ob_df, batch_ob], ignore_index=True).drop_duplicates()
            else:
                ob_df = batch_ob

        prog_bar.empty()
        status.empty()

        if end >= total:
            # All PDFs processed — run AI reasoning and save
            st.session_state["ob_processing"] = False
            if ob_df.empty:
                st.warning("No orderbook data found in any PDF.")
            else:
                cached_trends    = detect_trends(ob_df)
                cached_reasoning = ai_orderbook_reasoning(selected_symbol, ob_df, cached_trends)
                save_orderbook(selected_symbol, ob_df, cached_trends, cached_reasoning)
                st.success(f"✅ Fully extracted {len(ob_df)} orderbook entries from {total} PDFs!")
                st.rerun()
        else:
            # More batches to go — save partial, advance offset, auto-rerun
            st.session_state["ob_offset"] = end
            if not ob_df.empty:
                # Save partial so progress survives any crash
                ob_df.to_parquet(
                    __import__("pathlib").Path(__file__).parent.parent / "data" / "orderbook" / f"{selected_symbol}_orderbook.parquet",
                    index=False,
                )
            st.info(f"⏳ Batch done ({end}/{total} PDFs). Continuing automatically…")
            st.rerun()

    if run_reasoning and not ob_df.empty:
        with st.spinner("Running AI reasoning on orderbook..."):
            cached_trends    = detect_trends(ob_df)
            cached_reasoning = ai_orderbook_reasoning(selected_symbol, ob_df, cached_trends)
            save_orderbook(selected_symbol, ob_df, cached_trends, cached_reasoning)
        st.success("AI analysis refreshed!")
        st.rerun()

    if ob_df.empty:
        st.info(
            "Click **▶ Build Orderbook** to start. The engine will scan all available PDFs, "
            "extract numbers with context, cluster by size/type, and reason about trends."
        )
        st.stop()

    trends   = cached_trends
    reasoning = cached_reasoning

    # ── Summary metrics ───────────────────────────────────────────
    st.markdown("---")
    total_mw  = trends.get("total_mw", 0)
    total_inr = trends.get("total_inr_cr", 0)
    bull_pct  = round(trends.get("bullish_ratio", 0) * 100, 1)
    traj      = trends.get("growth_trajectory", "—")
    grade     = reasoning.get("investment_grade", "—")
    score     = reasoning.get("order_quality_score", 0)

    # Show MW only if energy sector data exists; otherwise show deal count
    has_mw_data = total_mw > 0
    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    for col, val, lbl in [
        (sm1, f"{total_mw:,.0f} MW" if has_mw_data else f"{len(ob_df[ob_df['value_inr_cr'].notna()]) if 'value_inr_cr' in ob_df.columns else 0} Deals", "Total Capacity Won" if has_mw_data else "Deals Extracted"),
        (sm2, f"₹{total_inr:,.0f} Cr",  "Total Contract Value"),
        (sm3, f"{len(ob_df):,}",         "Order Entries"),
        (sm4, f"{bull_pct}%",            "Bullish Signals"),
        (sm5, f"{score}/100",            "Quality Score"),
    ]:
        col.markdown(
            f'<div class="stat-card"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── AI Assessment Card ────────────────────────────────────────
    if not reasoning:
        st.warning("⚠️ AI reasoning not yet generated. Click **🤖 Refresh AI Analysis** to run it.")
    if reasoning:
        assess   = reasoning.get("overall_assessment", "neutral")
        action   = reasoning.get("recommended_action", "hold")
        a_color  = _OB_ASSESS_COLOR.get(assess, "#8b949e")
        g_color  = _OB_GRADE_COLOR.get(grade, "#8b949e")
        a_icon   = _OB_ACTION_ICON.get(action, "⏸")

        st.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin-bottom:16px">'
            f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">'
            f'<span style="font-size:32px;font-weight:800;color:{g_color}">Grade {grade}</span>'
            f'<span style="font-size:22px;font-weight:700;color:{a_color}">{assess.upper()}</span>'
            f'<span style="font-size:18px;color:#e6edf3">{a_icon} {action.replace("_"," ").upper()}</span>'
            f'<span style="color:#8b949e;font-size:12px;margin-left:auto">Trajectory: {traj.upper()}</span>'
            f'</div>'
            f'<p style="color:#e6edf3;margin:0 0 12px 0">{reasoning.get("executive_summary","")}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Framework Decision Tree ───────────────────────────
        dt = reasoning.get("decision_tree", {})
        if dt:
            st.markdown("**📋 6-Point Framework Analysis**")
            fw_rows = [
                ("1. Coverage Ratio",    dt.get("step1_coverage",   "—")),
                ("2. Market Cap Signal", dt.get("step2_market_signal","—")),
                ("3. Execution",         dt.get("step3_execution",  "—")),
                ("4. Client Quality",    dt.get("step4_client_quality","—")),
                ("5. Inflow Rate",       dt.get("step5_inflow",     "—")),
                ("6. Cash Quality",      dt.get("step6_cash",       "—")),
            ]
            for label, val in fw_rows:
                color = "#3fb950" if any(x in str(val).lower() for x in ["pass","high","real","growing","feasible"])                        else "#f85149" if any(x in str(val).lower() for x in ["fail","trap","depleting","accounting"])                        else "#ffa726"
                st.markdown(
                    f'<div style="background:#161b22;border-left:3px solid {color};padding:6px 12px;margin:4px 0;border-radius:0 6px 6px 0">'
                    f'<span style="color:#8b949e;font-size:12px">{label}</span><br>'
                    f'<span style="color:#e6edf3;font-size:13px">{val}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("")

        if reasoning.get("verdict"):
            st.success(f"**Verdict:** {reasoning['verdict']}")

        r1, r2 = st.columns(2)
        with r1:
            strengths = reasoning.get("key_strengths", [])
            concerns  = reasoning.get("key_concerns", [])
            if strengths:
                st.markdown("**✅ Strengths**")
                for s in strengths: st.markdown(f"- {s}")
            if concerns:
                st.markdown("**⚠️ Concerns**")
                for c in concerns: st.markdown(f"- {c}")
        with r2:
            cats  = reasoning.get("catalysts_to_watch", [])
            risks = reasoning.get("risks", [])
            if cats:
                st.markdown("**⚡ Catalysts**")
                for c in cats: st.markdown(f"🚀 {c}")
            if risks:
                st.markdown("**🔴 Risks**")
                for r in risks: st.markdown(f"⚠️ {r}")

        if reasoning.get("12m_outlook"):
            st.markdown("---")
            st.info(f"**12-Month Outlook:** {reasoning['12m_outlook']}")

    st.markdown("---")

    # ── Chart 1: Cumulative Orderbook (Stacked Area by Energy Type) ──
    monthly_mw = trends.get("monthly_mw", [])
    if monthly_mw:
        st.markdown("#### 📈 Cumulative Orderbook Buildup")
        mw_frame = pd.DataFrame(monthly_mw)
        mw_frame["month"] = pd.to_datetime(mw_frame["month"], errors="coerce")

        # Stacked area by energy type if available
        energy_by_month = (
            ob_df[ob_df["value_mw"].notna() & (ob_df["value_mw"] > 0)]
            .assign(month=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
            .groupby(["month", "energy_type"])["value_mw"]
            .sum()
            .reset_index()
        )

        fig_area = go.Figure()
        if not energy_by_month.empty:
            energy_by_month = energy_by_month.sort_values("month")
            for etype in energy_by_month["energy_type"].unique():
                sub = energy_by_month[energy_by_month["energy_type"] == etype]
                # Cumulative within type
                sub = sub.sort_values("month").copy()
                sub["cumulative"] = sub["value_mw"].cumsum()
                color = _ENERGY_COLORS.get(str(etype).lower(), "#8b949e")
                fig_area.add_trace(go.Scatter(
                    x=sub["month"], y=sub["cumulative"],
                    name=str(etype).title(),
                    mode="lines",
                    stackgroup="one",
                    fillcolor=color.replace("#", "rgba(") + ",0.6)" if color.startswith("#") else color,
                    line=dict(color=color, width=1.5),
                    hovertemplate="%{y:,.0f} MW<extra>%{fullData.name}</extra>",
                ))
        else:
            # Fallback: total cumulative only
            fig_area.add_trace(go.Scatter(
                x=mw_frame["month"], y=mw_frame["cumulative_mw"],
                name="Total MW",
                mode="lines",
                fill="tozeroy",
                line=dict(color="#ffa726", width=2),
            ))

        # Anomaly markers
        anomalies = trends.get("mw_anomalies", [])
        if anomalies:
            adf = pd.DataFrame(anomalies)
            adf["month"] = pd.to_datetime(adf["month"], errors="coerce")
            fig_area.add_trace(go.Scatter(
                x=adf["month"], y=adf["mw_added"],
                mode="markers",
                name="Anomaly (spike)",
                marker=dict(color="#f85149", size=12, symbol="star"),
                hovertemplate="%{y:,.0f} MW spike<extra>Anomaly</extra>",
            ))

        fig_area.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
            xaxis_title="Month", yaxis_title="Cumulative MW",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified", height=380,
            title=dict(text=f"{selected_symbol} — Cumulative Orderbook by Energy Type",
                       font=dict(color="#e6edf3")),
        )
        st.plotly_chart(fig_area, use_container_width=True)

    # ── Chart 2: Monthly Order Wins Bar ───────────────────────────
    if monthly_mw:
        st.markdown("#### 📊 Monthly Order Win Volume")
        mw_bar = pd.DataFrame(monthly_mw)
        mw_bar["month"] = pd.to_datetime(mw_bar["month"], errors="coerce")

        # Velocity as line overlay
        vel_data = pd.DataFrame(trends.get("mw_velocity", []))

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=mw_bar["month"], y=mw_bar["mw_added"],
            name="MW Won",
            marker_color="#ffa726",
            hovertemplate="%{y:,.0f} MW<extra>Monthly Win</extra>",
        ))
        if not vel_data.empty and "velocity" in vel_data.columns:
            vel_data["month"] = pd.to_datetime(vel_data["month"], errors="coerce")
            fig_bar.add_trace(go.Scatter(
                x=vel_data["month"], y=vel_data["velocity"],
                name="Momentum (Δ MW)",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color="#58a6ff", width=2, dash="dot"),
                marker=dict(size=5),
            ))
            fig_bar.update_layout(
                yaxis2=dict(
                    title="Momentum", overlaying="y", side="right",
                    showgrid=False, tickfont=dict(color="#58a6ff"),
                )
            )

        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
            xaxis_title="Month", yaxis_title="MW Won",
            barmode="group", height=320,
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Chart 3: INR Cr timeline ──────────────────────────────────
    monthly_inr = trends.get("monthly_inr", [])
    if monthly_inr:
        inr_frame = pd.DataFrame(monthly_inr)
        inr_frame["month"] = pd.to_datetime(inr_frame["month"], errors="coerce")
        fig_inr = go.Figure()
        fig_inr.add_trace(go.Bar(
            x=inr_frame["month"], y=inr_frame["inr_cr_added"],
            name="₹ Cr Won",
            marker_color="#3fb950",
            hovertemplate="₹%{y:,.0f} Cr<extra>Monthly</extra>",
        ))
        fig_inr.add_trace(go.Scatter(
            x=inr_frame["month"], y=inr_frame["cumulative_inr_cr"],
            name="Cumulative ₹ Cr",
            mode="lines",
            line=dict(color="#ffa726", width=2),
            yaxis="y2",
        ))
        fig_inr.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
            title="Monthly Contract Value (₹ Cr)",
            xaxis_title="Month", yaxis_title="₹ Cr Won",
            yaxis2=dict(title="Cumulative ₹ Cr", overlaying="y", side="right",
                        showgrid=False, tickfont=dict(color="#ffa726")),
            height=300, legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_inr, use_container_width=True)

    # ── Chart 4: Quarterly trend ──────────────────────────────────
    quarterly_data = trends.get("quarterly", [])
    if quarterly_data:
        q_df = pd.DataFrame(quarterly_data)
        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(
            x=q_df["quarter"], y=q_df["count"],
            name="Order Count",
            marker_color=["#3fb950" if b > c / 2 else "#f85149"
                          for b, c in zip(q_df.get("bullish", [0]*len(q_df)), q_df["count"])],
            hovertemplate="%{y} orders<extra>%{x}</extra>",
        ))
        if "total_mw" in q_df.columns:
            fig_q.add_trace(go.Scatter(
                x=q_df["quarter"], y=q_df["total_mw"],
                name="MW Trend",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color="#ffa726", width=2),
            ))
            fig_q.update_layout(
                yaxis2=dict(title="MW", overlaying="y", side="right",
                            showgrid=False, tickfont=dict(color="#ffa726"))
            )
        fig_q.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
            title="Quarterly Order Activity (green = majority bullish)",
            height=300, xaxis_title="Quarter",
        )
        st.plotly_chart(fig_q, use_container_width=True)

    # ── Chart 5: K-Means Cluster Scatter ─────────────────────────
    cluster_data = trends.get("clusters")
    if cluster_data:
        st.markdown("#### 🔬 Order Cluster Analysis (K-Means)")
        sil = cluster_data[0].get("silhouette", 0) if cluster_data else 0
        st.caption(
            f"Silhouette score: {sil:.3f} — higher = better cluster separation · "
            f"{len(cluster_data)} natural order clusters identified"
        )

        cl_rows = []
        for c in cluster_data:
            for entry in c.get("entries", []):
                cl_rows.append({
                    "label": c["label"],
                    "entry": entry[:60],
                    "avg_mw": c.get("avg_mw", 0) or 0,
                    "avg_inr_cr": c.get("avg_inr_cr", 0) or 0,
                    "bullish_pct": c.get("bullish_pct", 50),
                    "count": c["count"],
                })

        if cl_rows:
            cl_df = pd.DataFrame(cl_rows)
            fig_cl = px.scatter(
                cl_df,
                x="avg_inr_cr", y="avg_mw",
                color="label", size="count",
                hover_data=["entry", "bullish_pct"],
                title="Order Clusters — Size: order count · X: avg ₹ Cr · Y: avg MW",
                template="plotly_dark",
                color_discrete_sequence=["#ffa726", "#58a6ff", "#3fb950", "#f85149", "#d29922"],
            )
            fig_cl.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                xaxis_title="Avg Contract Value (₹ Cr)",
                yaxis_title="Avg Capacity (MW)",
                height=380,
            )
            st.plotly_chart(fig_cl, use_container_width=True)

        # Cluster summary cards
        ccols = st.columns(min(len(cluster_data), 4))
        for i, c in enumerate(cluster_data[:4]):
            g_col = "#3fb950" if c.get("bullish_pct", 50) >= 60 else "#f85149"
            ccols[i].markdown(
                f'<div class="stat-card">'
                f'<div style="font-size:13px;color:#58a6ff;font-weight:700">{c["label"]}</div>'
                f'<div style="color:#ffa726;font-size:18px;font-weight:800">{c["count"]} orders</div>'
                f'<div style="color:#e6edf3;font-size:12px">⚡ {c.get("avg_mw","?") or "?"} MW avg</div>'
                f'<div style="color:{g_col};font-size:12px">{c.get("bullish_pct","?")}% bullish</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Energy type breakdown ─────────────────────────────────────
    energy_data = trends.get("energy_breakdown", [])
    if energy_data:
        e_df = pd.DataFrame(energy_data)
        e_df = e_df[e_df["energy_type"].notna() & (e_df["energy_type"] != "")]
        if not e_df.empty:
            ec1, ec2 = st.columns(2)
            with ec1:
                fig_epie = px.pie(
                    e_df, values="count", names="energy_type",
                    title="Orders by Energy Type",
                    template="plotly_dark",
                    color="energy_type",
                    color_discrete_map=_ENERGY_COLORS,
                    hole=0.45,
                )
                fig_epie.update_layout(paper_bgcolor="#0e1117", height=300)
                st.plotly_chart(fig_epie, use_container_width=True)
            with ec2:
                if "total_mw" in e_df.columns:
                    fig_emw = px.bar(
                        e_df.sort_values("total_mw", ascending=False),
                        x="energy_type", y="total_mw",
                        title="MW Won by Energy Type",
                        template="plotly_dark",
                        color="energy_type",
                        color_discrete_map=_ENERGY_COLORS,
                    )
                    fig_emw.update_layout(
                        paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                        showlegend=False, height=300,
                    )
                    st.plotly_chart(fig_emw, use_container_width=True)

    st.markdown("---")

    # ── Individual Orders Table ───────────────────────────────────
    st.markdown("#### 📋 All Extracted Orders")
    ob_filter_type = st.multiselect(
        "Filter by type",
        ob_df["type"].dropna().unique().tolist(),
        default=[],
        key="ob_type_filter",
    )
    ob_filter_positive = st.toggle("Show only bullish signals", value=False, key="ob_bull")

    display_ob = ob_df.copy()
    if ob_filter_type:
        display_ob = display_ob[display_ob["type"].isin(ob_filter_type)]
    if ob_filter_positive:
        display_ob = display_ob[display_ob["is_positive"]]

    display_ob = display_ob.sort_values("date", ascending=False)
    st.caption(f"Showing {len(display_ob)} entries")

    for _, row in display_ob.head(80).iterrows():
        is_pos = bool(row.get("is_positive", True))
        icon   = "🟢" if is_pos else "🔴"
        etype  = str(row.get("energy_type", "")).title()
        ctype  = str(row.get("contract_type", "")).upper()
        val_str = (
            f"{row['value_numeric']:,.0f} {row['value_unit']}"
            if pd.notna(row.get("value_numeric")) else "—"
        )
        inr_str = f" · ₹{row['value_inr_cr']:,.0f} Cr" if pd.notna(row.get("value_inr_cr")) else ""
        mw_str  = f" · {row['value_mw']:,.0f} MW" if pd.notna(row.get("value_mw")) else ""

        with st.container(border=True):
            lc, rc = st.columns([5, 2])
            with lc:
                type_pill = (
                    f'<span style="background:#21262d;color:#79c0ff;padding:1px 8px;'
                    f'border-radius:8px;font-size:11px;margin-right:6px">{row["type"].replace("_"," ").title()}</span>'
                )
                st.markdown(
                    f'{icon} {type_pill}'
                    f'<span style="color:#e6edf3;font-weight:600">{str(row.get("description",""))[:80]}</span>',
                    unsafe_allow_html=True,
                )
                if row.get("reasoning"):
                    st.caption(f"💡 {row['reasoning']}")
            with rc:
                st.caption(f"📅 {str(row.get('date',''))[:10]}")
                st.caption(f"📊 {val_str}{mw_str}{inr_str}")
                if row.get("counterparty"):
                    st.caption(f"🤝 {row['counterparty']}")
                if etype and etype not in ["", "None", "Nan"]:
                    st.caption(f"⚡ {etype}" + (f" · {ctype}" if ctype and ctype != "NONE" else ""))


# ──────────────────────────────────────────────────────────────────
# TAB 5 — CHAT  (multi-agent, batch-processed, memory-backed)
# ──────────────────────────────────────────────────────────────────

with tab_chat:
    from agents.chat_agent import NSEChatAgent

    st.markdown("### 💬 Filing Intelligence Chat")
    st.caption(
        f"Ask anything about **{selected_symbol}** — answers are grounded in actual NSE filings. "
        "Context stored in Supermemory + Mem0 across sessions."
    )

    # ── Filing selector ───────────────────────────────────────────
    st.markdown("#### Select Filings as Context")
    use_all = st.toggle("Use ALL filings as context", value=True)

    if use_all:
        chat_filings = df.to_dict("records")  # ALL filings — AI relevance filter picks the best ones per question
        st.caption(f"Using ALL {len(chat_filings)} filings — AI selects the most relevant ones per question")
    else:
        # Let user pick individual filings
        filing_labels = {}
        for _, row in df.iterrows():
            dt_s  = str(row.get("broadcast_dt",""))[:10]
            label = f"{dt_s} — {str(row.get('subject','?'))[:60]}"
            filing_labels[label] = row.to_dict()

        selected_labels = st.multiselect(
            "Pick filings to include",
            list(filing_labels.keys()),
            default=list(filing_labels.keys())[:5],
        )
        chat_filings = [filing_labels[l] for l in selected_labels]
        st.caption(f"{len(chat_filings)} filing(s) selected as context")

    st.markdown("---")

    # ── Chat history (session state) ─────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_symbol" not in st.session_state:
        st.session_state.chat_symbol = ""

    # Reset chat if symbol changed
    if st.session_state.chat_symbol != selected_symbol:
        st.session_state.chat_history = []
        st.session_state.chat_symbol  = selected_symbol

    # Clear button
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

    # ── Display chat history ──────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                st.caption(
                    f"Model: {meta.get('model','?')} · "
                    f"Filings used: {meta.get('filings_used',0)} · "
                    f"Memory hit: {'✅' if meta.get('memory_hits') else '—'}"
                )
                if meta.get("sources"):
                    with st.expander("📂 Sources", expanded=False):
                        for s in meta["sources"]:
                            pdf_link = f" [↗]({s['pdf_url']})" if s.get("pdf_url") else ""
                            st.caption(f"• {s['date']} — {s['subject']}{pdf_link}")

    # ── Chat input ────────────────────────────────────────────────
    question = st.chat_input(
        f"Ask about {selected_symbol} filings — e.g. 'What are the key risks?' or 'Summarise recent acquisitions'"
    )

    if question:
        if not chat_filings:
            st.warning("Select at least one filing as context above.")
        else:
            # Show user message immediately
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(question)

            # Process with progress bar
            with st.chat_message("assistant", avatar="🤖"):
                total_steps  = len(chat_filings) + 2
                prog_bar     = st.progress(0, text="Initialising agents...")
                status_text  = st.empty()

                def update_progress(cur, tot, msg):
                    pct = int(cur / max(tot, 1) * 100)
                    prog_bar.progress(pct, text=msg)
                    status_text.caption(msg)

                agent = NSEChatAgent(selected_symbol)
                try:
                    result = agent.answer(
                        question   = question,
                        filings    = chat_filings,
                        progress   = update_progress,
                        batch_size = 5,
                    )
                    prog_bar.progress(100, text="Complete ✅")
                    status_text.empty()

                    answer_md = result["answer"]
                    st.markdown(answer_md)
                    st.caption(
                        f"Model: {result['model']} · "
                        f"Filings used: {result['filings_used']} · "
                        f"Memory hit: {'✅' if result['memory_hits'] else '—'}"
                    )
                    if result.get("sources"):
                        with st.expander("📂 Sources", expanded=False):
                            for s in result["sources"]:
                                pdf_link = f" [↗]({s['pdf_url']})" if s.get("pdf_url") else ""
                                st.caption(f"• {s['date']} — {s['subject']}{pdf_link}")

                    # Save to history
                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": answer_md,
                        "meta":    result,
                    })

                except Exception as e:
                    prog_bar.empty()
                    st.error(f"Agent error: {e}")
                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": f"❌ Error: {e}",
                        "meta":    {},
                    })
