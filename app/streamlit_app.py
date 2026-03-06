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

    from processors.sector_framework import (
        get_sector, fetch_fundamentals, compute_framework,
        score_all_filings,
        SECTOR_LABELS, ALTERNATIVE_METRICS, ORDERBOOK_SECTORS, PARTIAL_SECTORS
    )
    from processors.orderbook_engine import (
        batch_extract_orderbook, save_orderbook, load_orderbook,
    )

    sector       = get_sector(selected_symbol)
    sector_label = SECTOR_LABELS.get(sector, sector)
    applicable   = sector in ORDERBOOK_SECTORS
    partial      = sector in PARTIAL_SECTORS

    # ── Sector badge ──────────────────────────────────────────────
    badge_color = "#3fb950" if applicable else "#ffa726" if partial else "#8b949e"
    st.markdown(
        f'<span style="background:{badge_color};color:#0d1117;padding:3px 10px;'
        f'border-radius:6px;font-size:12px;font-weight:700">{sector_label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Not applicable — show what to use instead ─────────────────
    if not applicable and not partial:
        alt = ALTERNATIVE_METRICS.get(sector, "Sector-specific metrics")
        st.warning(
            f"**Orderbook framework does not apply to {sector_label} companies.**\n\n"
            f"For {selected_symbol}, use: **{alt}**"
        )
        st.info(
            "The Coverage Ratio and Market Cap % framework works for: "
            "EPC, Infrastructure, Defence, Capital Goods, Shipbuilding, Manufacturing. "
            f"{selected_symbol} is a {sector_label} company — "
            "different fundamentals drive value here."
        )
        st.stop()

    # ── Load / build orderbook ────────────────────────────────────
    ob_df, _, _ = load_orderbook(selected_symbol)
    has_cache   = not ob_df.empty

    col_run, col_clear, col_info = st.columns([2, 1, 4])
    with col_run:
        run_ob = st.button(
            "▶ Build Orderbook" if not has_cache else "✅ Cached — Rebuild",
            type="primary" if not has_cache else "secondary",
        )
    with col_clear:
        if st.button("🗑", help="Clear cache and rebuild"):
            from pathlib import Path
            from config import DATA_DIR
            for f in (DATA_DIR / "orderbook").glob(f"{selected_symbol}_*"):
                f.unlink(missing_ok=True)
            st.session_state.pop("ob_processing", None)
            st.session_state.pop("ob_offset", None)
            st.session_state.pop("ob_symbol", None)
            st.rerun()
    with col_info:
        if has_cache:
            st.caption(f"Cached: {len(ob_df)} entries")

    # ── Auto-batch extraction (30 PDFs/run to beat 60s timeout) ───
    OB_BATCH = 30
    if run_ob:
        st.session_state["ob_processing"] = True
        st.session_state["ob_offset"]     = 0
        st.session_state["ob_symbol"]     = selected_symbol

    if st.session_state.get("ob_processing") and st.session_state.get("ob_symbol") == selected_symbol:
        offset   = st.session_state.get("ob_offset", 0)
        total    = len(df)
        end      = min(offset + OB_BATCH, total)
        batch_df = df.iloc[offset:end]

        prog = st.progress(int(offset / max(total,1) * 100),
                           text=f"Processing PDFs {offset+1}–{end} of {total}…")

        def _prog(cur, tot, msg):
            prog.progress(int((offset+cur)/max(total,1)*100), text=msg[:80])

        with st.spinner(f"Extracting {offset+1}–{end} / {total}…"):
            batch_ob = batch_extract_orderbook(selected_symbol, batch_df, _prog, sector=sector)

        if not batch_ob.empty:
            ob_df = pd.concat([ob_df, batch_ob], ignore_index=True).drop_duplicates() if not ob_df.empty else batch_ob

        prog.empty()

        if end >= total:
            st.session_state["ob_processing"] = False
            if not ob_df.empty:
                save_orderbook(selected_symbol, ob_df, {}, {})
                st.success(f"✅ Extracted {len(ob_df)} entries from {total} PDFs.")
                st.rerun()
            else:
                st.warning("No orderbook entries found in PDFs.")
        else:
            st.session_state["ob_offset"] = end
            st.info(f"⏳ {end}/{total} done — continuing…")
            st.rerun()

    if ob_df.empty:
        st.info("Click **▶ Build Orderbook** to extract order data from all PDFs.")
        st.stop()

    # ── Fetch fundamentals & compute framework ────────────────────
    with st.spinner("Fetching live fundamentals (Revenue, MCap)…"):
        fundamentals = fetch_fundamentals(selected_symbol)

    # Exclude FLAGGED entries from totals (verification layer)
    if "include_in_totals" in ob_df.columns:
        clean_df = ob_df[ob_df["include_in_totals"] != False]
    else:
        clean_df = ob_df
    total_inr = float(clean_df["value_inr_cr"].dropna().sum()) if "value_inr_cr" in clean_df.columns else 0

    # Verification summary
    if "verification_status" in ob_df.columns:
        v_counts = ob_df["verification_status"].value_counts().to_dict()
        trusted   = v_counts.get("TRUSTED", 0)
        verified  = sum(v for k, v in v_counts.items() if "VERIFIED" in k and "UN" not in k)
        flagged   = sum(v for k, v in v_counts.items() if "FLAGGED" in k)
        unverified = v_counts.get("UNVERIFIED ⚠️", 0)
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
            f'padding:10px 16px;margin-bottom:12px;font-size:13px">'
            f'<b style="color:#e6edf3">Verification:</b>&nbsp;&nbsp;'
            f'<span style="color:#3fb950">✅ {trusted} Trusted</span>&nbsp;&nbsp;'
            f'<span style="color:#3fb950">✅ {verified} Verified</span>&nbsp;&nbsp;'
            f'<span style="color:#ffa726">⚠️ {unverified} Unverified</span>&nbsp;&nbsp;'
            f'<span style="color:#f85149">❌ {flagged} Flagged</span>'
            f'&nbsp;&nbsp;<span style="color:#8b949e;font-size:11px">'
            f'(Groq extracts → Gemini cross-checks confidence &lt; 0.8)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    fw = compute_framework(selected_symbol, total_inr, fundamentals)

    rev  = fundamentals.get("annual_revenue_cr", 0)
    mcap = fundamentals.get("market_cap_cr", 0)

    # ── Partial sector warning ────────────────────────────────────
    if partial and not applicable:
        st.warning(
            f"⚠️ **Partial applicability** — {sector_label} companies have limited orderbook signals. "
            f"Results below are indicative only."
        )
        if sector == "conglomerate":
            st.info("**Segment Rule:** Compare order value against SEGMENT revenue, not total company revenue. "
                    "Then identify suppliers to that segment — those smaller companies re-rate.")

    st.markdown("---")

    # ── THE TWO FORMULAS ──────────────────────────────────────────
    st.markdown("#### 📐 The Two Anchor Formulas")

    f1, f2 = st.columns(2)

    # Formula 1 — Coverage Ratio
    with f1:
        cov  = fw.get("coverage_ratio")
        csig = fw.get("coverage_signal", "NO_DATA")
        cclr = {"green": "#3fb950", "orange": "#ffa726", "red": "#f85149", "grey": "#8b949e"}.get(fw.get("coverage_color","grey"), "#8b949e")
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px">'
            f'<div style="color:#8b949e;font-size:12px;margin-bottom:6px">FORMULA 1 — Coverage Ratio</div>'
            f'<div style="font-size:13px;color:#8b949e;margin-bottom:8px">Orderbook ÷ Annual Revenue</div>'
            f'<div style="font-size:11px;color:#8b949e">₹{total_inr:,.0f} Cr ÷ ₹{rev:,.0f} Cr</div>'
            f'<div style="font-size:36px;font-weight:800;color:{cclr};margin:8px 0">'
            f'{"N/A" if cov is None else f"{cov:.2f}x"}</div>'
            f'<div style="font-size:14px;font-weight:700;color:{cclr}">{csig}</div>'
            f'<div style="font-size:11px;color:#8b949e;margin-top:8px">'
            f'&lt;1x WEAK · 1-2x STABLE · 2-4x GOOD · 4-6x STRONG · &gt;6x INVESTIGATE</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Formula 2 — Market Cap Coverage
    with f2:
        mcp  = fw.get("market_cap_pct")
        msig = fw.get("market_signal", "NO_DATA")
        mclr = {"green": "#3fb950", "orange": "#ffa726", "red": "#f85149", "grey": "#8b949e"}.get(fw.get("market_color","grey"), "#8b949e")
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px">'
            f'<div style="color:#8b949e;font-size:12px;margin-bottom:6px">FORMULA 2 — Market Cap Coverage</div>'
            f'<div style="font-size:13px;color:#8b949e;margin-bottom:8px">Orderbook ÷ Market Cap × 100</div>'
            f'<div style="font-size:11px;color:#8b949e">₹{total_inr:,.0f} Cr ÷ ₹{mcap:,.0f} Cr × 100</div>'
            f'<div style="font-size:36px;font-weight:800;color:{mclr};margin:8px 0">'
            f'{"N/A" if mcp is None else f"{mcp:.1f}%"}</div>'
            f'<div style="font-size:14px;font-weight:700;color:{mclr}">{msig}</div>'
            f'<div style="font-size:11px;color:#8b949e;margin-top:8px">'
            f'&lt;10% NOISE · 10-30% MODERATE · 30-100% SIGNIFICANT · &gt;100% DEEP VALUE</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── VERDICT ───────────────────────────────────────────────────
    verdict = fw.get("verdict", "")
    verdict_map = {
        "RE_RATING_CANDIDATE": ("#3fb950", "🚀 RE-RATING CANDIDATE", "Both checks pass — orderbook is large relative to revenue AND market cap hasn't priced it in. Stock re-rating likely."),
        "WATCH_FOR_RE_RATING":  ("#ffa726", "👀 WATCH FOR RE-RATING",  "Market cap coverage is significant but coverage ratio needs improvement. Monitor order inflow rate."),
        "HEALTHY_PIPELINE":     ("#58a6ff", "✅ HEALTHY PIPELINE",      "Strong coverage ratio — good revenue visibility. But market may already have priced this in."),
        "NOISE_OR_WEAK":        ("#f85149", "❌ NOISE / WEAK",          "Orderbook is small relative to both revenue and market cap. Individual orders won't move the stock."),
        "FRAMEWORK_NOT_APPLICABLE": ("#8b949e", "⚠️ N/A", ""),
    }
    v_color, v_label, v_text = verdict_map.get(verdict, ("#8b949e", verdict, ""))
    if v_text:
        st.markdown(
            f'<div style="background:#161b22;border-left:4px solid {v_color};padding:14px 20px;border-radius:0 8px 8px 0;margin:8px 0">'
            f'<div style="font-size:16px;font-weight:700;color:{v_color}">{v_label}</div>'
            f'<div style="color:#e6edf3;font-size:13px;margin-top:4px">{v_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Cash Conversion ───────────────────────────────────────────
    ccr  = fw.get("cash_conversion_ratio")
    csig = fw.get("cash_signal", "NO_DATA")
    if ccr is not None:
        cc_color = "#3fb950" if csig == "REAL" else "#ffa726" if csig == "MODERATE" else "#f85149"
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px;margin-top:8px">'
            f'<span style="color:#8b949e;font-size:12px">Cash Conversion (OCF/EBITDA): </span>'
            f'<span style="font-weight:700;color:{cc_color}">{ccr:.2f} → {csig}</span>'
            f'<span style="color:#8b949e;font-size:11px"> · &gt;0.8=Real profits · &lt;0.5=Accounting only</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Raw extracted entries ─────────────────────────────────────
    st.markdown(f"#### 📋 Extracted Order Entries ({len(ob_df)})")
    st.caption("Raw data extracted from NSE filings PDFs. Values are AI-extracted — verify against official announcements.")

    show_df = ob_df.copy()
    display_cols = [c for c in ["date","description","value_numeric","value_unit","value_inr_cr","counterparty","is_positive","confidence","verification_status"] if c in show_df.columns]
    if "value_inr_cr" in show_df.columns:
        show_df["value_inr_cr"] = show_df["value_inr_cr"].apply(lambda x: f"₹{x:,.0f} Cr" if pd.notna(x) and x > 0 else "—")
    if "confidence" in show_df.columns:
        show_df["confidence"] = show_df["confidence"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
    st.dataframe(show_df[display_cols].sort_values("date", ascending=False) if "date" in show_df.columns else show_df[display_cols],
                 use_container_width=True, hide_index=True)

    # ── Per-filing re-rating impact ────────────────────────────────
    if applicable or partial:
        st.markdown("---")
        st.markdown("#### 🎯 Per-Filing Impact (Which orders actually matter?)")
        st.caption("Each order scored individually. Only entries with ₹ value shown. Sorted by MCap impact.")

        scored = score_all_filings(selected_symbol, ob_df, fundamentals)
        scored = [s for s in scored if (s.get("mcap_impact_pct") or 0) > 0]

        if scored:
            rows = []
            for s in scored:
                color_map = {"green": "🟢", "orange": "🟡", "blue": "🔵", "grey": "⚫"}
                icon = color_map.get(s["impact_color"], "⚫")
                rows.append({
                    "Date":        s["date"],
                    "Description": s["description"][:60],
                    "Value (₹ Cr)": f"₹{s['order_inr_cr']:,.0f}" if s["order_inr_cr"] else "—",
                    "MCap Impact":  f"{s['mcap_impact_pct']:.2f}%" if s["mcap_impact_pct"] else "—",
                    "Revenue %":    f"{s['revenue_pct']:.2f}%" if s["revenue_pct"] else "—",
                    "Signal":       f"{icon} {s['impact_label']}",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

            rerating = [s for s in scored if s["is_rerating"]]
            if rerating:
                st.success(f"🚀 **{len(rerating)} RE-RATING EVENT(s) detected** — orders with MCap impact ≥ 5%")
                for r in rerating:
                    st.markdown(f"- **{r['date']}** · {r['description']} · MCap impact **{r['mcap_impact_pct']:.2f}%**")
            else:
                st.info("No single order crosses the 5% MCap impact threshold. Framework says: NOISE.")
        else:
            st.info("No ₹-valued entries found for impact scoring.")



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
