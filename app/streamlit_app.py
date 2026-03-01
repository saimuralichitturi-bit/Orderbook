"""
NSE Corporate Filings Intelligence Dashboard
Hedge Fund grade — DeepSeek · Gemini · Groq · OpenRouter
"""

import json
import io
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
sys.path.append(str(Path(__file__).parent.parent))
from config import FILING_CATEGORIES, PARQUET_DIR, ANALYSIS_DIR

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

# Map NSE subject strings → primary category
_NSE_SUBJECT_MAP = {
    "financial result": "Results",
    "quarterly result": "Results",
    "annual result": "Results",
    "financial statement": "Results",
    "investor presentation": "Investor Presentation",
    "analyst": "Conference Call",
    "institutional investor meet": "Conference Call",
    "con. call": "Conference Call",
    "conference call": "Conference Call",
    "acquisition": "Acquisition",
    "merger": "Merger/Demerger",
    "demerger": "Merger/Demerger",
    "dividend": "Dividend",
    "buyback": "Buyback",
    "bonus": "Bonus",
    "rights issue": "Rights Issue",
    "capital structure": "Capital Structure",
    "esop": "Capital Structure",
    "esos": "Capital Structure",
    "new order": "New Order",
    "order win": "Order Win",
    "capacity expansion": "Capacity Expansion",
    "joint venture": "Joint Venture",
    "press release": "Press Release",
    "newspaper publication": "Press Release",
    "credit rating": "Credit Rating",
    "resolution": "Resolution",
    "agm": "Resolution",
    "egm": "Resolution",
    "board meeting": "Management Update",
    "management": "Management Update",
    "insider trading": "Regulatory",
    "sast": "Regulatory",
    "trading window": "Regulatory",
    "regulatory": "Regulatory",
    "ipo": "IPO",
    "sale or transfer": "Sale or Transfer of Assets",
    "assets": "Sale or Transfer of Assets",
    "solar": "Solar",
    "tax": "Tax Related",
    "incident": "Incident",
    "update": "Update",
    "general update": "Update",
    "loss of share": "Other",
    "duplicate share": "Other",
}


def categorise_subject(subject: str) -> str:
    s = str(subject).lower()
    for keyword, cat in _NSE_SUBJECT_MAP.items():
        if keyword in s:
            return cat
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
    df["primary_category"] = df["subject"].apply(categorise_subject)
    return df


@st.cache_data(ttl=600)
def load_symbol_parquet(symbol: str, category: str) -> pd.DataFrame:
    safe_cat = category.replace(" ", "_").replace("/", "_")
    path = PARQUET_DIR / f"{symbol}_{safe_cat}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["broadcast_dt"] = pd.to_datetime(df["broadcast_dt"], errors="coerce")
        df["primary_category"] = df["subject"].apply(categorise_subject)
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

def _render_analysis(a: dict):
    sentiment = a.get("sentiment", "neutral")
    score     = float(a.get("sentiment_score", 0) or 0)
    signal    = a.get("action_signal", "watch")
    conf      = float(a.get("confidence", 0) or 0)
    model     = a.get("model_used", "")

    st.markdown(
        f'{_sent_badge(sentiment)} &nbsp; {_sig_badge(signal)} &nbsp;'
        f'<span style="color:#8b949e;font-size:12px">Conf: {conf:.0%} · {model}</span>',
        unsafe_allow_html=True,
    )
    if a.get("summary"):
        st.info(a["summary"])

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
    st.plotly_chart(fig, use_container_width=True)


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
  .stat-card .val { font-size:36px; font-weight:800; color:#58a6ff; }
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

if not selected_symbol:
    st.markdown("## 📊 NSE Corporate Filings Intelligence")
    st.caption(f"Last updated: {date.today().strftime('%d %B %Y')} · Updates Daily")
    st.markdown("---")

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

    total_filings = len(all_df) if not all_df.empty else 0
    total_symbols = all_df["symbol"].nunique() if not all_df.empty else 0
    analyzed_count = int(all_df["ai_summary"].notna().sum()) if ("ai_summary" in all_df.columns and not all_df.empty) else 0

    # ── Summary cards
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(f"""<div class="stat-card"><div class="val">{total_filings:,}</div><div class="lbl">Total Filings</div></div>""", unsafe_allow_html=True)
    mc2.markdown(f"""<div class="stat-card"><div class="val">{total_symbols}</div><div class="lbl">Companies</div></div>""", unsafe_allow_html=True)
    mc3.markdown(f"""<div class="stat-card"><div class="val">{analyzed_count:,}</div><div class="lbl">AI Analyzed</div></div>""", unsafe_allow_html=True)
    pct = f"{analyzed_count/total_filings*100:.1f}%" if total_filings > 0 else "—"
    mc4.markdown(f"""<div class="stat-card"><div class="val" style="color:#3fb950">{pct}</div><div class="lbl">Analysis Coverage</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    if all_df.empty:
        st.info("No local parquet data found. Run the pipeline first: `python run_pipeline.py`")
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
tab_filings, tab_charts, tab_ai, tab_chat = st.tabs(["📋 Filings", "📈 Charts", "🤖 AI Insights", "💬 Chat"])


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
                analysis = load_analysis(filing_id)
                if not analysis:
                    # PDF must be on Drive — we can only show the link
                    if has_pdf:
                        st.info("PDF is on Google Drive. Download it locally and re-run the pipeline to generate AI analysis.")
                    else:
                        st.info("No PDF available for this filing.")
                if analysis:
                    with st.expander(f"AI Analysis — {subject[:55]}", expanded=True):
                        _render_analysis(analysis)


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
# TAB 3 — AI INSIGHTS
# ──────────────────────────────────────────────────────────────────

with tab_ai:
    analyzed_df = df[df["ai_summary"].notna()] if "ai_summary" in df.columns else pd.DataFrame()

    if analyzed_df.empty:
        st.info("No AI analyses yet for this view. Click **🤖 Analyze** on any filing in the Filings tab.")
        st.code("python run_pipeline.py " + selected_symbol, language="bash")
    else:
        # Summary sentiment bar
        sent_dist = analyzed_df["ai_sentiment"].value_counts().reset_index()
        sent_dist.columns = ["sentiment","count"]
        fig_s = px.bar(sent_dist, x="sentiment", y="count",
                       color="sentiment",
                       color_discrete_map={"bullish":"#3fb950","bearish":"#f85149","neutral":"#8b949e"},
                       template="plotly_dark", title="Sentiment Distribution")
        fig_s.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                            showlegend=False, height=220)
        st.plotly_chart(fig_s, use_container_width=True)

        for _, row in analyzed_df.iterrows():
            icon  = "📈" if row.get("ai_sentiment")=="bullish" else "📉" if row.get("ai_sentiment")=="bearish" else "📋"
            dt_s  = str(row.get("broadcast_dt",""))[:10]
            label = f"{icon} {row.get('subject','—')} — {dt_s}"
            with st.expander(label[:80], expanded=False):
                _render_analysis({
                    "summary":         row.get("ai_summary",""),
                    "sentiment":       row.get("ai_sentiment",""),
                    "sentiment_score": row.get("ai_sentiment_score", 0),
                    "action_signal":   row.get("ai_signal",""),
                    "key_highlights":  json.loads(row.get("ai_highlights","[]") or "[]"),
                    "risk_factors":    json.loads(row.get("ai_risks","[]") or "[]"),
                    "financial_data":  json.loads(row.get("ai_financial","{}") or "{}"),
                    "confidence":      row.get("ai_confidence",0),
                    "model_used":      row.get("ai_model",""),
                })


# ──────────────────────────────────────────────────────────────────
# TAB 4 — CHAT  (multi-agent, batch-processed, memory-backed)
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
        chat_filings = df.head(50).to_dict("records")  # cap at 50 for speed
        st.caption(f"Using {len(chat_filings)} most recent filings as context")
    else:
        # Let user pick individual filings
        filing_labels = {}
        for _, row in df.head(80).iterrows():
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
