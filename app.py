
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==================================================
# FinSight AI | Market Intelligence Dashboard
# ==================================================

st.set_page_config(
    page_title="FinSight AI | Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# COLORS
# --------------------------------------------------

BG = "#0A0D18"
PANEL = "#111827"
BORDER = "#263248"
TEXT = "#F3F6FC"
MUTED = "#96A3B8"
VIOLET = "#9B8CFF"
CYAN = "#57D7E8"
GREEN = "#49D7A0"
RED = "#FF7285"
BLUE = "#76A9FF"

# --------------------------------------------------
# STOCKS AND MARKET INDICES
# --------------------------------------------------

stock_options = {
    # US Technology
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Meta Platforms": "META",
    "NVIDIA": "NVDA",
    "Tesla": "TSLA",
    "Netflix": "NFLX",
    "AMD": "AMD",
    "Intel": "INTC",
    "Broadcom": "AVGO",
    "Qualcomm": "QCOM",
    "Oracle": "ORCL",
    "Salesforce": "CRM",
    "Adobe": "ADBE",
    "Palantir": "PLTR",
    "Uber": "UBER",
    "Shopify": "SHOP",

    # US Consumer, Finance and Healthcare
    "Walmart": "WMT",
    "Coca-Cola": "KO",
    "McDonald's": "MCD",
    "JPMorgan Chase": "JPM",
    "Visa": "V",
    "Mastercard": "MA",
    "Berkshire Hathaway": "BRK-B",
    "Johnson & Johnson": "JNJ",
    "Eli Lilly": "LLY",
    "ExxonMobil": "XOM",
    "Nike": "NKE",
    "Disney": "DIS",

    # Indian IT
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Wipro": "WIPRO.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "LTIMindtree": "LTIM.NS",

    # Indian Banking and Finance
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",

    # Indian Energy and Infrastructure
    "Reliance Industries": "RELIANCE.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",
    "ONGC": "ONGC.NS",
    "Coal India": "COALINDIA.NS",

    # Indian Automobile
    "Tata Motors Passenger Vehicles": "TMPV.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Eicher Motors": "EICHERMOT.NS",

    # Indian Consumer and Healthcare
    "ITC": "ITC.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Titan Company": "TITAN.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Nestle India": "NESTLEIND.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr Reddys Laboratories": "DRREDDY.NS",
    "Eternal": "ETERNAL.NS",
    "Trent": "TRENT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",

    # Market Indices
    "NIFTY 50": "^NSEI",
    "NIFTY Bank": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "SENSEX": "^BSESN",
    "Dow Jones": "^DJI",
    "NASDAQ Composite": "^IXIC",
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "Russell 2000": "^RUT",
    "VIX Volatility Index": "^VIX"
}

period_options = {
    "1 Month": "1mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y"
}

# --------------------------------------------------
# ASSET HELPERS
# --------------------------------------------------

def get_currency(ticker):
    """Return the display unit for an asset."""
    ticker = str(ticker)

    if ticker.startswith("^"):
        return "POINTS"

    if ticker.endswith((".NS", ".BO")):
        return "INR"

    return "USD"


def price_text(value, ticker):
    """Format prices and safely handle missing values."""
    try:
        value = float(value)

        if not np.isfinite(value):
            return "N/A"

    except (TypeError, ValueError):
        return "N/A"

    currency = get_currency(ticker)

    if currency == "INR":
        return f"₹{value:,.2f}"

    if currency == "USD":
        return f"${value:,.2f}"

    return f"{value:,.2f} pts"


def price_axis(ticker):
    currency = get_currency(ticker)

    if currency == "INR":
        return "Price (INR ₹)"

    if currency == "USD":
        return "Price (USD $)"

    return "Index points"


def unit_label(ticker):
    currency = get_currency(ticker)

    if currency == "INR":
        return "INR"

    if currency == "USD":
        return "USD"

    return "Points"


def is_indian(ticker):
    return str(ticker).endswith((".NS", ".BO"))


def is_index(ticker):
    return str(ticker).startswith("^")


def safe_float(value):
    try:
        result = float(value)

        if np.isfinite(result):
            return result

        return np.nan

    except (TypeError, ValueError):
        return np.nan


# --------------------------------------------------
# CUSTOM DESIGN AND FONTS
# --------------------------------------------------

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    :root {
        color-scheme: dark;
    }

    html, body, [class*="css"],
    [data-testid="stAppViewContainer"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
          radial-gradient(ellipse at 12% 0%,
          rgba(111,91,255,.11), transparent 34%),
          radial-gradient(ellipse at 95% 10%,
          rgba(42,202,191,.07), transparent 28%),
          #0A0D18;
        color: #F3F6FC;
    }

    [data-testid="stHeader"] {
        background: rgba(10,13,24,.82);
    }

    [data-testid="stSidebar"] {
        background: #0E1422;
        border-right: 1px solid #263248;
    }

    [data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    h1, h2, h3, h4,
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -.035em;
        color: #F3F6FC !important;
    }

    h1 {
        font-size: clamp(2rem, 4vw, 3.1rem) !important;
    }

    h2 {
        font-size: 1.55rem !important;
    }

    h3 {
        font-size: 1.2rem !important;
    }

    p, label, .stMarkdown,
    [data-testid="stCaptionContainer"] {
        color: #B7C2D4;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg,#151E30,#101726);
        border: 1px solid #2A3850;
        border-radius: 16px;
        padding: 16px 18px;
        min-height: 110px;
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
    }

    [data-testid="stMetricLabel"] {
        color: #96A3B8 !important;
        font-size: .82rem !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        font-size: clamp(1.15rem, 2vw, 1.75rem) !important;
        font-weight: 700;
    }

    .stButton > button {
        width: 100%;
        min-height: 46px;
        border-radius: 12px;
        border: 1px solid #35425A;
        background: #151D2E;
        color: #F3F6FC;
        font-weight: 700;
        transition: all .18s ease;
    }

    .stButton > button:hover {
        border-color: #9B8CFF;
        color: white;
        background: #20243D;
        transform: translateY(-1px);
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #263248;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stExpander"] {
        background: rgba(17,24,39,.75);
        border: 1px solid #263248;
        border-radius: 12px;
    }

    div[data-baseweb="select"] > div {
        background: #151D2E;
        border-color: #35425A;
        border-radius: 10px;
    }

    hr {
        border-color: #263248;
        margin: 1.2rem 0;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 28px 32px;
        border-radius: 22px;
        border: 1px solid rgba(155,140,255,.28);
        background:
          radial-gradient(circle at 90% 10%,
          rgba(87,215,232,.15), transparent 27%),
          linear-gradient(125deg,
          rgba(35,31,72,.94),
          rgba(17,31,48,.96) 65%,
          rgba(13,43,46,.92));
        margin: 4px 0 22px;
        box-shadow: 0 18px 48px rgba(0,0,0,.18);
    }

    .hero h1 {
        margin: 8px 0;
        line-height: 1.08;
    }

    .hero p {
        max-width: 720px;
        margin: 0;
        color: #C2CCDE;
        font-size: 1rem;
    }

    .eyebrow {
        color: #AFA4FF;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .18em;
        text-transform: uppercase;
    }

    .hero-pills {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .pill {
        display: inline-block;
        border: 1px solid rgba(196,205,228,.2);
        border-radius: 999px;
        padding: 6px 11px;
        color: #D5DDF0;
        background: rgba(255,255,255,.045);
        font-size: .76rem;
    }

    .section-kicker {
        color: #9B8CFF;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-size: .7rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .section-note {
        color: #96A3B8;
        font-size: .9rem;
        margin-top: -5px;
    }

    .company-strip {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 14px 18px;
        background: linear-gradient(130deg,#151E30,#101725);
        border: 1px solid #2A3850;
        border-radius: 16px;
        margin-bottom: 18px;
    }

    .company-strip img {
        width: 42px;
        height: 42px;
        object-fit: contain;
        border-radius: 12px;
        padding: 7px;
        background: #F7F8FC;
    }

    .company-name {
        color: #F3F6FC;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
    }

    .company-sub {
        color: #96A3B8;
        font-size: .83rem;
    }

    .home-card {
        height: 100%;
        padding: 22px;
        border-radius: 17px;
        background: linear-gradient(145deg,#151E30,#101725);
        border: 1px solid #2A3850;
    }

    .home-icon {
        font-size: 1.5rem;
        margin-bottom: 12px;
    }

    .home-card h3 {
        margin-bottom: 8px;
    }

    .home-card p {
        color: #A9B6CB;
    }

    .footer-note {
        color: #718098;
        font-size: .78rem;
        text-align: center;
        padding: 14px 0 6px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# REUSABLE FUNCTIONS
# --------------------------------------------------

def style_chart(fig, height=360, y_title=None):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            color="#B7C2D4",
            size=12
        ),
        margin=dict(l=12, r=18, t=25, b=12),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#C8D2E2")
        ),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor=BORDER,
            tickfont=dict(color="#8D9BB1"),
            automargin=True
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor="rgba(148,163,184,.13)",
            zeroline=False,
            linecolor=BORDER,
            tickfont=dict(color="#8D9BB1"),
            automargin=True
        )
    )

    return fig


def add_line(fig, x, y, name, color, width=2.5, dash=None):
    line = dict(color=color, width=width)

    if dash:
        line["dash"] = dash

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=name,
            mode="lines",
            line=line,
            connectgaps=False,
            hovertemplate=(
                "%{x}<br>%{y:,.2f}<extra>" + name + "</extra>"
            )
        )
    )


def section_heading(title, subtitle=None, kicker=None):
    if kicker:
        st.markdown(
            f'<div class="section-kicker">{kicker}</div>',
            unsafe_allow_html=True
        )

    st.subheader(title)

    if subtitle:
        st.markdown(
            f'<div class="section-note">{subtitle}</div>',
            unsafe_allow_html=True
        )


def company_header(name, ticker):
    domain_map = {
        "Apple": "apple.com",
        "Microsoft": "microsoft.com",
        "Google": "google.com",
        "Amazon": "amazon.com",
        "Tesla": "tesla.com",
        "NVIDIA": "nvidia.com",
        "TCS": "tcs.com",
        "Infosys": "infosys.com",
        "Reliance Industries": "ril.com",
        "HDFC Bank": "hdfcbank.com",
        "ICICI Bank": "icicibank.com"
    }

    domain = (
        "finance.yahoo.com"
        if is_index(ticker)
        else domain_map.get(name, "finance.yahoo.com")
    )

    logo_url = (
        "https://www.google.com/s2/favicons"
        f"?domain={domain}&sz=128"
    )

    if is_index(ticker):
        market_name = "Market Index"
    elif is_indian(ticker):
        market_name = "Indian Market"
    else:
        market_name = "US Market"

    st.markdown(
        f'<div class="company-strip">'
        f'<img src="{logo_url}" alt="Asset logo">'
        f'<div>'
        f'<div class="company-name">{name}</div>'
        f'<div class="company-sub">'
        f'{ticker} &nbsp;·&nbsp; {market_name}'
        f'</div>'
        f'</div></div>',
        unsafe_allow_html=True
    )


def footer():
    st.markdown(
        '<div class="footer-note">'
        'FINSIGHT AI &nbsp;·&nbsp; Market insights '
        '&nbsp;·&nbsp; Not investment advice'
        '</div>',
        unsafe_allow_html=True
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker, period):
    """Fetch and cache historical data for five minutes."""
    try:
        data = yf.Ticker(ticker).history(
            period=period,
            auto_adjust=True,
            timeout=15
        )

        if data is None or data.empty or "Close" not in data.columns:
            return pd.DataFrame()

        data = data.copy()
        data = data[~data.index.duplicated(keep="last")]
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data = data.dropna(subset=["Close"])
        data = data[data["Close"] > 0]

        return data

    except Exception:
        return pd.DataFrame()


def calculate_rsi(close, period=14):
    """Calculate RSI and handle zero-gain or zero-loss windows."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(
        period,
        min_periods=period
    ).mean()

    avg_loss = losses.rolling(
        period,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # No losses means RSI is 100 when gains exist.
    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100
    )

    # No gains means RSI is 0 when losses exist.
    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss > 0),
        0
    )

    # No gains and no losses means neutral RSI.
    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss == 0),
        50
    )

    return rsi


def calculate_indicators(data):
    data = data.copy()
    close = data["Close"]

    data["SMA_20"] = close.rolling(
        20,
        min_periods=20
    ).mean()

    data["SMA_50"] = close.rolling(
        50,
        min_periods=50
    ).mean()

    rolling_std = close.rolling(
        20,
        min_periods=20
    ).std()

    data["BB_Middle"] = data["SMA_20"]
    data["BB_Upper"] = data["BB_Middle"] + 2 * rolling_std
    data["BB_Lower"] = data["BB_Middle"] - 2 * rolling_std
    data["RSI_14"] = calculate_rsi(close)

    return data


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
  <div class="eyebrow">Your market, made clearer</div>
  <h1>FinSight <span style="color:#9B8CFF">AI</span></h1>
  <p>
    Explore global stock markets, analyze company performance,
    understand historical risk, compare assets, and examine
    experimental price estimates in one workspace.
  </p>
  <div class="hero-pills">
    <span class="pill">✦ Market insights</span>
    <span class="pill">↗ Performance & risk</span>
    <span class="pill">⌁ Technical indicators</span>
    <span class="pill">◎ Stock comparison</span>
    <span class="pill">🌐 US & Indian markets</span>
  </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.markdown("## ✦ FinSight AI")
st.sidebar.caption("A clearer view of the market")
st.sidebar.divider()

st.sidebar.markdown("### Dashboard settings")

market_filter = st.sidebar.selectbox(
    "Market",
    [
        "All Markets",
        "US Stocks",
        "Indian Stocks",
        "Market Indices"
    ]
)

if market_filter == "US Stocks":
    available_companies = [
        name
        for name, symbol in stock_options.items()
        if not is_indian(symbol) and not is_index(symbol)
    ]

elif market_filter == "Indian Stocks":
    available_companies = [
        name
        for name, symbol in stock_options.items()
        if is_indian(symbol)
    ]

elif market_filter == "Market Indices":
    available_companies = [
        name
        for name, symbol in stock_options.items()
        if is_index(symbol)
    ]

else:
    available_companies = list(stock_options.keys())

if not available_companies:
    st.error("No assets are available for this market filter.")
    st.stop()

company = st.sidebar.selectbox(
    "Primary company or index",
    available_companies
)

period = st.sidebar.selectbox(
    "Historical period",
    list(period_options.keys()),
    index=2
)

comparison_options = [
    name for name in stock_options
    if name != company
]

comparison_company = st.sidebar.selectbox(
    "Compare against",
    comparison_options
)

st.sidebar.caption(
    "Select an asset, choose a time period, and open either dashboard view."
)

st.sidebar.divider()
st.sidebar.markdown("### ✦ About FinSight AI")

st.sidebar.markdown("""
<div style="
    padding: 10px 12px;
    background: linear-gradient(135deg,#151E30,#101725);
    border: 1px solid #2A3850;
    border-radius: 12px;
">
    <div style="
        color:#F3F6FC;
        font-size:.9rem;
        font-weight:700;
        margin-bottom:5px;
    ">
        Your market, made clearer.
    </div>
    <div style="
        color:#96A3B8;
        font-size:.77rem;
        line-height:1.5;
    ">
        Explore stock performance, analyze historical
        trends, compare companies, and evaluate
        experimental price estimates.
    </div>
    <div style="
        margin-top:8px;
        padding-top:7px;
        border-top:1px solid #263248;
        color:#9B8CFF;
        font-size:.7rem;
    ">
        Market insights · Not investment advice
    </div>
</div>
""", unsafe_allow_html=True)

ticker = stock_options[company]
comparison_ticker = stock_options[comparison_company]
selected_period = period_options[period]


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

if "active_view" not in st.session_state:
    st.session_state.active_view = "home"

nav1, nav2 = st.columns(2, gap="medium")

with nav1:
    if st.button("📈  Stock Analysis", key="analysis_button"):
        st.session_state.active_view = "analysis"

with nav2:
    if st.button("↔  Compare Stocks", key="comparison_button"):
        st.session_state.active_view = "comparison"


# ==================================================
# STOCK ANALYSIS
# ==================================================

def render_stock_analysis(
    company,
    ticker,
    period_label,
    period_code
):
    try:
        with st.spinner(f"Getting market data for {company}..."):
            raw_data = fetch_history(ticker, period_code)

        if raw_data.empty:
            st.error(
                f"No market data was returned for {company} ({ticker}). "
                "The ticker may be unavailable or Yahoo Finance may be "
                "temporarily unreachable. Try another asset or period."
            )
            return

        if len(raw_data) < 2:
            st.error("There is not enough historical data for this asset.")
            return

        data = calculate_indicators(raw_data)
        close = data["Close"]

        latest = safe_float(close.iloc[-1])
        previous = safe_float(close.iloc[-2])

        if (
            not np.isfinite(latest)
            or not np.isfinite(previous)
            or previous == 0
        ):
            st.error("The latest market values are unavailable or invalid.")
            return

        change = latest - previous
        change_pct = change / previous * 100

        currency = get_currency(ticker)
        price_unit = price_axis(ticker)

        # Market snapshot

        company_header(company, ticker)

        section_heading(
            "Market snapshot",
            "The latest available close and day-over-day movement.",
            "01 / At a glance"
        )

        c1, c2, c3 = st.columns(3, gap="medium")

        c1.metric(
            "Latest close",
            price_text(latest, ticker)
        )

        c2.metric(
            "Daily change",
            price_text(change, ticker),
            f"{change_pct:+.2f}%"
        )

        c3.metric(
            "Historical period",
            period_label
        )

        if is_index(ticker):
            st.caption(
                "Index values are displayed in points. "
                "The latest available close may not be a live quote."
            )
        else:
            st.caption(
                f"Prices are displayed in {currency}. "
                "The latest available close may not be a live quote."
            )

        # Price history

        st.divider()

        section_heading(
            "Price journey",
            "Explore historical closing-price movement.",
            "02 / Price history"
        )

        fig = go.Figure()

        add_line(
            fig,
            data.index,
            close,
            "Closing price",
            GREEN,
            2.8
        )

        fig.update_traces(
            fill="tozeroy",
            fillcolor="rgba(73,215,160,.07)",
            selector=dict(name="Closing price")
        )

        style_chart(fig, 390, price_unit)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        with st.expander("View recent market data"):
            display = data.tail(10).copy()
            display.index = display.index.strftime("%Y-%m-%d")

            st.dataframe(
                display,
                use_container_width=True
            )

        # Performance and risk

        st.divider()

        returns = (
            close.pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        total_return = (
            close.iloc[-1] / close.iloc[0] - 1
        ) * 100

        daily_volatility = (
            returns.std() * 100
            if len(returns) > 1
            else np.nan
        )

        drawdown = close / close.cummax() - 1
        max_drawdown = drawdown.min() * 100

        avg_volume = np.nan

        if "Volume" in data.columns and not is_index(ticker):
            valid_volume = pd.to_numeric(
                data["Volume"],
                errors="coerce"
            ).dropna()

            if not valid_volume.empty:
                avg_volume = valid_volume.mean()

        section_heading(
            "Performance & risk",
            "A historical view of return, variability, and downside movement.",
            "03 / Risk dashboard"
        )

        m1, m2, m3, m4 = st.columns(4, gap="medium")

        m1.metric(
            "Total return",
            f"{total_return:+.2f}%"
        )

        m2.metric(
            "Daily volatility",
            f"{daily_volatility:.2f}%"
            if np.isfinite(daily_volatility)
            else "N/A"
        )

        m3.metric(
            "Max drawdown",
            f"{max_drawdown:.2f}%"
        )

        m4.metric(
            "Avg. daily volume",
            f"{avg_volume:,.0f}"
            if np.isfinite(avg_volume)
            else "N/A"
        )

        chart1, chart2 = st.columns(2, gap="medium")

        with chart1:
            st.markdown("#### Daily returns")

            fig_r = go.Figure()

            return_colors = [
                GREEN if value >= 0 else RED
                for value in returns.to_numpy()
            ]

            fig_r.add_trace(
                go.Bar(
                    x=returns.index,
                    y=returns * 100,
                    name="Daily return",
                    marker_color=return_colors,
                    hovertemplate=(
                        "%{x}<br>%{y:+.2f}%<extra></extra>"
                    )
                )
            )

            style_chart(fig_r, 290, "Return (%)")
            fig_r.update_layout(
                showlegend=False,
                bargap=0.18
            )

            st.plotly_chart(
                fig_r,
                use_container_width=True,
                config={"displayModeBar": False}
            )

        with chart2:
            st.markdown("#### Historical drawdown")

            fig_d = go.Figure()

            add_line(
                fig_d,
                drawdown.index,
                drawdown * 100,
                "Drawdown",
                RED,
                2.2
            )

            fig_d.update_traces(
                fill="tozeroy",
                fillcolor="rgba(255,114,133,.09)"
            )

            style_chart(fig_d, 290, "Drawdown (%)")

            st.plotly_chart(
                fig_d,
                use_container_width=True,
                config={"displayModeBar": False}
            )

        with st.expander("What do these metrics mean?"):
            st.markdown("""
            - **Total return:** Percentage change from the first to the latest close.
            - **Daily volatility:** Standard deviation of daily percentage returns.
            - **Maximum drawdown:** Largest decline from a previous closing-price peak.
            - **Average daily volume:** Average shares traded per day, where available.
            """)

        # Moving averages

        st.divider()

        section_heading(
            "Technical indicators",
            "Historical trend and momentum indicators, not guaranteed predictions.",
            "04 / Technical view"
        )

        st.markdown("#### Moving averages")

        st.caption(
            "SMA 20 and SMA 50 smooth the closing price over trading sessions."
        )

        fig_sma = go.Figure()

        add_line(
            fig_sma,
            data.index,
            close,
            "Close",
            "#D8DEEA",
            1.8
        )

        add_line(
            fig_sma,
            data.index,
            data["SMA_20"],
            "SMA 20",
            VIOLET,
            2.2
        )

        add_line(
            fig_sma,
            data.index,
            data["SMA_50"],
            "SMA 50",
            CYAN,
            2.2
        )

        style_chart(fig_sma, 350, price_unit)

        st.plotly_chart(
            fig_sma,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        sma20 = safe_float(data["SMA_20"].iloc[-1])
        sma50 = safe_float(data["SMA_50"].iloc[-1])

        sma1, sma2 = st.columns(2)

        sma1.metric(
            "Latest SMA 20",
            price_text(sma20, ticker)
        )

        sma2.metric(
            "Latest SMA 50",
            price_text(sma50, ticker)
        )

        if np.isfinite(sma20) and np.isfinite(sma50):
            if sma20 > sma50:
                st.info(
                    "SMA 20 is above SMA 50; "
                    "the shorter-term average is higher."
                )
            elif sma20 < sma50:
                st.info(
                    "SMA 20 is below SMA 50; "
                    "the shorter-term average is lower."
                )
            else:
                st.info(
                    "SMA 20 and SMA 50 are approximately equal."
                )

        # RSI

        st.divider()
        st.markdown("#### Relative Strength Index (RSI)")

        st.caption(
            "RSI tracks recent momentum on a 0–100 scale."
        )

        fig_rsi = go.Figure()

        add_line(
            fig_rsi,
            data.index,
            data["RSI_14"],
            "RSI (14)",
            VIOLET,
            2.4
        )

        fig_rsi.add_hline(
            y=70,
            line_dash="dash",
            line_color=RED,
            annotation_text="70 · upper level"
        )

        fig_rsi.add_hline(
            y=30,
            line_dash="dash",
            line_color=CYAN,
            annotation_text="30 · lower level"
        )

        fig_rsi.update_yaxes(range=[0, 100])
        style_chart(fig_rsi, 285, "RSI")

        st.plotly_chart(
            fig_rsi,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        latest_rsi = safe_float(data["RSI_14"].iloc[-1])

        if not np.isfinite(latest_rsi):
            st.metric(
                "Latest RSI (14)",
                "Not enough data"
            )
        else:
            r1, r2 = st.columns(2)

            r1.metric(
                "Latest RSI (14)",
                f"{latest_rsi:.2f}"
            )

            if latest_rsi > 70:
                r2.metric("Momentum zone", "Above 70")
                st.warning(
                    "RSI is above 70, indicating strong recent upward momentum."
                )

            elif latest_rsi < 30:
                r2.metric("Momentum zone", "Below 30")
                st.info(
                    "RSI is below 30, indicating strong recent downward momentum."
                )

            else:
                r2.metric("Momentum zone", "Between 30–70")
                st.info(
                    "RSI is between the commonly used reference levels."
                )

        # Bollinger Bands

        st.divider()
        st.markdown("#### Bollinger Bands")

        st.caption(
            "The bands use a 20-session average and two standard deviations."
        )

        fig_bb = go.Figure()

        add_line(
            fig_bb, data.index, data["BB_Upper"],
            "Upper band", CYAN, 1.8
        )

        add_line(
            fig_bb, data.index, data["BB_Middle"],
            "Middle band", VIOLET, 2
        )

        add_line(
            fig_bb, data.index, data["BB_Lower"],
            "Lower band", CYAN, 1.8
        )

        add_line(
            fig_bb, data.index, close,
            "Closing price", "#F3F6FC", 2.2
        )

        style_chart(fig_bb, 350, price_unit)

        st.plotly_chart(
            fig_bb,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        upper = safe_float(data["BB_Upper"].iloc[-1])
        middle = safe_float(data["BB_Middle"].iloc[-1])
        lower = safe_float(data["BB_Lower"].iloc[-1])

        b1, b2, b3 = st.columns(3)

        b1.metric(
            "Upper band",
            price_text(upper, ticker)
        )

        b2.metric(
            "Middle band",
            price_text(middle, ticker)
        )

        b3.metric(
            "Lower band",
            price_text(lower, ticker)
        )

        with st.expander("How to read technical indicators"):
            st.markdown("""
            - **SMA 20 / SMA 50:** Average closing price over 20 / 50 sessions.
            - **RSI 14:** A momentum indicator based on recent gains and losses.
            - **Bollinger Bands:** A moving average with upper and lower standard-deviation bands.
            - These indicators describe historical patterns and do not guarantee future outcomes.
            """)

        # Experimental price estimate

        st.divider()

        section_heading(
            "Price outlook",
            "A simple experimental next-session estimate compared with a last-price baseline.",
            "05 / Experimental estimate"
        )

        prices = close.reset_index(drop=True).astype(float)
        window = 5

        X = []
        y = []

        for i in range(window, len(prices)):
            X.append(
                prices.iloc[i-window:i].tolist()
            )
            y.append(
                float(prices.iloc[i])
            )

        if len(X) < 10:
            st.warning(
                "There is not enough historical data to train and evaluate this model."
            )
            footer()
            return

        # Time-ordered split: earlier observations train the model;
        # later observations test it.

        split = int(len(X) * 0.8)
        split = max(1, min(split, len(X) - 1))

        X_train = X[:split]
        y_train = y[:split]

        X_test = X[split:]
        y_test = y[split:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        baseline = np.array(
            [row[-1] for row in X_test],
            dtype=float
        )

        y_test_array = np.array(
            y_test,
            dtype=float
        )

        model_mae = mean_absolute_error(
            y_test_array,
            predictions
        )

        baseline_mae = mean_absolute_error(
            y_test_array,
            baseline
        )

        model_rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test_array,
                    predictions
                )
            )
        )

        baseline_rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test_array,
                    baseline
                )
            )
        )

        direction_actual = np.sign(
            y_test_array - baseline
        )

        direction_pred = np.sign(
            predictions - baseline
        )

        direction_accuracy = float(
            np.mean(
                direction_actual == direction_pred
            ) * 100
        )

        final_model = LinearRegression().fit(X, y)

        estimate = float(
            final_model.predict([
                prices.iloc[-window:].tolist()
            ])[0]
        )

        estimate_change = (
            (estimate / latest - 1) * 100
            if latest
            else np.nan
        )

        e1, e2 = st.columns(2, gap="medium")

        e1.metric(
            "Estimated next close",
            price_text(estimate, ticker)
        )

        e2.metric(
            "Change vs. latest close",
            f"{estimate_change:+.2f}%"
            if np.isfinite(estimate_change)
            else "N/A"
        )

        st.caption(
            "Experimental estimate only — not a trading signal or reliable forecast."
        )

        st.markdown("#### Historical evaluation")

        q1, q2, q3 = st.columns(3, gap="medium")

        q1.metric(
            "Average absolute error",
            price_text(model_mae, ticker)
        )

        q2.metric(
            "Root mean squared error",
            price_text(model_rmse, ticker)
        )

        q3.metric(
            "Directional accuracy",
            f"{direction_accuracy:.1f}%"
        )

        evaluation = pd.DataFrame({
            "Approach": [
                "Linear Regression",
                "Last-price baseline"
            ],
            f"MAE ({unit_label(ticker)})": [
                model_mae,
                baseline_mae
            ],
            f"RMSE ({unit_label(ticker)})": [
                model_rmse,
                baseline_rmse
            ]
        })

        st.dataframe(
            evaluation,
            use_container_width=True,
            hide_index=True
        )

        if model_mae < baseline_mae:
            st.success(
                "Linear Regression had lower MAE on this test period."
            )
        elif model_mae > baseline_mae:
            st.warning(
                "The last-price baseline performed better on this test period."
            )
        else:
            st.info(
                "Both approaches had the same MAE on this test period."
            )

        results = pd.DataFrame({
            "Actual close": y_test_array,
            "Linear Regression": predictions,
            "Last-price baseline": baseline
        })

        fig_eval = go.Figure()
        x_eval = list(range(1, len(results) + 1))

        add_line(
            fig_eval,
            x_eval,
            results["Actual close"],
            "Actual close",
            "#F3F6FC",
            2.4
        )

        add_line(
            fig_eval,
            x_eval,
            results["Linear Regression"],
            "Estimated",
            VIOLET,
            2.2
        )

        add_line(
            fig_eval,
            x_eval,
            results["Last-price baseline"],
            "Baseline",
            CYAN,
            1.8,
            "dash"
        )

        fig_eval.update_xaxes(
            title="Test observation"
        )

        style_chart(
            fig_eval,
            330,
            price_unit
        )

        st.plotly_chart(
            fig_eval,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        with st.expander("How the estimate works"):
            st.markdown("""
            The model uses the previous five closing prices as inputs.
            Earlier observations are used for training and newer observations
            are held out for testing. Its results are compared with a baseline
            using the last known price. It does not consider news, company
            fundamentals, or wider market conditions.
            """)

        footer()

    except Exception as error:
        st.error(
            "Unable to load market analysis. "
            "Please check your connection and try again."
        )

        with st.expander("Technical details"):
            st.code(
                f"{type(error).__name__}: {error}"
            )


# ==================================================
# STOCK COMPARISON
# ==================================================

def render_stock_comparison(
    company,
    ticker,
    comparison_company,
    comparison_ticker,
    period_label,
    period_code
):
    try:
        with st.spinner("Bringing both assets into view..."):
            first = fetch_history(
                ticker,
                period_code
            )

            second = fetch_history(
                comparison_ticker,
                period_code
            )

        if first.empty:
            st.error(
                f"No market data was returned for {company} ({ticker}). "
                "Try another asset or period."
            )
            return

        if second.empty:
            st.error(
                f"No market data was returned for "
                f"{comparison_company} ({comparison_ticker}). "
                "Try another asset or period."
            )
            return

        if len(first) < 2 or len(second) < 2:
            st.error(
                "Not enough historical data to compare these assets."
            )
            return

        first_close = first["Close"].astype(float)
        second_close = second["Close"].astype(float)

        first_start = float(first_close.iloc[0])
        second_start = float(second_close.iloc[0])

        if first_start <= 0 or second_start <= 0:
            st.error(
                "The starting prices are invalid, so the assets cannot be normalized."
            )
            return

        # Normalize independently to account for different trading calendars.

        first_normalized = (
            first_close / first_start * 100
        )

        second_normalized = (
            second_close / second_start * 100
        )

        company_header(company, ticker)

        section_heading(
            "Two assets. One clear view.",
            "Compare relative performance and historical variability.",
            "Stock comparison"
        )

        st.caption(
            f"{period_label} · {company} ({ticker}) vs. "
            f"{comparison_company} ({comparison_ticker})"
        )

        fig = go.Figure()

        add_line(
            fig,
            first_normalized.index,
            first_normalized,
            company,
            VIOLET,
            2.8
        )

        add_line(
            fig,
            second_normalized.index,
            second_normalized,
            comparison_company,
            CYAN,
            2.8
        )

        fig.add_hline(
            y=100,
            line_dash="dot",
            line_color="#64748B",
            annotation_text="Starting point = 100"
        )

        style_chart(
            fig,
            410,
            "Indexed performance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.caption(
            "Each asset starts at 100 on its own first available date. "
            "This compares percentage movement, not absolute prices."
        )

        # Performance

        first_return = (
            first_close.iloc[-1] / first_close.iloc[0] - 1
        ) * 100

        second_return = (
            second_close.iloc[-1] / second_close.iloc[0] - 1
        ) * 100

        # Annualized volatility is an approximation using 252 sessions.

        first_daily = (
            first_close.pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        second_daily = (
            second_close.pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        first_vol = (
            first_daily.std() * np.sqrt(252) * 100
            if len(first_daily) > 1
            else np.nan
        )

        second_vol = (
            second_daily.std() * np.sqrt(252) * 100
            if len(second_daily) > 1
            else np.nan
        )

        st.divider()

        section_heading(
            "Performance & risk",
            "Side-by-side metrics calculated from each asset's historical data.",
            "At a glance"
        )

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown(
                f"#### {company} · {ticker}"
            )

            st.metric(
                "Total return",
                f"{first_return:+.2f}%"
            )

            st.metric(
                "Annualized volatility",
                f"{first_vol:.2f}%"
                if np.isfinite(first_vol)
                else "N/A"
            )

            st.metric(
                "Latest close",
                price_text(
                    float(first_close.iloc[-1]),
                    ticker
                )
            )

        with right:
            st.markdown(
                f"#### {comparison_company} · {comparison_ticker}"
            )

            st.metric(
                "Total return",
                f"{second_return:+.2f}%"
            )

            st.metric(
                "Annualized volatility",
                f"{second_vol:.2f}%"
                if np.isfinite(second_vol)
                else "N/A"
            )

            st.metric(
                "Latest close",
                price_text(
                    float(second_close.iloc[-1]),
                    comparison_ticker
                )
            )

        summary = pd.DataFrame({
            "Metric": [
                "Total return (%)",
                "Annualized volatility (%)",
                "Latest close",
                "Display unit"
            ],
            company: [
                round(float(first_return), 4),
                round(float(first_vol), 4)
                if np.isfinite(first_vol)
                else np.nan,
                round(float(first_close.iloc[-1]), 4),
                unit_label(ticker)
            ],
            comparison_company: [
                round(float(second_return), 4),
                round(float(second_vol), 4)
                if np.isfinite(second_vol)
                else np.nan,
                round(float(second_close.iloc[-1]), 4),
                unit_label(comparison_ticker)
            ]
        })

        st.markdown("#### Quick comparison")

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        # Automatic insights

        st.divider()

        section_heading(
            "What stands out",
            "Simple observations from the selected historical period.",
            "Comparison notes"
        )

        insight1, insight2 = st.columns(2, gap="medium")

        with insight1:
            if first_return > second_return:
                st.success(
                    f"{company} returned "
                    f"{first_return - second_return:.2f} "
                    "percentage points more."
                )

            elif second_return > first_return:
                st.success(
                    f"{comparison_company} returned "
                    f"{second_return - first_return:.2f} "
                    "percentage points more."
                )

            else:
                st.info(
                    "Both assets had the same measured total return."
                )

        with insight2:
            if (
                not np.isfinite(first_vol)
                or not np.isfinite(second_vol)
            ):
                st.info(
                    "There is not enough return data to compare volatility."
                )

            elif first_vol < second_vol:
                st.info(
                    f"{company} had lower historical volatility."
                )

            elif second_vol < first_vol:
                st.info(
                    f"{comparison_company} had lower historical volatility."
                )

            else:
                st.info(
                    "Both assets had the same measured volatility."
                )

        st.caption(
            "Volatility measures variability, not whether an investment "
            "is good or bad. Past performance does not guarantee future results."
        )

        footer()

    except Exception as error:
        st.error(
            "Unable to compare the selected assets. "
            "Please check your connection and try again."
        )

        with st.expander("Technical details"):
            st.code(
                f"{type(error).__name__}: {error}"
            )


# ==================================================
# DISPLAY SELECTED VIEW
# ==================================================

if st.session_state.active_view == "analysis":

    render_stock_analysis(
        company,
        ticker,
        period,
        selected_period
    )

elif st.session_state.active_view == "comparison":

    render_stock_comparison(
        company,
        ticker,
        comparison_company,
        comparison_ticker,
        period,
        selected_period
    )

else:

    st.markdown(
        '<div class="section-kicker">Pick your starting point</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "What would you like to explore?"
    )

    st.markdown(
        '<div class="section-note">'
        'Two focused tools, one clean workspace. '
        'Choose a view above to get started.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    a, b = st.columns(2, gap="large")

    with a:
        st.markdown("""
        <div class="home-card">
          <div class="home-icon">📈</div>
          <h3>Stock analysis</h3>
          <p>
            Explore price history, market performance,
            risk, technical indicators, and experimental
            price estimates for US and Indian assets.
          </p>
          <span class="pill">Price history</span>
          <span class="pill">Risk metrics</span>
          <br><br>
          <span class="pill">Technical indicators</span>
          <span class="pill">Price outlook</span>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="home-card">
          <div class="home-icon">↔️</div>
          <h3>Stock comparison</h3>
          <p>
            Compare two companies or indices through
            normalized performance charts, historical
            returns, and volatility metrics.
          </p>
          <span class="pill">US & Indian markets</span>
          <span class="pill">Normalized chart</span>
          <br><br>
          <span class="pill">Volatility</span>
          <span class="pill">Quick insights</span>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.caption(
        "Choose a market, asset, and historical period in the sidebar. "
        "Then select Stock Analysis or Compare Stocks above."
    )

    footer()
