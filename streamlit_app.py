import math
import re
import os
import json
import datetime
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import twstock
except Exception:
    twstock = None

# =========================================================
# Basic Settings
# =========================================================
TW_TZ = timezone(timedelta(hours=8))
SCAN_CACHE_MINUTES = 30
FIRST_STAGE_KEEP = 50
BATCH_SIZE = 80
MIN_AVG_TURNOVER = 8_000_000
STRONG_BULL_SCORE = 2.0
PULLBACK_MIN_SCORE = 0.0


# =========================================================
# Stock Universe / Data Retrieval
# =========================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_full_market_pool():
    rows = []
    if twstock is not None:
        try:
            for code, info in twstock.codes.items():
                code = str(code).strip()
                if not re.fullmatch(r"\d{4}", code):
                    continue
                market = str(getattr(info, "market", ""))
                kind = str(getattr(info, "type", getattr(info, "security_type", "")))
                name = str(getattr(info, "name", code))
                if market not in ("上市", "上櫃"):
                    continue
                if kind and "股票" not in kind:
                    continue
                suffix = ".TW" if market == "上市" else ".TWO"
                rows.append((code, name, suffix))
        except Exception:
            rows = []

    if not rows:
        rows = []

    dedup = {code: (code, name, suffix) for code, name, suffix in rows}
    return sorted(dedup.values(), key=lambda x: x[0])


def market_lookup():
    return {code: (name, suffix) for code, name, suffix in get_full_market_pool()}


def resolve_stock_name(stock_id: str) -> str:
    lookup = market_lookup()
    if stock_id in lookup:
        return lookup[stock_id][0]
    return stock_id


def suffix_candidates(stock_id: str):
    lookup = market_lookup()
    if stock_id in lookup and lookup[stock_id][1] in (".TW", ".TWO"):
        return [lookup[stock_id][1]]
    return [".TW", ".TWO"]


def clean_price_frame(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = [c for c in ["Close", "High", "Low"] if c in df.columns]
    if len(needed) < 3:
        return pd.DataFrame()
    df = df.dropna(subset=needed)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    else:
        df["Volume"] = df["Volume"].fillna(0)
    return df


@st.cache_data(ttl=180, show_spinner=False)
def fetch_stock_df(stock_id: str, interval="1d", period="6mo"):
    for suffix in suffix_candidates(stock_id):
        try:
            df = yf.download(
                f"{stock_id}{suffix}",
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            df = clean_price_frame(df)
            if len(df) >= 25:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def _extract_symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    try:
        if not isinstance(raw.columns, pd.MultiIndex):
            df = raw.copy()
        elif symbol in raw.columns.get_level_values(0):
            df = raw[symbol].copy()
        elif symbol in raw.columns.get_level_values(1):
            df = raw.xs(symbol, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
        return clean_price_frame(df)
    except Exception:
        return pd.DataFrame()


# =========================================================
# Technical Indicators and Scoring (same core logic as the Telegram bot)
# =========================================================
def calculate_indicators(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["Close", "High", "Low"]).copy()
    if len(df) < 25:
        raise ValueError("Insufficient valid price data")

    close = df["Close"].squeeze().astype(float)
    high = df["High"].squeeze().astype(float)
    low = df["Low"].squeeze().astype(float)
    volume = df["Volume"].squeeze().fillna(0).astype(float)

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(min(60, len(close))).mean()

    std20 = close.rolling(20).std()
    bb_up = ma20 + 2 * std20
    bb_low = ma20 - 2 * std20

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9 + 1e-10) * 100.0
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    macd = dif.ewm(span=9, adjust=False).mean()
    osc = dif - macd

    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    vol_ma20 = volume.rolling(20).mean()

    return {
        "c_now": float(close.iloc[-1]),
        "c_prev": float(close.iloc[-2]),
        "ma5": float(ma5.iloc[-1]),
        "ma20": float(ma20.iloc[-1]),
        "ma60": float(ma60.iloc[-1]),
        "bb_up": float(bb_up.iloc[-1]),
        "bb_low": float(bb_low.iloc[-1]),
        "rsi": float(rsi.iloc[-1]),
        "k_now": float(k.iloc[-1]),
        "d_now": float(d.iloc[-1]),
        "k_prev": float(k.iloc[-2]),
        "d_prev": float(d.iloc[-2]),
        "dif_now": float(dif.iloc[-1]),
        "macd_now": float(macd.iloc[-1]),
        "osc_now": float(osc.iloc[-1]),
        "atr": float(atr.iloc[-1]) if math.isfinite(float(atr.iloc[-1])) else float(close.iloc[-1]) * 0.02,
        "vol_now": float(volume.iloc[-1]),
        "vol_ma20": float(vol_ma20.iloc[-1]) if float(vol_ma20.iloc[-1]) > 0 else 1.0,
    }


def evaluate_score(ind: dict) -> dict:
    score = 0.0
    bull_signals, bear_signals = [], []

    c = ind["c_now"]
    ma5, ma20, ma60 = ind["ma5"], ind["ma20"], ind["ma60"]

    if c > ma5 > ma20 > ma60:
        score += 1.5
        bull_signals.append("Price > MA5 > MA20 > MA60 (+1.5)")
    elif c > ma20:
        score += 1.0
        bull_signals.append("Price above MA20 (+1.0)")
    elif c < ma5 < ma20 < ma60:
        score -= 1.5
        bear_signals.append("Price < MA5 < MA20 < MA60 (-1.5)")
    else:
        score -= 1.0
        bear_signals.append("Price below MA20 (-1.0)")

    rsi = ind["rsi"]
    if 55 <= rsi <= 75:
        score += 1.0
        bull_signals.append("RSI between 55 and 75 (+1.0)")
    elif rsi < 30:
        score += 1.0
        bull_signals.append("Oversold RSI, possible rebound (+1.0)")
    elif rsi > 80:
        score -= 1.0
        bear_signals.append("Overbought RSI (-1.0)")
    elif rsi < 45:
        score -= 0.5
        bear_signals.append("Weak RSI momentum (-0.5)")

    k_n, d_n = ind["k_now"], ind["d_now"]
    k_p, d_p = ind["k_prev"], ind["d_prev"]
    if k_p < d_p and k_n > d_n:
        score += 1.0
        bull_signals.append("KD golden cross (K crossed above D) (+1.0)")
    elif k_n > d_n:
        score += 0.5
        bull_signals.append("K is above D (+0.5)")
    elif k_p > d_p and k_n < d_n:
        score -= 1.0
        bear_signals.append("KD death cross (K crossed below D) (-1.0)")
    else:
        score -= 0.5
        bear_signals.append("K is at or below D (-0.5)")

    if ind["dif_now"] > ind["macd_now"]:
        score += 1.0
        bull_signals.append("MACD bullish momentum (+1.0)")
    else:
        score -= 1.0
        bear_signals.append("MACD bearish momentum (-1.0)")

    vol_ratio = ind["vol_now"] / ind["vol_ma20"]
    if vol_ratio >= 1.5 and c > ind["c_prev"]:
        score += 1.0
        bull_signals.append(f"Price rose on {vol_ratio:.1f}× the 20-period average volume (+1.0)")
    elif c < ind["bb_low"]:
        score += 0.5
        bull_signals.append("Price below the lower Bollinger Band (+0.5)")
    elif c > ind["bb_up"]:
        score -= 0.5
        bear_signals.append("Price above the upper Bollinger Band (-0.5)")

    final_score = max(-5.0, min(5.0, round(score, 1)))
    return {
        "score": final_score,
        "bull_pct": (final_score + 5.0) / 10.0 * 100.0,
        "bull_signals": bull_signals,
        "bear_signals": bear_signals,
        "vol_ratio": vol_ratio,
    }


def build_trading_strategy(ind: dict, ev: dict) -> dict:
    c = ind["c_now"]
    ma5, ma20, ma60 = ind["ma5"], ind["ma20"], ind["ma60"]
    rsi, score, vol_ratio = ind["rsi"], ev["score"], ev["vol_ratio"]
    atr = max(ind["atr"], c * 0.01)
    stop = max(ma20, c - 2.0 * atr) if c > ma20 else c - 2.0 * atr
    target = max(ind["bb_up"], c * 1.06)
    buy_low, buy_high = min(ma5, ma20), max(ma5, ma20)

    if score >= 3.0 and c > ma20:
        stance = "🟢 Bullish / hold"
        new_entry = f"Avoid chasing the price. If it pulls back to the {buy_low:.2f} to {buy_high:.2f} range and steadies, consider buying in stages."
        holding = f"Hold for now. Consider taking some profits near {target:.2f}. Trim or close the position if the price falls below {stop:.2f}."
    elif score >= 1.5 and c > ma20:
        stance = "🟡 Buy on a pullback"
        new_entry = f"Wait for a pullback toward the {buy_low:.2f} to {buy_high:.2f} range. If the price steadies, consider buying in stages."
        holding = f"You can keep holding at your current position size. Trim it if the price falls below MA20 ({ma20:.2f}) and use {stop:.2f} as the stop level."
    elif score >= 0.5:
        stance = "🟡 Small position / wait for confirmation"
        new_entry = f"Start small. Add only if the price holds above MA20 ({ma20:.2f}) and volume picks up."
        holding = f"Hold for now and trim the position if the price falls below {ma20:.2f}."
    elif score <= -2.5:
        stance = "🔴 Bearish / avoid new entries"
        new_entry = f"Wait until the price moves back above MA20 ({ma20:.2f}) and both KD and MACD improve."
        holding = f"Use rebounds to trim the position. Exit if the price falls below {stop:.2f}."
    elif score <= -0.5:
        stance = "🟠 Weak / wait"
        new_entry = f"Reassess once the price steadies and moves back above MA20 ({ma20:.2f})."
        holding = f"Keep the position small and trim it if the price falls below {stop:.2f}. Reassess once the price moves back above MA20."
    else:
        stance = "⚪ Range-bound"
        new_entry = f"Consider an entry only near the {buy_low:.2f} to {buy_high:.2f} range. Reassess if the price breaks out of its range."
        holding = f"Keep your current position size. Exit if the price falls below {stop:.2f} and consider taking some profits near {target:.2f}."

    warns = []
    if rsi >= 75:
        warns.append("RSI is high. Avoid chasing the price.")
    if vol_ratio >= 2.0:
        warns.append("Volume is unusually high. Watch for larger price swings next session.")
    if c < ma60:
        warns.append("The price is below MA60, a sign of a weak medium-term trend.")

    return {
        "stance": stance,
        "new_entry": new_entry,
        "holding": holding,
        "warning": " ".join(warns) if warns else "No additional technical warnings.",
        "stop": stop,
        "target": target,
    }


# =========================================================
# Watchlist (stored separately in each visitor's Streamlit session)
# =========================================================
def load_favorites() -> list[str]:
    """Read the current visitor's watchlist for the selected market."""
    if "favorites_by_market" not in st.session_state:
        st.session_state["favorites_by_market"] = {"TW Stock": [], "US Stock": []}
    if market not in st.session_state["favorites_by_market"]:
        st.session_state["favorites_by_market"][market] = []
    return list(st.session_state["favorites_by_market"][market])


def save_favorites(items: list[str]) -> None:
    """Store watchlist data only in the current visitor session; do not write to shared JSON."""
    cleaned = []
    for x in items:
        x = str(x).strip().upper()
        valid = (
            re.fullmatch(r"\d{4,6}", x) if market == "TW Stock"
            else re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", x)
        )
        if valid and x not in cleaned:
            cleaned.append(x)
    st.session_state["favorites_by_market"][market] = cleaned


def add_favorite(stock_id: str) -> bool:
    stock_id = str(stock_id).strip().upper()
    valid = (
        re.fullmatch(r"\d{4,6}", stock_id) if market == "TW Stock"
        else re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", stock_id)
    )
    if not valid:
        return False
    favs = load_favorites()
    if stock_id not in favs:
        favs.append(stock_id)
        save_favorites(favs)
    return True


def remove_favorite(stock_id: str) -> None:
    favs = [x for x in load_favorites() if x != str(stock_id)]
    save_favorites(favs)





# =========================================================
# Market Switch: Taiwan / U.S.
# =========================================================
def normalize_symbol_by_market(stock_id: str, market: str) -> str:
    stock_id = stock_id.strip().upper()
    if market == "US Stock":
        return stock_id
    return stock_id


def resolve_name_by_market(stock_id: str, market: str) -> str:
    sid = stock_id.strip().upper()
    if market == "US Stock":
        if sid in US_LOOKUP:
            return US_LOOKUP[sid]
        try:
            info = yf.Ticker(sid).fast_info
            # fast_info often lacks name; fall back to info only if needed
            name = yf.Ticker(sid).info.get("shortName") or yf.Ticker(sid).info.get("longName")
            if name:
                return str(name)
        except Exception:
            pass
        return sid
    try:
        return resolve_stock_name(sid)
    except Exception:
        return sid


def fetch_market_df(stock_id: str, market: str, interval: str = "1d", period: str = "6mo") -> pd.DataFrame:
    """Shared market-data loader. Taiwan stocks use .TW/.TWO; U.S. stocks use Yahoo tickers directly."""
    sid = stock_id.strip().upper()
    if market == "US Stock":
        try:
            df = yf.download(
                sid,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            required = [c for c in ["Close", "High", "Low"] if c in df.columns]
            if len(required) < 3:
                return pd.DataFrame()
            df = df.dropna(subset=required).copy()
            if "Volume" not in df.columns:
                df["Volume"] = 0.0
            else:
                df["Volume"] = df["Volume"].fillna(0)
            return df
        except Exception:
            return pd.DataFrame()

    # Taiwan stocks: use the existing retrieval function
    try:
        # Some versions of fetch_stock_df are async; the Streamlit version is normally synchronous.
        result = fetch_stock_df(sid, interval=interval, period=period)
        return result
    except TypeError:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def get_us_index_overview() -> list[dict]:
    indexes = [
        ("SPY", "S&P 500"),
        ("QQQ", "Nasdaq 100"),
        ("DIA", "Dow Jones"),
        ("IWM", "Russell 2000"),
    ]
    result = []
    for ticker, label in indexes:
        df = fetch_market_df(ticker, "US Stock", interval="1d", period="1y")
        if df.empty or len(df) < 25:
            continue
        try:
            ind = calculate_indicators(df)
            ev = evaluate_score(ind)
            pct = (ind["c_now"] - ind["c_prev"]) / max(ind["c_prev"], 1e-9) * 100
            trend = (
                "🟢 Bullish" if ind["c_now"] > ind["ma20"] > ind["ma60"]
                else "🔴 Bearish" if ind["c_now"] < ind["ma20"] < ind["ma60"]
                else "🟡 Mixed"
            )
            result.append({
                "ticker": ticker,
                "name": label,
                "price": round(ind["c_now"], 2),
                "pct_ch": round(pct, 2),
                "score": ev["score"],
                "rsi": round(ind["rsi"], 1),
                "trend": trend,
            })
        except Exception:
            continue
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def run_us_scan_cached() -> dict:
    """U.S. radar: scan a representative high-liquidity stock universe. Individual stock analysis is not limited to this list."""
    rows = []
    above_ma20 = 0
    total = 0

    for ticker, name in US_QUICK_POOL:
        df = fetch_market_df(ticker, "US Stock", interval="1d", period="6mo")
        if df.empty or len(df) < 25:
            continue
        try:
            ind = calculate_indicators(df)
            ev = evaluate_score(ind)
            total += 1
            if ind["c_now"] > ind["ma20"]:
                above_ma20 += 1

            pct_ch = (ind["c_now"] - ind["c_prev"]) / max(ind["c_prev"], 1e-9) * 100
            item = {
                "ticker": ticker,
                "name": name,
                "price": ind["c_now"],
                "pct_ch": pct_ch,
                "score": ev["score"],
                "rsi": ind["rsi"],
                "vol_ratio": ev["vol_ratio"],
                "ma20": ind["ma20"],
                "ma60": ind["ma60"],
            }
            rows.append(item)
        except Exception:
            continue

    strong = [r for r in rows if r["score"] >= 1.5 and r["price"] > r["ma20"]]
    pullback = [r for r in rows if 0 <= r["score"] < 2.0 and r["price"] > r["ma60"] and r["pct_ch"] < 0]
    strong.sort(key=lambda r: r["score"], reverse=True)
    pullback.sort(key=lambda r: r["score"], reverse=True)

    return {
        "rows": rows,
        "strong": strong,
        "pullback": pullback,
        "above_ma20": above_ma20,
        "total": total,
        "time": datetime.datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
    }


# =========================================================
# AI Access Password Protection
# =========================================================
def ai_access_unlocked() -> bool:
    return bool(st.session_state.get("ai_access_unlocked", False))


def render_ai_login() -> bool:
    expected_password = get_secret("AI_ACCESS_PASSWORD")

    if not expected_password:
        st.warning("Set AI_ACCESS_PASSWORD to enable AI analysis.")
        return False

    if ai_access_unlocked():
        c1, c2 = st.columns([4, 1])
        with c1:
            st.success("AI analysis is unlocked.")
        with c2:
            if st.button("Lock AI access", key="ai_logout_btn"):
                st.session_state["ai_access_unlocked"] = False
                st.rerun()
        return True

    password = st.text_input(
        "AI access password",
        type="password",
        placeholder="Enter your AI access password",
        key="ai_access_password_input",
    )

    if st.button("Unlock AI access", key="ai_unlock_btn"):
        if password == expected_password:
            st.session_state["ai_access_unlocked"] = True
            st.success("AI analysis is unlocked.")
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False


# =========================================================
# AI Analysis (OpenAI Responses API)
# =========================================================
def get_secret(name: str, default=None):
    """Read Streamlit Secrets first; fall back to environment variables when running locally."""
    try:
        value = st.secrets.get(name, default)
        if value is not None:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fundamentals(stock_id: str, market: str = "TW Stock") -> dict:
    """Retrieve available fundamentals; return an empty result if Yahoo has no data so technical analysis can still run."""
    out = {}
    if market == "US Stock":
        symbol = stock_id.strip().upper()
    else:
        suffixes = suffix_candidates(stock_id)
        if not suffixes:
            return out
        symbol = f"{stock_id}{suffixes[0]}"
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        return out

    fields = {
        "marketCap": "Market Cap",
        "trailingPE": "P/E (TTM)",
        "forwardPE": "Forward P/E",
        "priceToBook": "Price-to-Book",
        "revenueGrowth": "Revenue Growth",
        "earningsGrowth": "Earnings Growth",
        "returnOnEquity": "ROE",
        "debtToEquity": "Debt-to-Equity",
        "profitMargins": "Profit Margin",
        "dividendYield": "Dividend Yield",
    }
    for key, label in fields.items():
        value = info.get(key)
        if value is not None:
            out[label] = value
    return out


def build_ai_payload(stock_id: str, stock_name: str, interval_label: str, ind: dict, ev: dict, strategy: dict, fundamentals: dict) -> dict:
    return {
        "Stock": f"{stock_name} ({stock_id})",
        "Chart Interval": interval_label,
        "Data Time": datetime.datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "Technical Data": {
            "Current Price": round(ind["c_now"], 2),
            "Previous Close": round(ind["c_prev"], 2),
            "MA5": round(ind["ma5"], 2),
            "MA20": round(ind["ma20"], 2),
            "MA60": round(ind["ma60"], 2),
            "Upper Bollinger Band": round(ind["bb_up"], 2),
            "Lower Bollinger Band": round(ind["bb_low"], 2),
            "RSI": round(ind["rsi"], 1),
            "K": round(ind["k_now"], 1),
            "D": round(ind["d_now"], 1),
            "DIF": round(ind["dif_now"], 3),
            "MACD": round(ind["macd_now"], 3),
            "ATR": round(ind["atr"], 2),
            "Volume Ratio": round(ev["vol_ratio"], 2),
            "Quantitative Score": ev["score"],
            "Bullish Strength %": round(ev["bull_pct"], 0),
            "Bullish Signals": ev["bull_signals"],
            "Bearish Signals": ev["bear_signals"],
        },
        "Quantitative Strategy": {
            "Stance": strategy["stance"],
            "New Entry": strategy["new_entry"],
            "Existing Position": strategy["holding"],
            "Risk Note": strategy["warning"],
            "Stop Reference": round(strategy["stop"], 2),
            "Target Reference": round(strategy["target"], 2),
        },
        "Available Fundamentals": fundamentals,
    }


def generate_ai_advice(stock_id: str, stock_name: str, interval_label: str, ind: dict, ev: dict, strategy: dict, market: str = "TW Stock") -> str:
    api_key = get_secret("OPENAI_API_KEY")
    model = get_secret("OPENAI_MODEL", "gpt-6-luna")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed. Please update requirements.txt.")

    fundamentals = fetch_fundamentals(stock_id, market)
    payload = build_ai_payload(stock_id, stock_name, interval_label, ind, ev, strategy, fundamentals)

    instructions = """
You are a stock research assistant. Analyze only the quantitative technical data and the available fundamental summary provided by the user.

Do not pretend to know real-time news, institutional trading flows, detailed financial statements, or price data that were not provided.
Write clearly and specifically in English, and do not guarantee investment returns.

Use the following structure:

## ⚡ Short-Term View
- Trend assessment:
- New-entry approach:
- Existing-position approach:
- Key support / weakening conditions:
- Short-term risks:

## 🌳 Long-Term View
- Medium- to long-term trend:
- Available fundamental information:
- Suitable approach:
- Conditions that would strengthen the long-term view:
- Long-term risks:

## 🎯 AI Summary
Summarize the overall view in 2–4 sentences as bullish, neutral, or bearish, and identify the most important condition to monitor.

Treat short term as several days to several weeks, and long term as several months or longer.
If fundamental data is insufficient, explicitly state "Insufficient fundamental data" rather than using technical indicators as a substitute for full fundamental analysis.
"""

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
    )
    return response.output_text.strip()


# =========================================================
# Charts
# =========================================================
def price_chart(df: pd.DataFrame, ind: dict):
    d = df.copy()
    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA60"] = d["Close"].rolling(min(60, len(d))).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index,
        open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="Candlestick"
    ))
    fig.add_trace(go.Scatter(x=d.index, y=d["MA5"], name="MA5", mode="lines"))
    fig.add_trace(go.Scatter(x=d.index, y=d["MA20"], name="MA20", mode="lines"))
    fig.add_trace(go.Scatter(x=d.index, y=d["MA60"], name="MA60", mode="lines"))
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
    )
    return fig


def score_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 5"},
        title={"text": "Technical score"},
        gauge={
            "axis": {"range": [-5, 5]},
            "steps": [
                {"range": [-5, -2], "color": "#f8d7da"},
                {"range": [-2, 2], "color": "#fff3cd"},
                {"range": [2, 5], "color": "#d1e7dd"},
            ],
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# =========================================================
# Full-Market Screening / Market Breadth
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def first_stage_market_scan(limit=FIRST_STAGE_KEEP):
    pool = get_full_market_pool()
    candidates = []
    above_ma20 = 0
    valid_total = 0

    for start in range(0, len(pool), BATCH_SIZE):
        chunk = pool[start:start+BATCH_SIZE]
        symbols = [code + suffix for code, _, suffix in chunk if suffix]
        if not symbols:
            continue
        try:
            raw = yf.download(
                tickers=" ".join(symbols),
                period="3mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        for code, name, suffix in chunk:
            if not suffix:
                continue
            df = _extract_symbol_frame(raw, code + suffix)
            if len(df) < 25:
                continue
            try:
                close = df["Close"].astype(float)
                volume = df["Volume"].fillna(0).astype(float)
                c = float(close.iloc[-1])
                ma5 = float(close.tail(5).mean())
                ma20 = float(close.tail(20).mean())
                ret5 = (c / float(close.iloc[-6]) - 1) * 100
                ret20 = (c / float(close.iloc[-21]) - 1) * 100
                vol20 = float(volume.tail(20).mean())
                vol_ratio = float(volume.iloc[-1] / vol20) if vol20 > 0 else 0.0
                avg_turnover = float((close.tail(20) * volume.tail(20)).mean())
                if not all(map(math.isfinite, [c, ma5, ma20, ret5, ret20, vol_ratio, avg_turnover])):
                    continue

                valid_total += 1
                if c > ma20:
                    above_ma20 += 1
                if avg_turnover < MIN_AVG_TURNOVER:
                    continue

                fast_score = 0.0
                fast_score += 2.0 if c > ma20 else -1.0
                fast_score += 1.0 if ma5 > ma20 else 0.0
                fast_score += max(-1.0, min(2.0, ret20 / 8.0))
                fast_score += max(-0.5, min(1.5, ret5 / 5.0))
                fast_score += max(0.0, min(1.5, vol_ratio - 1.0))

                candidates.append({
                    "ticker": code, "name": name, "suffix": suffix,
                    "fast_score": fast_score, "ret5": ret5, "ret20": ret20,
                    "vol_ratio": vol_ratio, "price": c,
                })
            except Exception:
                continue

    candidates.sort(key=lambda x: (x["fast_score"], x["ret20"], x["vol_ratio"]), reverse=True)
    return candidates[:limit], above_ma20, valid_total


def fetch_candidate_frames(candidates, period="6mo"):
    """Batch-download stage-two candidates to avoid sending 50 separate Yahoo requests."""
    result = {}
    for start in range(0, len(candidates), BATCH_SIZE):
        chunk = candidates[start:start+BATCH_SIZE]
        symbols = [c["ticker"] + c["suffix"] for c in chunk if c.get("suffix")]
        if not symbols:
            continue
        try:
            raw = yf.download(
                tickers=" ".join(symbols),
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception:
            continue
        for cand in chunk:
            symbol = cand["ticker"] + cand.get("suffix", "")
            df = _extract_symbol_frame(raw, symbol)
            if len(df) >= 25:
                result[cand["ticker"]] = df
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def run_full_market_scan():
    candidates, above_ma20, valid_total = first_stage_market_scan(FIRST_STAGE_KEEP)
    strong, pullback = [], []
    frames = fetch_candidate_frames(candidates, period="6mo")

    for cand in candidates:
        df = frames.get(cand["ticker"], pd.DataFrame())
        if df.empty:
            df = fetch_stock_df(cand["ticker"], "1d", "6mo")
        if df.empty:
            continue
        try:
            ind = calculate_indicators(df)
            ev = evaluate_score(ind)
            pct_ch = (ind["c_now"] - ind["c_prev"]) / max(ind["c_prev"], 1e-10) * 100
            row = {
                "Ticker": cand["ticker"],
                "Name": cand["name"],
                "Current Price": round(ind["c_now"], 2),
                "Score": ev["score"],
                "RSI": round(ind["rsi"], 1),
                "20D Return %": round(cand["ret20"], 1),
                "Volume Ratio": round(cand["vol_ratio"], 2),
                "Stage-1 Score": round(cand["fast_score"], 2),
            }
            if ev["score"] >= STRONG_BULL_SCORE and ind["c_now"] > ind["ma20"]:
                strong.append(row)
            elif PULLBACK_MIN_SCORE <= ev["score"] < STRONG_BULL_SCORE and ind["c_now"] > ind["ma60"] and pct_ch < 0:
                pullback.append(row)
        except Exception:
            continue

    strong.sort(key=lambda x: x["Score"], reverse=True)
    pullback.sort(key=lambda x: x["Score"], reverse=True)

    return {
        "strong": pd.DataFrame(strong),
        "pullback": pd.DataFrame(pullback),
        "above_ma20": above_ma20,
        "valid_total": valid_total,
        "candidates": len(candidates),
        "updated": datetime.datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def calculate_market_breadth():
    _, above_ma20, valid_total = first_stage_market_scan(FIRST_STAGE_KEEP)
    pct = above_ma20 / max(valid_total, 1) * 100
    return {
        "above_ma20": above_ma20,
        "total": valid_total,
        "pct": pct,
        "updated": datetime.datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
    }




# =========================================================
# Streamlit UI
# =========================================================

# U.S. popular-stock radar universe (not the full U.S. market)
US_QUICK_POOL = [
    ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"),
    ("AMZN", "Amazon"), ("META", "Meta"), ("GOOGL", "Alphabet"),
    ("TSLA", "Tesla"), ("AVGO", "Broadcom"), ("AMD", "AMD"),
    ("NFLX", "Netflix"), ("CRM", "Salesforce"), ("ORCL", "Oracle"),
    ("ADBE", "Adobe"), ("QCOM", "Qualcomm"), ("INTC", "Intel"),
    ("MU", "Micron"), ("AMAT", "Applied Materials"), ("LRCX", "Lam Research"),
    ("KLAC", "KLA"), ("PANW", "Palo Alto Networks"), ("CRWD", "CrowdStrike"),
    ("PLTR", "Palantir"), ("UBER", "Uber"), ("SHOP", "Shopify"),
    ("COIN", "Coinbase"), ("JPM", "JPMorgan"), ("BAC", "Bank of America"),
    ("GS", "Goldman Sachs"), ("V", "Visa"), ("MA", "Mastercard"),
    ("WMT", "Walmart"), ("COST", "Costco"), ("HD", "Home Depot"),
    ("LLY", "Eli Lilly"), ("UNH", "UnitedHealth"), ("JNJ", "Johnson & Johnson"),
    ("XOM", "Exxon Mobil"), ("CVX", "Chevron"), ("CAT", "Caterpillar"),
    ("GE", "GE Aerospace"), ("BA", "Boeing"),
]
US_LOOKUP = {ticker: name for ticker, name in US_QUICK_POOL}

st.set_page_config(
    page_title="I-Cheng Stock Bot",
    page_icon="📈",
    layout="wide",
)

if "favorites_combined" not in st.session_state:
    st.session_state["favorites_combined"] = []
if "tw_stock_id" not in st.session_state:
    st.session_state["tw_stock_id"] = "2330"
if "us_stock_id" not in st.session_state:
    st.session_state["us_stock_id"] = "AAPL"

# =========================================================
# Shared Watchlist (Taiwan + U.S.)
# =========================================================
def load_combined_favorites() -> list[dict]:
    return list(st.session_state.get("favorites_combined", []))


def save_combined_favorites(items: list[dict]) -> None:
    cleaned = []
    seen = set()
    for item in items:
        market = item.get("market")
        ticker = str(item.get("ticker", "")).strip().upper()
        if market not in ("TW Stock", "US Stock") or not ticker:
            continue
        key = (market, ticker)
        if key not in seen:
            seen.add(key)
            cleaned.append({"market": market, "ticker": ticker})
    st.session_state["favorites_combined"] = cleaned


def add_combined_favorite(ticker: str, market: str) -> bool:
    ticker = str(ticker).strip().upper()
    if market == "TW Stock":
        valid = bool(re.fullmatch(r"\d{4,6}", ticker))
    else:
        valid = bool(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker))
    if not valid:
        return False

    favs = load_combined_favorites()
    if not any(x["market"] == market and x["ticker"] == ticker for x in favs):
        favs.append({"market": market, "ticker": ticker})
        save_combined_favorites(favs)
    return True


def remove_combined_favorite(ticker: str, market: str) -> None:
    favs = [
        x for x in load_combined_favorites()
        if not (x["market"] == market and x["ticker"] == ticker)
    ]
    save_combined_favorites(favs)


def navigate_to_favorite(ticker: str, market: str) -> None:
    """Widget callback executes before Streamlit instantiates widgets on the next rerun."""
    st.session_state["top_market_selector"] = market
    if market == "TW Stock":
        st.session_state["tw_subpage"] = "Individual Stock Analysis"
        st.session_state["tw_stock_id"] = ticker
    else:
        st.session_state["us_subpage"] = "Individual Stock Analysis"
        st.session_state["us_stock_id"] = ticker


# =========================================================
# Sidebar Navigation: Select Market, Then Market-Specific Function
# =========================================================
st.sidebar.title("📈 I-Cheng Stock Bot")

selected_market = st.sidebar.radio(
    "Market",
    ["TW Stock", "US Stock"],
    horizontal=True,
    key="top_market_selector",
)

if selected_market == "TW Stock":
    subpage = st.sidebar.radio(
        "Taiwan stock tools",
        ["Individual Stock Analysis", "Full-Market Screening", "Scoring Rules"],
        key="tw_subpage",
    )
else:
    subpage = st.sidebar.radio(
        "U.S. stock tools",
        ["Individual Stock Analysis", "Market Overview", "Scoring Rules"],
        key="us_subpage",
    )

# =========================================================
# Shared Watchlist in Sidebar
# =========================================================
st.sidebar.divider()
st.sidebar.subheader("Watchlist")

fav_market = st.sidebar.selectbox(
    "Market for this ticker",
    ["TW Stock", "US Stock"],
    key="fav_add_market",
)
fav_placeholder = "e.g., 2330" if fav_market == "TW Stock" else "e.g., AAPL or MU"
fav_input = st.sidebar.text_input(
    "Ticker",
    placeholder=fav_placeholder,
    key="fav_add_input",
)

if st.sidebar.button("Add to watchlist", use_container_width=True):
    sid = fav_input.strip().upper()
    if add_combined_favorite(sid, fav_market):
        st.sidebar.success(f"Added {resolve_name_by_market(sid, fav_market)} ({sid})")
        st.rerun()
    else:
        st.sidebar.warning("Enter a valid ticker for the selected market.")

favorites = load_combined_favorites()
if not favorites:
    st.sidebar.caption("Add a ticker to start your watchlist.")
else:
    for item in favorites:
        f_market = item["market"]
        sid = item["ticker"]
        name = resolve_name_by_market(sid, f_market)
        c1, c2 = st.sidebar.columns([4, 1])
        c1.button(
            f"{f_market} {name} ({sid})",
            key=f"fav_open_{f_market}_{sid}",
            use_container_width=True,
            on_click=navigate_to_favorite,
            args=(sid, f_market),
        )

        if c2.button("✕", key=f"fav_del_{f_market}_{sid}", help=f"Remove {name}"):
            remove_combined_favorite(sid, f_market)
            st.rerun()

st.sidebar.caption("Keep Taiwan and U.S. stocks in one watchlist for your current session.")

# =========================================================
# Page Header
# =========================================================
st.title("📈 I-Cheng Stock Bot")
st.caption(
    "Analyze individual stocks, scan the Taiwan market, or follow major U.S. ETFs and actively traded stocks. "
    "Use one watchlist for both markets. You can access up to 5 years of daily price history or 10 years of weekly price history. AI analysis is optional."
)


def render_stock_analysis(selected_market: str):
    is_tw = selected_market == "TW Stock"
    stock_key = "tw_stock_id" if is_tw else "us_stock_id"

    col1, col2 = st.columns([1, 3])
    with col1:
        stock_id = st.text_input(
            "Ticker",
            key=stock_key,
            max_chars=10,
            placeholder="e.g., 2330" if is_tw else "e.g., AAPL, MU, or NVDA",
        ).strip().upper()

        interval_label = st.selectbox(
            "Chart interval",
            ["Daily", "Weekly", "60m", "15m", "5m"],
            key=f"interval_{selected_market}",
        )
        interval_map = {"Daily": "1d", "Weekly": "1wk", "60m": "60m", "15m": "15m", "5m": "5m"}
        period_map = {
            "5m": "1mo",
            "15m": "1mo",
            "60m": "1y",
            "1d": "5y",
            "1wk": "10y",
        }

        chart_range = st.selectbox(
            "Chart range",
            ["6 months", "1 year", "3 years", "5 years", "All"],
            index=0,
            key=f"chart_range_{selected_market}",
        )
        chart_range_map = {
            "6 months": 126,
            "1 year": 252,
            "3 years": 756,
            "5 years": 1260,
            "All": None,
        }

        st.button(
            "🔍 Analyze",
            use_container_width=True,
            type="primary",
            key=f"search_{selected_market}",
        )

        if st.button(
            "Add to watchlist",
            use_container_width=True,
            key=f"add_current_{selected_market}",
        ):
            if add_combined_favorite(stock_id, selected_market):
                st.success("Added to your watchlist.")
                st.rerun()

    if not stock_id:
        return

    interval = interval_map[interval_label]
    with st.spinner("Loading price data..."):
        df = fetch_market_df(stock_id, selected_market, interval, period_map[interval])

    if df.empty:
        example = "2330" if is_tw else "MU, AAPL, or NVDA"
        st.error(f"No price data found. Check the ticker and try again (e.g., {example}).")
        return

    ind = calculate_indicators(df)
    ev = evaluate_score(ind)
    strategy = build_trading_strategy(ind, ev)
    pct_ch = (ind["c_now"] - ind["c_prev"]) / max(ind["c_prev"], 1e-10) * 100
    name = resolve_name_by_market(stock_id, selected_market)

    st.subheader(f"{name} ({stock_id})")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Latest price", f"{ind['c_now']:.2f}", f"{pct_ch:+.2f}%")
    m2.metric("Technical score", f"{ev['score']:+.1f} / 5")
    m3.metric("RSI", f"{ind['rsi']:.1f}")
    m4.metric("Volume ratio", f"{ev['vol_ratio']:.2f}x")
    m5.metric("Bullish strength", f"{ev['bull_pct']:.0f}%")

    chart_limit = chart_range_map[chart_range]
    chart_df = df if chart_limit is None else df.tail(chart_limit)

    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.plotly_chart(price_chart(chart_df, ind), use_container_width=True)
    with c2:
        st.plotly_chart(score_gauge(ev["score"]), use_container_width=True)
        st.info(f"**Strategy: {strategy['stance']}**")
        st.write(f"**Considering an entry?** {strategy['new_entry']}")
        st.write(f"**Already holding the stock?** {strategy['holding']}")
        st.write(f"**Risk note:** {strategy['warning']}")

    t1, t2, t3 = st.tabs(["Technical indicators", "Scoring signals", "Risk"])
    with t1:
        st.dataframe(pd.DataFrame({
            "Indicator": ["MA5", "MA20", "MA60", "Upper Bollinger Band", "Lower Bollinger Band", "RSI", "K", "D", "DIF", "MACD", "ATR"],
            "Value": [
                ind["ma5"], ind["ma20"], ind["ma60"], ind["bb_up"], ind["bb_low"],
                ind["rsi"], ind["k_now"], ind["d_now"], ind["dif_now"], ind["macd_now"], ind["atr"],
            ],
        }), use_container_width=True, hide_index=True)
    with t2:
        st.success("Bullish signals: " + (" · ".join(ev["bull_signals"]) if ev["bull_signals"] else "None"))
        st.error("Bearish signals: " + (" · ".join(ev["bear_signals"]) if ev["bear_signals"] else "None"))
    with t3:
        risk_pct = abs(ind["c_now"] - strategy["stop"]) / ind["c_now"] * 100
        reward_pct = abs(strategy["target"] - ind["c_now"]) / ind["c_now"] * 100
        rr = reward_pct / max(risk_pct, 0.5)
        r1, r2, r3 = st.columns(3)
        r1.metric("Stop level", f"{strategy['stop']:.2f}", f"-{risk_pct:.1f}%")
        r2.metric("Price target", f"{strategy['target']:.2f}", f"+{reward_pct:.1f}%")
        r3.metric("Reward / risk", f"{rr:.1f} : 1")

    st.divider()
    st.subheader("AI analysis")
    st.caption("Unlock AI access and select Generate AI analysis to request a short- and long-term view from OpenAI.")

    if render_ai_login():
        ai_key = f"ai_advice::{selected_market}::{stock_id}::{interval}"
        if not get_secret("OPENAI_API_KEY"):
            st.info("OPENAI_API_KEY is not configured. Add it in Streamlit App Settings → Secrets.")
        else:
            if st.button(
                "Generate AI analysis",
                type="primary",
                key=f"ai_btn_{selected_market}_{stock_id}_{interval}",
            ):
                try:
                    with st.spinner("Generating AI analysis..."):
                        st.session_state[ai_key] = generate_ai_advice(
                            stock_id, name, interval_label, ind, ev, strategy, selected_market
                        )
                except Exception as e:
                    st.error(f"Couldn't generate the AI analysis. {e}")

            if st.session_state.get(ai_key):
                st.markdown(st.session_state[ai_key])
                if st.button(
                    "Regenerate AI analysis",
                    key=f"ai_refresh_{selected_market}_{stock_id}_{interval}",
                ):
                    try:
                        with st.spinner("Regenerating AI analysis..."):
                            st.session_state[ai_key] = generate_ai_advice(
                                stock_id, name, interval_label, ind, ev, strategy, selected_market
                            )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Couldn't regenerate the AI analysis. {e}")


# =========================================================
# TW Stock Functions
# =========================================================
if selected_market == "TW Stock":
    if subpage == "Individual Stock Analysis":
        st.subheader("Taiwan stock analysis")
        render_stock_analysis("TW Stock")

    elif subpage == "Full-Market Screening":
        st.subheader("Taiwan market screening and breadth")
        st.caption(
            "The scan covers listed and OTC common stocks in Taiwan. It filters for liquidity and ranks eligible stocks by trend, returns, and trading volume. "
            "The full technical score is calculated for up to 50 top-ranked stocks. Market breadth uses the same market-wide data."
        )

        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Rescan the Taiwan market", type="primary", use_container_width=True):
                run_full_market_scan.clear()
                first_stage_market_scan.clear()
                st.rerun()
        with c2:
            st.info("The app reuses scan results for 30 minutes. Select Rescan the Taiwan market to refresh them now.")

        with st.spinner("Scanning the Taiwan market. The first scan may take a while."):
            scan = run_full_market_scan()

        pct = scan["above_ma20"] / max(scan["valid_total"], 1) * 100
        status = "🟢 Broad participation" if pct >= 50 else "🟡 Mixed participation" if pct >= 30 else "🔴 Weak participation"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stocks above MA20", f"{pct:.1f}%")
        m2.metric("Stocks with usable data", scan["valid_total"])
        m3.metric("Stocks shortlisted", scan["candidates"])
        m4.metric("Last updated", scan["updated"])

        st.write(f"**Market breadth: {status}**")
        st.caption(
            "Market breadth measures the share of stocks trading above their 20-day moving averages. "
            "A higher share means more stocks are participating in the market's strength."
        )

        tab1, tab2 = st.tabs(["Strong stocks", "Bullish pullbacks"])
        with tab1:
            if scan["strong"].empty:
                st.info("No stocks meet the strong-stock criteria in this scan.")
            else:
                st.dataframe(scan["strong"], use_container_width=True, hide_index=True)
        with tab2:
            if scan["pullback"].empty:
                st.info("No stocks meet the bullish pullback criteria in this scan.")
            else:
                st.dataframe(scan["pullback"], use_container_width=True, hide_index=True)

    elif subpage == "Scoring Rules":
        st.subheader("How Taiwan stocks are scored")
        st.markdown(
            """
**1. Moving-average trend (MA5 / MA20 / MA60)**
- Price > MA5 > MA20 > MA60 (+1.5)
- Price above MA20 (+1.0)
- Price < MA5 < MA20 < MA60 (-1.5)
- Price below MA20 (-1.0)

**2. RSI (14)**
- 55 to 75 (+1.0)
- Below 30 (+1.0)
- Above 80 (-1.0)
- Below 45 (-0.5)

**3. KD (9)**
- K crosses above D (golden cross, +1.0)
- K > D (+0.5)
- K crosses below D (death cross, -1.0)
- K < D (-0.5)

**4. MACD**
- DIF > MACD (+1.0)
- DIF < MACD (-1.0)

**5. Volume and Bollinger Bands**
- Price rises on at least 1.5× the 20-day average volume (+1.0)
- Price below the lower Bollinger Band (+0.5)
- Price above the upper Bollinger Band (-0.5)
            """
        )

# =========================================================
# US Stock Functions
# =========================================================
else:
    if subpage == "Individual Stock Analysis":
        st.subheader("U.S. stock analysis")
        st.caption("Enter a Yahoo Finance ticker such as AAPL, NVDA, MU, or TSLA.")
        render_stock_analysis("US Stock")

    elif subpage == "Market Overview":
        st.subheader("U.S. market overview")
        st.caption(
            "SPY, QQQ, DIA, and IWM show how major U.S. indexes are moving. "
            "The stock scan covers a selected group of actively traded stocks."
        )

        if st.button("Refresh U.S. market data", type="primary"):
            get_us_index_overview.clear()
            run_us_scan_cached.clear()
            st.rerun()

        st.markdown("### Major U.S. ETFs")
        index_rows = get_us_index_overview()
        if index_rows:
            cols = st.columns(len(index_rows))
            for col, row in zip(cols, index_rows):
                with col:
                    st.metric(
                        f"{row['ticker']} · {row['name']}",
                        f"${row['price']:.2f}",
                        f"{row['pct_ch']:+.2f}%",
                    )
                    st.caption(f"{row['trend']} | Score {row['score']:+.1f} | RSI {row['rsi']:.1f}")
        else:
            st.info("U.S. ETF data is unavailable right now.")

        st.divider()
        st.markdown("### U.S. stock scan")
        with st.spinner("Scanning U.S. stocks..."):
            us_scan = run_us_scan_cached()

        us_pct = us_scan["above_ma20"] / max(us_scan["total"], 1) * 100
        u1, u2, u3 = st.columns(3)
        u1.metric("Scanned stocks above MA20", f"{us_pct:.1f}%")
        u2.metric("Stocks with usable data", us_scan["total"])
        u3.metric("Last updated", us_scan["time"])

        tab1, tab2 = st.tabs(["Strong stocks", "Pullbacks"])
        with tab1:
            if us_scan["strong"]:
                df_us_strong = pd.DataFrame(us_scan["strong"]).copy()
                df_us_strong = df_us_strong.rename(columns={
                    "ticker": "Ticker", "name": "Name", "price": "Current Price", "pct_ch": "Change %",
                    "score": "Score", "rsi": "RSI", "vol_ratio": "Volume Ratio"
                })
                show_cols = [c for c in ["Ticker", "Name", "Current Price", "Change %", "Score", "RSI", "Volume Ratio"] if c in df_us_strong.columns]
                st.dataframe(df_us_strong[show_cols], use_container_width=True, hide_index=True)
            else:
                st.info("No stocks meet the strong-stock criteria in this scan.")

        with tab2:
            if us_scan["pullback"]:
                df_us_pull = pd.DataFrame(us_scan["pullback"]).copy()
                df_us_pull = df_us_pull.rename(columns={
                    "ticker": "Ticker", "name": "Name", "price": "Current Price", "pct_ch": "Change %",
                    "score": "Score", "rsi": "RSI", "vol_ratio": "Volume Ratio"
                })
                show_cols = [c for c in ["Ticker", "Name", "Current Price", "Change %", "Score", "RSI", "Volume Ratio"] if c in df_us_pull.columns]
                st.dataframe(df_us_pull[show_cols], use_container_width=True, hide_index=True)
            else:
                st.info("No stocks meet the pullback criteria in this scan.")

    elif subpage == "Scoring Rules":
        st.subheader("How U.S. stocks are scored (-5 to +5)")
        st.caption(
            "The U.S. and Taiwan tools use the same technical scoring rules. "
            "A higher score signals stronger bullish momentum. Consider market conditions, company fundamentals, and risk alongside the score."
        )
        st.markdown(
            """
### Moving-average trend (MA5 / MA20 / MA60)

Compare the price with its short- and medium-term moving averages.

- **Price > MA5 > MA20 > MA60** earns +1.5 for a bullish alignment.
- **Price above MA20** earns +1.0.
- **Price < MA5 < MA20 < MA60** gets -1.5 for a bearish alignment.
- **Price below MA20** gets -1.0.

### Relative Strength Index (RSI, 14)

- **RSI from 55 to 75** earns +1.0 for stronger buying momentum.
- **RSI < 30** earns +1.0. The stock is oversold and may rebound.
- **RSI > 80** gets -1.0. The stock is overbought, which makes chasing the price riskier.
- **RSI < 45** gets -0.5 for weak momentum.

### KD stochastic indicator (9)

- **K crosses above D (golden cross)** earns +1.0 for a short-term bullish signal.
- **K > D** earns +0.5.
- **K crosses below D (death cross)** gets -1.0 for a short-term bearish signal.
- **K < D** gets -0.5.

### MACD (12 / 26 / 9)

- **DIF > MACD** earns +1.0 for bullish momentum.
- **DIF < MACD** gets -1.0 for bearish momentum.

### Volume and Bollinger Bands

- **Price rises on at least 1.5× the 20-day average volume** earns +1.0.
- **Price below the lower Bollinger Band** earns +0.5. Watch for a possible rebound.
- **Price above the upper Bollinger Band** gets -0.5. The price may be stretched, which makes chasing it riskier.

### How to read the score

- **+3.0 to +5.0 (strongly bullish).** Most trend and momentum signals are bullish. Avoid chasing the price.
- **+1.5 to +2.5 (bullish).** A pullback toward support may offer a better entry.
- **+0.5 to +1.0 (mildly bullish).** The trend is positive and still developing.
- **-0.4 to +0.4 (neutral or range-bound).** Bullish and bearish signals are roughly balanced. Wait for a clearer direction.
- **-2.0 to -0.5 (weak).** Short-term momentum is weak, so keep position size in check.
- **-5.0 to -2.5 (bearish).** Consider waiting for signs of recovery.

> The score uses technical indicators only. It leaves out full financial statements, valuation, macroeconomic conditions, and real-time news. For long-term investing, also consider company fundamentals and industry trends.
            """
        )

st.divider()
st.caption(
    "⚠️ Use this site and its AI analysis for quantitative research, technical analysis, coursework, and portfolio demonstrations. "
    "Check AI results independently because they can contain errors. This material provides general information only, with no guarantee of returns or personalized investment advice."
)
