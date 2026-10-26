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
# 基本設定
# =========================================================
TW_TZ = timezone(timedelta(hours=8))
SCAN_CACHE_MINUTES = 30
FIRST_STAGE_KEEP = 50
BATCH_SIZE = 80
MIN_AVG_TURNOVER = 8_000_000
STRONG_BULL_SCORE = 2.0
PULLBACK_MIN_SCORE = 0.0


# =========================================================
# 股票池 / 資料抓取
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
# 技術指標與評分（沿用原 Telegram Bot 邏輯）
# =========================================================
def calculate_indicators(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["Close", "High", "Low"]).copy()
    if len(df) < 25:
        raise ValueError("有效股價資料不足")

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
        bull_signals.append("均線多頭排列 (+1.5)")
    elif c > ma20:
        score += 1.0
        bull_signals.append("站上20日均線 (+1.0)")
    elif c < ma5 < ma20 < ma60:
        score -= 1.5
        bear_signals.append("均線空頭排列 (-1.5)")
    else:
        score -= 1.0
        bear_signals.append("跌破20日均線 (-1.0)")

    rsi = ind["rsi"]
    if 55 <= rsi <= 75:
        score += 1.0
        bull_signals.append("RSI 強勢多頭區 (+1.0)")
    elif rsi < 30:
        score += 1.0
        bull_signals.append("RSI 超跌反彈訊號 (+1.0)")
    elif rsi > 80:
        score -= 1.0
        bear_signals.append("RSI 過熱超買 (-1.0)")
    elif rsi < 45:
        score -= 0.5
        bear_signals.append("RSI 動能偏弱 (-0.5)")

    k_n, d_n = ind["k_now"], ind["d_now"]
    k_p, d_p = ind["k_prev"], ind["d_prev"]
    if k_p < d_p and k_n > d_n:
        score += 1.0
        bull_signals.append("KD 黃金交叉 (+1.0)")
    elif k_n > d_n:
        score += 0.5
        bull_signals.append("KD 多頭格局 (+0.5)")
    elif k_p > d_p and k_n < d_n:
        score -= 1.0
        bear_signals.append("KD 死亡交叉 (-1.0)")
    else:
        score -= 0.5
        bear_signals.append("KD 空頭整理 (-0.5)")

    if ind["dif_now"] > ind["macd_now"]:
        score += 1.0
        bull_signals.append("MACD 多頭柱狀 (+1.0)")
    else:
        score -= 1.0
        bear_signals.append("MACD 空頭柱狀 (-1.0)")

    vol_ratio = ind["vol_now"] / ind["vol_ma20"]
    if vol_ratio >= 1.5 and c > ind["c_prev"]:
        score += 1.0
        bull_signals.append(f"爆量上攻 {vol_ratio:.1f}倍均量 (+1.0)")
    elif c < ind["bb_low"]:
        score += 0.5
        bull_signals.append("觸及布林下軌支撐 (+0.5)")
    elif c > ind["bb_up"]:
        score -= 0.5
        bear_signals.append("突破布林上軌乖離過大 (-0.5)")

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
        stance = "🟢 偏多續抱"
        new_entry = f"不追高；拉回 {buy_low:.2f}～{buy_high:.2f} 附近止跌可分批進場"
        holding = f"續抱為主；接近 {target:.2f} 可分批停利，跌破 {stop:.2f} 減碼或停損"
    elif score >= 1.5 and c > ma20:
        stance = "🟡 拉回布局"
        new_entry = f"等待回測 {buy_low:.2f}～{buy_high:.2f}，確認止跌再分批進場"
        holding = f"可續抱但不追價；跌破月線 {ma20:.2f} 先減碼，{stop:.2f} 為防守線"
    elif score >= 0.5:
        stance = "🟡 小量試單 / 等確認"
        new_entry = f"先小量，站穩月線 {ma20:.2f} 且量能轉強再加碼"
        holding = f"續抱觀察；無法站穩 {ma20:.2f} 可降低部位"
    elif score <= -2.5:
        stance = "🔴 偏空避開"
        new_entry = f"暫不進場；等重新站回月線 {ma20:.2f} 且 KD/MACD 改善"
        holding = f"反彈以減碼為主；跌破 {stop:.2f} 嚴格停損"
    elif score <= -0.5:
        stance = "🟠 弱勢觀望"
        new_entry = f"先等止跌，重新站回月線 {ma20:.2f} 再評估"
        holding = f"控制部位；跌破 {stop:.2f} 減碼，站回月線再轉積極"
    else:
        stance = "⚪ 區間整理"
        new_entry = f"靠近 {buy_low:.2f}～{buy_high:.2f} 才考慮，突破區間後再追蹤"
        holding = f"續抱但不加碼；跌破 {stop:.2f} 退出，接近 {target:.2f} 分批停利"

    warns = []
    if rsi >= 75:
        warns.append("RSI 偏高，避免追價")
    if vol_ratio >= 2.0:
        warns.append("成交量明顯放大，留意隔日震盪")
    if c < ma60:
        warns.append("仍在季線下，中期趨勢偏弱")

    return {
        "stance": stance,
        "new_entry": new_entry,
        "holding": holding,
        "warning": "；".join(warns) if warns else "無額外技術面警示",
        "stop": stop,
        "target": target,
    }


# =========================================================
# 自選股（每位訪客各自的 Streamlit Session，不共用主機檔案）
# =========================================================
def load_favorites() -> list[str]:
    """讀取目前訪客、目前市場的自選股。"""
    if "favorites_by_market" not in st.session_state:
        st.session_state["favorites_by_market"] = {"🇹🇼 台股": [], "🇺🇸 美股": []}
    if market not in st.session_state["favorites_by_market"]:
        st.session_state["favorites_by_market"][market] = []
    return list(st.session_state["favorites_by_market"][market])


def save_favorites(items: list[str]) -> None:
    """只儲存在目前訪客的 session，不寫入共用 JSON。"""
    cleaned = []
    for x in items:
        x = str(x).strip().upper()
        valid = (
            re.fullmatch(r"\d{4,6}", x) if market == "🇹🇼 台股"
            else re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", x)
        )
        if valid and x not in cleaned:
            cleaned.append(x)
    st.session_state["favorites_by_market"][market] = cleaned


def add_favorite(stock_id: str) -> bool:
    stock_id = str(stock_id).strip().upper()
    valid = (
        re.fullmatch(r"\d{4,6}", stock_id) if market == "🇹🇼 台股"
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
# 市場切換：台股 / 美股
# =========================================================
def normalize_symbol_by_market(stock_id: str, market: str) -> str:
    stock_id = stock_id.strip().upper()
    if market == "🇺🇸 美股":
        return stock_id
    return stock_id


def resolve_name_by_market(stock_id: str, market: str) -> str:
    sid = stock_id.strip().upper()
    if market == "🇺🇸 美股":
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
    """市場共用抓價。台股沿用 .TW/.TWO；美股直接使用 Yahoo ticker。"""
    sid = stock_id.strip().upper()
    if market == "🇺🇸 美股":
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

    # 台股：使用既有函式
    try:
        # 既有 fetch_stock_df 有些版本是 async；Streamlit 版通常為同步函式。
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
        df = fetch_market_df(ticker, "🇺🇸 美股", interval="1d", period="1y")
        if df.empty or len(df) < 25:
            continue
        try:
            ind = calculate_indicators(df)
            ev = evaluate_score(ind)
            pct = (ind["c_now"] - ind["c_prev"]) / max(ind["c_prev"], 1e-9) * 100
            trend = (
                "🟢 偏多" if ind["c_now"] > ind["ma20"] > ind["ma60"]
                else "🔴 偏空" if ind["c_now"] < ind["ma20"] < ind["ma60"]
                else "🟡 震盪"
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


def run_us_scan_cached() -> dict:
    """美股掃描中心：掃描高流動性代表性股票池。個股分析本身不限此清單。"""
    rows = []
    above_ma20 = 0
    total = 0

    for ticker, name in US_QUICK_POOL:
        df = fetch_market_df(ticker, "🇺🇸 美股", interval="1d", period="6mo")
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
# AI 存取密碼保護
# =========================================================
def ai_access_unlocked() -> bool:
    return bool(st.session_state.get("ai_access_unlocked", False))


def render_ai_login() -> bool:
    expected_password = get_secret("AI_ACCESS_PASSWORD")

    if not expected_password:
        st.warning("尚未設定 AI_ACCESS_PASSWORD，AI 功能目前鎖定。")
        return False

    if ai_access_unlocked():
        c1, c2 = st.columns([4, 1])
        with c1:
            st.success("🔓 AI 功能已解鎖")
        with c2:
            if st.button("登出 AI", key="ai_logout_btn"):
                st.session_state["ai_access_unlocked"] = False
                st.rerun()
        return True

    password = st.text_input(
        "AI 使用密碼",
        type="password",
        placeholder="輸入密碼後才能使用 AI",
        key="ai_access_password_input",
    )

    if st.button("🔐 解鎖 AI 功能", key="ai_unlock_btn"):
        if password == expected_password:
            st.session_state["ai_access_unlocked"] = True
            st.success("AI 功能已解鎖。")
            st.rerun()
        else:
            st.error("密碼不正確。")

    return False


# =========================================================
# AI 分析（OpenAI Responses API）
# =========================================================
def get_secret(name: str, default=None):
    """Streamlit Cloud 優先讀取 Secrets；本機則可使用環境變數。"""
    try:
        value = st.secrets.get(name, default)
        if value is not None:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fundamentals(stock_id: str, market: str = "🇹🇼 台股") -> dict:
    """抓取可用的基本面摘要；Yahoo 無資料時回傳空值，不影響技術分析。"""
    out = {}
    if market == "🇺🇸 美股":
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
        "marketCap": "市值",
        "trailingPE": "本益比(TTM)",
        "forwardPE": "預估本益比",
        "priceToBook": "股價淨值比",
        "revenueGrowth": "營收年增率",
        "earningsGrowth": "獲利年增率",
        "returnOnEquity": "ROE",
        "debtToEquity": "負債權益比",
        "profitMargins": "淨利率",
        "dividendYield": "股息殖利率",
    }
    for key, label in fields.items():
        value = info.get(key)
        if value is not None:
            out[label] = value
    return out


def build_ai_payload(stock_id: str, stock_name: str, interval_label: str, ind: dict, ev: dict, strategy: dict, fundamentals: dict) -> dict:
    return {
        "股票": f"{stock_name} ({stock_id})",
        "K線週期": interval_label,
        "資料時間": datetime.datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "技術資料": {
            "現價": round(ind["c_now"], 2),
            "前一根收盤": round(ind["c_prev"], 2),
            "MA5": round(ind["ma5"], 2),
            "MA20": round(ind["ma20"], 2),
            "MA60": round(ind["ma60"], 2),
            "布林上軌": round(ind["bb_up"], 2),
            "布林下軌": round(ind["bb_low"], 2),
            "RSI": round(ind["rsi"], 1),
            "K": round(ind["k_now"], 1),
            "D": round(ind["d_now"], 1),
            "DIF": round(ind["dif_now"], 3),
            "MACD": round(ind["macd_now"], 3),
            "ATR": round(ind["atr"], 2),
            "量比": round(ev["vol_ratio"], 2),
            "量化評分": ev["score"],
            "多頭強度%": round(ev["bull_pct"], 0),
            "看多訊號": ev["bull_signals"],
            "看空訊號": ev["bear_signals"],
        },
        "原量化策略": {
            "定位": strategy["stance"],
            "新進場": strategy["new_entry"],
            "已持有": strategy["holding"],
            "注意": strategy["warning"],
            "停損參考": round(strategy["stop"], 2),
            "目標參考": round(strategy["target"], 2),
        },
        "可取得基本面": fundamentals,
    }


def generate_ai_advice(stock_id: str, stock_name: str, interval_label: str, ind: dict, ev: dict, strategy: dict, market: str = "🇹🇼 台股") -> str:
    api_key = get_secret("OPENAI_API_KEY")
    model = get_secret("OPENAI_MODEL", "gpt-6-luna")

    if not api_key:
        raise RuntimeError("尚未設定 OPENAI_API_KEY。")
    if OpenAI is None:
        raise RuntimeError("尚未安裝 openai 套件，請更新 requirements.txt。")

    fundamentals = fetch_fundamentals(stock_id, market)
    payload = build_ai_payload(stock_id, stock_name, interval_label, ind, ev, strategy, fundamentals)

    instructions = """
你是台股研究助理。只能根據使用者提供的量化技術資料與可取得的基本面摘要分析，
不得假裝知道未提供的即時新聞、法人籌碼、財報細節或未提供的價格資料。
請用繁體中文，內容清楚、具體，但不要保證獲利。

請固定輸出：
## ⚡ 短線觀點
- 趨勢判斷：
- 新進場策略：
- 已持有策略：
- 關鍵支撐／轉弱條件：
- 短線風險：

## 🌳 長線觀點
- 中長期趨勢：
- 基本面可見資訊：
- 適合的操作方式：
- 長線轉強條件：
- 長線風險：

## 🎯 AI 結論
用 2～4 句總結「偏多／中性／偏空」以及最重要的觀察點。

短線以數天到數週為主；長線以數月以上為主。
若基本面資料不足，要明確寫「基本面資料不足」，不要用技術指標冒充完整基本面分析。
"""

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
    )
    return response.output_text.strip()


# =========================================================
# 圖表
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
        name="K線"
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
        title={"text": "多因子動能評分"},
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
# 全市場掃描 / 大盤廣度
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
    """候選股第二階段批次下載，避免逐檔發送 50 次 Yahoo request。"""
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
                "代號": cand["ticker"],
                "名稱": cand["name"],
                "現價": round(ind["c_now"], 2),
                "評分": ev["score"],
                "RSI": round(ind["rsi"], 1),
                "20日漲幅%": round(cand["ret20"], 1),
                "量比": round(cand["vol_ratio"], 2),
                "快篩分": round(cand["fast_score"], 2),
            }
            if ev["score"] >= STRONG_BULL_SCORE and ind["c_now"] > ind["ma20"]:
                strong.append(row)
            elif PULLBACK_MIN_SCORE <= ev["score"] < STRONG_BULL_SCORE and ind["c_now"] > ind["ma60"] and pct_ch < 0:
                pullback.append(row)
        except Exception:
            continue

    strong.sort(key=lambda x: x["評分"], reverse=True)
    pullback.sort(key=lambda x: x["評分"], reverse=True)

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

# 美股熱門股雷達清單（不是全美股市場）
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

if "favorites_by_market" not in st.session_state:
    st.session_state["favorites_by_market"] = {"🇹🇼 台股": [], "🇺🇸 美股": []}
if "tw_stock_id" not in st.session_state:
    st.session_state["tw_stock_id"] = "2330"
if "us_stock_id" not in st.session_state:
    st.session_state["us_stock_id"] = "AAPL"

# -------------------------
# 左側功能選單：台股 / 美股完全分開
# -------------------------
st.sidebar.title("📈 I-Cheng Stock Bot")
page = st.sidebar.radio(
    "功能選單",
    [
        "🇹🇼 台股個股分析",
        "🇹🇼 台股全市場掃描",
        "🇺🇸 美股個股分析",
        "🇺🇸 美股市場總覽",
        "📖 評分規則",
    ],
    key="page_select_v2",
)

market = "🇺🇸 美股" if page.startswith("🇺🇸") else "🇹🇼 台股"

# -------------------------
# 左側自選股：依目前頁面市場分開
# -------------------------
if page in ("🇹🇼 台股個股分析", "🇺🇸 美股個股分析"):
    st.sidebar.divider()
    st.sidebar.subheader("⭐ 我的自選股")

    placeholder = "例如 2330" if market == "🇹🇼 台股" else "例如 AAPL / MU"
    sidebar_add = st.sidebar.text_input(
        "加入股票代號",
        placeholder=placeholder,
        key=f"sidebar_add_{market}",
    )

    if st.sidebar.button("➕ 加入自選", use_container_width=True, key=f"addfav_{market}"):
        sid = sidebar_add.strip().upper()
        if add_favorite(sid):
            st.sidebar.success(f"已加入 {resolve_name_by_market(sid, market)} ({sid})")
            st.rerun()
        else:
            st.sidebar.warning("請輸入正確股票代號。")

    favorites = load_favorites()
    if not favorites:
        st.sidebar.caption("目前尚無自選股。")
    else:
        for sid in favorites:
            name = resolve_name_by_market(sid, market)
            c1, c2 = st.sidebar.columns([4, 1])
            if c1.button(f"{name} ({sid})", key=f"fav_open_{market}_{sid}", use_container_width=True):
                if market == "🇹🇼 台股":
                    st.session_state["tw_stock_id"] = sid
                else:
                    st.session_state["us_stock_id"] = sid
                st.rerun()
            if c2.button("✕", key=f"fav_del_{market}_{sid}", help=f"移除 {name}"):
                remove_favorite(sid)
                st.rerun()

    st.sidebar.caption("自選股只存在目前訪客的使用階段，不與其他人共用。")

# -------------------------
# 頁首
# -------------------------
st.title("📈 I-Cheng Stock Bot")
st.caption(
    "台股全市場掃描｜美股市場總覽＋熱門股雷達｜日K最長5年、週K最長10年｜"
    "MA / RSI / KD / MACD / 布林通道 / 成交量 / AI 短線與長線觀點"
)


def render_stock_analysis(selected_market: str):
    is_tw = selected_market == "🇹🇼 台股"
    stock_key = "tw_stock_id" if is_tw else "us_stock_id"

    col1, col2 = st.columns([1, 3])
    with col1:
        stock_id = st.text_input(
            "股票代號",
            key=stock_key,
            max_chars=10,
            placeholder="例如 2330" if is_tw else "例如 AAPL / MU / NVDA",
        ).strip().upper()

        interval_label = st.selectbox(
            "K線週期",
            ["日K", "週K", "60分", "15分", "5分"],
            key=f"interval_{selected_market}",
        )
        interval_map = {"日K": "1d", "週K": "1wk", "60分": "60m", "15分": "15m", "5分": "5m"}
        period_map = {
            "5m": "1mo",
            "15m": "1mo",
            "60m": "1y",
            "1d": "5y",
            "1wk": "10y",
        }

        chart_range = st.selectbox(
            "K線顯示區間",
            ["6個月", "1年", "3年", "5年", "全部"],
            index=2,
            key=f"chart_range_{selected_market}",
        )
        chart_range_map = {
            "6個月": 126,
            "1年": 252,
            "3年": 756,
            "5年": 1260,
            "全部": None,
        }

        do_search = st.button(
            "🔍 開始分析",
            use_container_width=True,
            type="primary",
            key=f"search_{selected_market}",
        )

        if st.button("⭐ 加入目前股票到自選", use_container_width=True, key=f"add_current_{selected_market}"):
            if add_favorite(stock_id):
                st.success("已加入自選股")
                st.rerun()

    if not stock_id:
        return

    interval = interval_map[interval_label]
    if do_search or stock_id:
        with st.spinner("抓取股價與計算技術指標中..."):
            df = fetch_market_df(stock_id, selected_market, interval, period_map[interval])

        if df.empty:
            example = "2330" if is_tw else "MU、AAPL、NVDA"
            st.error(f"查無股價資料，請確認股票代號。範例：{example}")
            return

        ind = calculate_indicators(df)
        ev = evaluate_score(ind)
        strategy = build_trading_strategy(ind, ev)
        pct_ch = (ind["c_now"] - ind["c_prev"]) / max(ind["c_prev"], 1e-10) * 100
        name = resolve_name_by_market(stock_id, selected_market)

        st.subheader(f"{name} ({stock_id})")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("最新價格", f"{ind['c_now']:.2f}", f"{pct_ch:+.2f}%")
        m2.metric("綜合評分", f"{ev['score']:+.1f} / 5")
        m3.metric("RSI", f"{ind['rsi']:.1f}")
        m4.metric("量比", f"{ev['vol_ratio']:.2f}x")
        m5.metric("多頭強度", f"{ev['bull_pct']:.0f}%")

        chart_limit = chart_range_map[chart_range]
        chart_df = df if chart_limit is None else df.tail(chart_limit)

        c1, c2 = st.columns([2.2, 1])
        with c1:
            st.plotly_chart(price_chart(chart_df, ind), use_container_width=True)
        with c2:
            st.plotly_chart(score_gauge(ev["score"]), use_container_width=True)
            st.info(f"**策略定位：{strategy['stance']}**")
            st.write(f"**新進場：** {strategy['new_entry']}")
            st.write(f"**已持有：** {strategy['holding']}")
            st.write(f"**注意：** {strategy['warning']}")

        t1, t2, t3 = st.tabs(["技術指標", "評分觸發", "風控"])
        with t1:
            st.dataframe(pd.DataFrame({
                "指標": ["MA5", "MA20", "MA60", "布林上軌", "布林下軌", "RSI", "K", "D", "DIF", "MACD", "ATR"],
                "數值": [
                    ind["ma5"], ind["ma20"], ind["ma60"], ind["bb_up"], ind["bb_low"],
                    ind["rsi"], ind["k_now"], ind["d_now"], ind["dif_now"], ind["macd_now"], ind["atr"],
                ],
            }), use_container_width=True, hide_index=True)
        with t2:
            st.success("看多訊號：" + ("、".join(ev["bull_signals"]) if ev["bull_signals"] else "無"))
            st.error("看空訊號：" + ("、".join(ev["bear_signals"]) if ev["bear_signals"] else "無"))
        with t3:
            risk_pct = abs(ind["c_now"] - strategy["stop"]) / ind["c_now"] * 100
            reward_pct = abs(strategy["target"] - ind["c_now"]) / ind["c_now"] * 100
            rr = reward_pct / max(risk_pct, 0.5)
            r1, r2, r3 = st.columns(3)
            r1.metric("停損參考", f"{strategy['stop']:.2f}", f"-{risk_pct:.1f}%")
            r2.metric("目標參考", f"{strategy['target']:.2f}", f"+{reward_pct:.1f}%")
            r3.metric("風報比", f"{rr:.1f} : 1")

        st.divider()
        st.subheader("🤖 AI 短線／長線分析")
        st.caption("AI 為私人模式；只有輸入正確密碼後才會呼叫你的 OpenAI API。")

        if render_ai_login():
            ai_key = f"ai_advice::{selected_market}::{stock_id}::{interval}"
            if not get_secret("OPENAI_API_KEY"):
                st.info("尚未設定 OPENAI_API_KEY。請到 Streamlit App Settings → Secrets 加入金鑰。")
            else:
                if st.button("🤖 產生 AI 短線＋長線建議", type="primary", key=f"ai_btn_{selected_market}_{stock_id}_{interval}"):
                    try:
                        with st.spinner("AI 正在整理短線與長線觀點..."):
                            st.session_state[ai_key] = generate_ai_advice(
                                stock_id, name, interval_label, ind, ev, strategy, selected_market
                            )
                    except Exception as e:
                        st.error(f"AI 分析目前無法使用：{e}")

                if st.session_state.get(ai_key):
                    st.markdown(st.session_state[ai_key])
                    if st.button("🔄 重新產生 AI 建議", key=f"ai_refresh_{selected_market}_{stock_id}_{interval}"):
                        try:
                            with st.spinner("重新產生 AI 分析..."):
                                st.session_state[ai_key] = generate_ai_advice(
                                    stock_id, name, interval_label, ind, ev, strategy, selected_market
                                )
                            st.rerun()
                        except Exception as e:
                            st.error(f"AI 分析目前無法使用：{e}")


# -------------------------
# 🇹🇼 台股個股
# -------------------------
if page == "🇹🇼 台股個股分析":
    st.subheader("🇹🇼 台股個股分析")
    render_stock_analysis("🇹🇼 台股")

# -------------------------
# 🇹🇼 台股全市場掃描
# -------------------------
elif page == "🇹🇼 台股全市場掃描":
    st.subheader("🇹🇼 台股全市場掃描＋大盤廣度")
    st.caption(
        "一次掃描台灣上市＋上櫃普通股，先以趨勢、漲幅、量能與流動性做第一階段快篩，"
        "再對前 50 檔候選進行完整技術評分。大盤廣度與強勢股使用同一批全市場資料。"
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("🚀 立即全市場重新掃描", type="primary", use_container_width=True):
            run_full_market_scan.clear()
            first_stage_market_scan.clear()
            st.rerun()
    with c2:
        st.info("快取有效 30 分鐘；手動按重新掃描會立即清除快取並重新抓取。")

    with st.spinner("正在掃描台股全市場，第一次可能需要一些時間..."):
        scan = run_full_market_scan()

    pct = scan["above_ma20"] / max(scan["valid_total"], 1) * 100
    status = "🟢 多頭健康" if pct >= 50 else "🟡 震盪整理" if pct >= 30 else "🔴 空頭偏弱"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("站上 MA20 比例", f"{pct:.1f}%")
    m2.metric("有效市場樣本", scan["valid_total"])
    m3.metric("第一階段候選", scan["candidates"])
    m4.metric("資料更新時間", scan["updated"])

    st.write(f"**市場狀態：{status}**")
    st.caption(
        "大盤廣度＝全市場中股價站上 20 日均線的股票比例。比例越高，代表上漲有較多股票共同參與；"
        "比例偏低時，代表市場內部結構較弱。"
    )

    tab1, tab2 = st.tabs(["🔥 強勢多頭股", "🔻 多頭回調股"])
    with tab1:
        if scan["strong"].empty:
            st.info("目前沒有符合條件的強勢多頭股。")
        else:
            st.dataframe(scan["strong"], use_container_width=True, hide_index=True)
    with tab2:
        if scan["pullback"].empty:
            st.info("目前沒有符合條件的多頭回調股。")
        else:
            st.dataframe(scan["pullback"], use_container_width=True, hide_index=True)

# -------------------------
# 🇺🇸 美股個股
# -------------------------
elif page == "🇺🇸 美股個股分析":
    st.subheader("🇺🇸 美股個股分析")
    st.caption("直接輸入 Yahoo Finance 美股代號，例如 AAPL、NVDA、MU、TSLA。")
    render_stock_analysis("🇺🇸 美股")

# -------------------------
# 🇺🇸 美股市場總覽
# -------------------------
elif page == "🇺🇸 美股市場總覽":
    st.subheader("🇺🇸 美股市場總覽＋熱門股雷達")
    st.caption(
        "這裡不是全美股掃描。上半部以 SPY、QQQ、DIA、IWM 觀察主要市場方向；"
        "下半部使用高流動性熱門股建立強弱雷達。"
    )

    if st.button("🔄 更新美股市場資料", type="primary"):
        get_us_index_overview.clear()
        run_us_scan_cached.clear()
        st.rerun()

    st.markdown("### 📈 主要指數趨勢")
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
                st.caption(f"{row['trend']}｜評分 {row['score']:+.1f}｜RSI {row['rsi']:.1f}")
    else:
        st.info("目前無法取得美股主要指數資料。")

    st.divider()
    st.markdown("### 🔭 熱門股雷達")
    with st.spinner("整理美股熱門股資料中..."):
        us_scan = run_us_scan_cached()

    us_pct = us_scan["above_ma20"] / max(us_scan["total"], 1) * 100
    u1, u2, u3 = st.columns(3)
    u1.metric("熱門股站上 MA20", f"{us_pct:.1f}%")
    u2.metric("有效熱門股樣本", us_scan["total"])
    u3.metric("資料更新時間", us_scan["time"])

    tab1, tab2 = st.tabs(["🔥 熱門強勢股", "🔻 熱門股拉回"])
    with tab1:
        if us_scan["strong"]:
            df_us_strong = pd.DataFrame(us_scan["strong"]).copy()
            df_us_strong = df_us_strong.rename(columns={
                "ticker": "代號", "name": "名稱", "price": "現價", "pct_ch": "漲跌%",
                "score": "評分", "rsi": "RSI", "vol_ratio": "量比"
            })
            show_cols = [c for c in ["代號", "名稱", "現價", "漲跌%", "評分", "RSI", "量比"] if c in df_us_strong.columns]
            st.dataframe(df_us_strong[show_cols], use_container_width=True, hide_index=True)
        else:
            st.info("目前熱門股池中沒有符合條件的強勢股。")

    with tab2:
        if us_scan["pullback"]:
            df_us_pull = pd.DataFrame(us_scan["pullback"]).copy()
            df_us_pull = df_us_pull.rename(columns={
                "ticker": "代號", "name": "名稱", "price": "現價", "pct_ch": "漲跌%",
                "score": "評分", "rsi": "RSI", "vol_ratio": "量比"
            })
            show_cols = [c for c in ["代號", "名稱", "現價", "漲跌%", "評分", "RSI", "量比"] if c in df_us_pull.columns]
            st.dataframe(df_us_pull[show_cols], use_container_width=True, hide_index=True)
        else:
            st.info("目前熱門股池中沒有符合條件的拉回股。")

# -------------------------
# 評分規則
# -------------------------
elif page == "📖 評分規則":
    st.subheader("📖 多因子技術動能評分規則（-5 至 +5）")
    st.markdown(
        """
**1. 均線趨勢（MA5 / MA20 / MA60）**
- 均線多頭排列：+1.5
- 站上月線：+1.0
- 均線空頭排列：-1.5
- 跌破月線：-1.0

**2. RSI（14）**
- 55～75：+1.0
- 低於 30：+1.0
- 高於 80：-1.0
- 低於 45：-0.5

**3. KD（9）**
- 黃金交叉：+1.0
- K > D：+0.5
- 死亡交叉：-1.0
- K < D：-0.5

**4. MACD**
- DIF > MACD：+1.0
- DIF < MACD：-1.0

**5. 成交量與布林通道**
- 成交量 ≥ 20 日均量 1.5 倍且上漲：+1.0
- 觸及布林下軌：+0.5
- 突破布林上軌且乖離過大：-0.5
        """
    )

st.divider()
st.caption(
    "⚠️ 本網站與 AI 內容為量化研究、技術分析與課程展示用途；"
    "AI 可能出錯，請自行查證，不構成保證獲利或個人化投資建議。"
)
