import math
import re
import datetime
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

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

QUICK_POOL = [
    ("0050", "元大台灣50"), ("0056", "元大高股息"),
    ("2330", "台積電"), ("2317", "鴻海"), ("2454", "聯發科"), ("2308", "台達電"),
    ("2382", "廣達"), ("2357", "華碩"), ("2379", "瑞昱"), ("2327", "國巨"),
    ("2303", "聯電"), ("2408", "南亞科"), ("2344", "華邦電"), ("2337", "旺宏"),
    ("8299", "群聯"), ("3034", "聯詠"), ("3661", "世芯-KY"), ("3443", "創意"),
    ("3711", "日月光投控"), ("2449", "京元電子"), ("6239", "力成"), ("8150", "南茂"),
    ("3583", "辛耘"), ("6187", "萬潤"), ("3131", "弘塑"), ("6643", "M31"),
    ("3231", "緯創"), ("6669", "緯穎"), ("3017", "奇鋐"), ("3324", "雙鴻"),
    ("2345", "智邦"), ("2383", "台光電"), ("3037", "欣興"), ("8046", "南電"),
    ("2368", "金像電"), ("3013", "晟銘電"), ("2356", "英業達"), ("2376", "技嘉"),
    ("3081", "聯亞"), ("3363", "上詮"), ("4979", "華星光"), ("6442", "光聖"),
    ("3163", "波若威"), ("4908", "前鼎"),
    ("1519", "華城"), ("1513", "中興電"), ("1503", "士電"), ("1609", "大亞"),
    ("2492", "華新科"), ("3026", "禾伸堂"),
    ("2603", "長榮"), ("2609", "陽明"), ("2615", "萬海"), ("2618", "長榮航"),
    ("2881", "富邦金"), ("2882", "國泰金"), ("2891", "中信金"), ("2886", "兆豐金"),
    ("3702", "大聯大"), ("3036", "文曄"), ("8112", "至上"), ("8069", "元太")
]

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
        rows = [(code, name, "") for code, name in QUICK_POOL]

    dedup = {code: (code, name, suffix) for code, name, suffix in rows}
    return sorted(dedup.values(), key=lambda x: x[0])


def market_lookup():
    return {code: (name, suffix) for code, name, suffix in get_full_market_pool()}


def resolve_stock_name(stock_id: str) -> str:
    lookup = market_lookup()
    if stock_id in lookup:
        return lookup[stock_id][0]
    for code, name in QUICK_POOL:
        if code == stock_id:
            return name
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
    """讀取目前訪客這個 Streamlit session 的自選股。"""
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []
    return list(st.session_state["favorites"])


def save_favorites(items: list[str]) -> None:
    """只儲存在目前訪客的 session，不寫入共用 JSON。"""
    cleaned = []
    for x in items:
        x = str(x).strip()
        if re.fullmatch(r"\d{4,6}", x) and x not in cleaned:
            cleaned.append(x)
    st.session_state["favorites"] = cleaned


def add_favorite(stock_id: str) -> bool:
    stock_id = str(stock_id).strip()
    if not re.fullmatch(r"\d{4,6}", stock_id):
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
st.set_page_config(
    page_title="I-Cheng Stock Bot",
    page_icon="📈",
    layout="wide",
)

if "stock_id_input" not in st.session_state:
    st.session_state["stock_id_input"] = "2330"
if "page_select" not in st.session_state:
    st.session_state["page_select"] = "個股分析"


def open_favorite(stock_id: str):
    st.session_state["stock_id_input"] = stock_id
    st.session_state["page_select"] = "個股分析"


# -------------------------
# 左側導覽 + 自選股
# -------------------------
st.sidebar.title("📈 I-Cheng Stock Bot")
page = st.sidebar.radio(
    "功能選單",
    ["個股分析", "掃描中心＋大盤廣度", "評分規則"],
    key="page_select",
)

st.sidebar.divider()
st.sidebar.subheader("⭐ 我的自選股")
sidebar_add = st.sidebar.text_input("加入股票代號", placeholder="例如 2330", key="sidebar_add_stock")
if st.sidebar.button("➕ 加入自選", use_container_width=True):
    sid = sidebar_add.strip()
    if add_favorite(sid):
        st.sidebar.success(f"已加入 {resolve_stock_name(sid)} ({sid})")
        st.rerun()
    else:
        st.sidebar.warning("請輸入正確股票代號。")

favorites = load_favorites()
if not favorites:
    st.sidebar.caption("目前尚無自選股。")
else:
    for sid in favorites:
        name = resolve_stock_name(sid)
        c1, c2 = st.sidebar.columns([4, 1])
        c1.button(
            f"{name} ({sid})",
            key=f"fav_open_{sid}",
            use_container_width=True,
            on_click=open_favorite,
            args=(sid,),
        )
        if c2.button("✕", key=f"fav_del_{sid}", help=f"移除 {name}"):
            remove_favorite(sid)
            st.rerun()

st.sidebar.caption("自選股只屬於目前訪客的使用階段，不與其他使用者共用。")

# -------------------------
# 頁首
# -------------------------
st.title("📈 I-Cheng Stock Bot")
st.caption("台股量化技術分析｜MA / RSI / KD / MACD / 布林通道 / 成交量｜全市場兩階段掃描")

if page == "個股分析":
    col1, col2 = st.columns([1, 3])
    with col1:
        stock_id = st.text_input("股票代號", key="stock_id_input", max_chars=6).strip()
        interval_label = st.selectbox("K線週期", ["日K", "週K", "60分", "15分", "5分"])
        interval_map = {"日K": "1d", "週K": "1wk", "60分": "60m", "15分": "15m", "5分": "5m"}
        period_map = {"1d": "6mo", "1wk": "2y", "60m": "3mo", "15m": "1mo", "5m": "1mo"}
        do_search = st.button("🔍 開始分析", use_container_width=True, type="primary")
        if st.button("⭐ 加入目前股票到自選", use_container_width=True):
            if add_favorite(stock_id):
                st.success("已加入自選股")
                st.rerun()

    if do_search or stock_id:
        interval = interval_map[interval_label]
        with st.spinner("抓取股價與計算技術指標中..."):
            df = fetch_stock_df(stock_id, interval, period_map[interval])

        if df.empty:
            st.error("查無股價資料，請確認股票代號。")
        else:
            ind = calculate_indicators(df)
            ev = evaluate_score(ind)
            strategy = build_trading_strategy(ind, ev)
            pct_ch = (ind["c_now"] - ind["c_prev"]) / max(ind["c_prev"], 1e-10) * 100
            name = resolve_stock_name(stock_id)

            st.subheader(f"{name} ({stock_id})")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("最新價格", f"{ind['c_now']:.2f}", f"{pct_ch:+.2f}%")
            m2.metric("綜合評分", f"{ev['score']:+.1f} / 5")
            m3.metric("RSI", f"{ind['rsi']:.1f}")
            m4.metric("量比", f"{ev['vol_ratio']:.2f}x")
            m5.metric("多頭強度", f"{ev['bull_pct']:.0f}%")

            c1, c2 = st.columns([2.2, 1])
            with c1:
                st.plotly_chart(price_chart(df, ind), use_container_width=True)
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
                        ind["rsi"], ind["k_now"], ind["d_now"], ind["dif_now"], ind["macd_now"], ind["atr"]
                    ]
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

elif page == "掃描中心＋大盤廣度":
    st.subheader("📊 全市場掃描中心＋大盤廣度")
    st.markdown(
        """
這裡把 **全市場選股** 與 **大盤廣度** 整合成同一次資料更新：

- 第一階段掃描上市／上櫃股票，依趨勢、5/20 日漲幅、量能與流動性排序，保留前 50 檔候選。
- 第二階段對候選股套用完整 MA / RSI / KD / MACD / 布林 / 成交量評分。
- **大盤廣度** 同時統計全市場「股價站上 20 日均線（MA20）」的比例，不另外重掃一次。
- 廣度越高，代表上漲有較多股票共同參與；廣度偏低，代表市場可能只有少數股票撐盤。
- 資料快取有效 **30 分鐘**；超過 30 分鐘後，再次進入本頁會取得新一批資料。也可隨時手動重新掃描。
        """
    )

    force_scan = st.button("🚀 立即全市場重新掃描", type="primary", use_container_width=False)
    if force_scan:
        first_stage_market_scan.clear()
        run_full_market_scan.clear()
        st.session_state.pop("market_scan_result", None)

    result = st.session_state.get("market_scan_result")
    stale = True
    if result and result.get("updated"):
        try:
            dt = datetime.datetime.strptime(result["updated"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=TW_TZ)
            stale = (datetime.datetime.now(TW_TZ) - dt).total_seconds() >= SCAN_CACHE_MINUTES * 60
        except Exception:
            stale = True

    if force_scan or result is None or stale:
        with st.spinner("正在掃描上市／上櫃市場並同步計算大盤廣度..."):
            result = run_full_market_scan()
        st.session_state["market_scan_result"] = result

    if result:
        pct = result["above_ma20"] / max(result["valid_total"], 1) * 100
        status = "🟢 多頭健康" if pct >= 50 else "🟡 震盪整理" if pct >= 30 else "🔴 空頭偏弱"

        st.caption(f"🕒 全市場資料更新時間：{result['updated']}（同一批資料供選股與大盤廣度使用）")

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("大盤廣度｜站上 MA20", f"{pct:.1f}%")
        a2.metric("站上 MA20 檔數", result["above_ma20"])
        a3.metric("有效市場樣本", result["valid_total"])
        a4.metric("市場狀態", status)
        st.progress(min(max(pct / 100, 0.0), 1.0))

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("第一階段候選", result["candidates"])
        c2.metric("🔥 強勢多頭股", len(result["strong"]))
        c3.metric("🔻 多頭回調股", len(result["pullback"]))

        tab1, tab2 = st.tabs(["🔥 強勢多頭", "🔻 多頭回調"])
        with tab1:
            st.caption("完整評分達強勢門檻、且股價站上 MA20 的候選股。")
            if result["strong"].empty:
                st.info("目前沒有符合條件的強勢股。")
            else:
                st.dataframe(result["strong"], use_container_width=True, hide_index=True)
        with tab2:
            st.caption("中期仍在多頭架構，但當日回檔、評分尚未達強勢門檻的候選股。")
            if result["pullback"].empty:
                st.info("目前沒有符合條件的回調股。")
            else:
                st.dataframe(result["pullback"], use_container_width=True, hide_index=True)

elif page == "評分規則":
    st.subheader("📖 多因子技術動能評分規則（-5 至 +5）")
    st.markdown("""
**1. 均線趨勢**
- 均線多頭排列：+1.5
- 站上 MA20：+1.0
- 均線空頭排列：-1.5
- 跌破 MA20：-1.0

**2. RSI**
- 55～75：+1.0
- < 30：+1.0
- > 80：-1.0
- < 45：-0.5

**3. KD**
- 黃金交叉：+1.0
- K > D：+0.5
- 死亡交叉：-1.0
- K < D：-0.5

**4. MACD**
- DIF > MACD：+1.0
- DIF < MACD：-1.0

**5. 成交量與布林**
- 量達 20 日均量 1.5 倍且上漲：+1.0
- 跌破布林下軌：+0.5
- 突破布林上軌：-0.5
""")

st.divider()
st.caption("⚠️ 本網站為技術分析與課程展示用途，不構成投資建議。")
