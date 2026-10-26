"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Taiwan Stock Quantitative Analysis Telegram Bot (Clean & Interactive Edition)                ║
║   Core indicators: Moving Averages (MA) / Bollinger Bands / RSI / KD / MACD / Volume                   ║
║   Scoring system: Multi-factor momentum model (-5 to +5) + SQLite local cache               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import datetime
import json
import logging
import math
import os
import re
import sqlite3
import time
import warnings
from datetime import timedelta, timezone
from typing import Optional

import nest_asyncio
import numpy as np
import pandas as pd
import yfinance as yf
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

nest_asyncio.apply()

# Suppress excessive yfinance inplace deprecation warnings under newer pandas versions
warnings.filterwarnings(
    "ignore",
    message=r".*inplace keyword.*deprecated.*",
)

# Reduce console noise from invalid, delisted, or unsupported Yahoo Finance symbols
logging.getLogger("yfinance").setLevel(logging.CRITICAL)



# ═════════════════════════════════════════════════════════════════════════════
#  Basic Settings and Constants
# ═════════════════════════════════════════════════════════════════════════════

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TW_TZ = timezone(timedelta(hours=8))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "scan_cache.db")
FAV_FILE = os.path.join(BASE_DIR, "favorites.json")

# Quick-access stock list: used only for menu shortcuts and name fallback; not used in the full-market strength scan.
# This list can be customized based on frequently viewed stocks.
QUICK_POOL = [
    # ETFs / Large-Cap Stocks
    ("0050", "元大台灣50"), ("0056", "元大高股息"),
    ("2330", "台積電"), ("2317", "鴻海"), ("2454", "聯發科"), ("2308", "台達電"),
    ("2382", "廣達"), ("2357", "華碩"), ("2379", "瑞昱"), ("2327", "國巨"),

    # Semiconductors / Memory / IC
    ("2303", "聯電"), ("2408", "南亞科"), ("2344", "華邦電"), ("2337", "旺宏"),
    ("8299", "群聯"), ("3034", "聯詠"), ("3661", "世芯-KY"), ("3443", "創意"),
    ("3711", "日月光投控"), ("2449", "京元電子"), ("6239", "力成"), ("8150", "南茂"),
    ("3583", "辛耘"), ("6187", "萬潤"), ("3131", "弘塑"), ("6643", "M31"),

    # AI / Servers / Thermal / PCB
    ("3231", "緯創"), ("6669", "緯穎"), ("3017", "奇鋐"), ("3324", "雙鴻"),
    ("2345", "智邦"), ("2383", "台光電"), ("3037", "欣興"), ("8046", "南電"),
    ("2368", "金像電"), ("3013", "晟銘電"), ("2356", "英業達"), ("2376", "技嘉"),

    # Networking / Optical Communications
    ("3081", "聯亞"), ("3363", "上詮"), ("4979", "華星光"), ("6442", "光聖"),
    ("3163", "波若威"), ("4908", "前鼎"),

    # Power / Heavy Electrical / Passive Components
    ("1519", "華城"), ("1513", "中興電"), ("1503", "士電"), ("1609", "大亞"),
    ("2492", "華新科"), ("3026", "禾伸堂"),

    # Shipping
    ("2603", "長榮"), ("2609", "陽明"), ("2615", "萬海"), ("2618", "長榮航"),

    # Financials
    ("2881", "富邦金"), ("2882", "國泰金"), ("2891", "中信金"), ("2886", "兆豐金"),

    # Other Popular Electronics / Distribution
    ("3702", "大聯大"), ("3036", "文曄"), ("8112", "至上"), ("8069", "元太")
]

AUTO_SCAN_TASK = None
_report_cache: dict = {}
global_market_status = {"above_ma20": 0, "total": 0, "time": "Not updated yet"}

# Full-Market Scan Settings
MARKET_POOL_CACHE: list[tuple[str, str, str]] = []   # (ticker, name, Yahoo suffix)
MARKET_POOL_CACHE_DATE = ""
MARKET_LOOKUP: dict[str, tuple[str, str]] = {}  # code -> (name, suffix)
SCAN_LOCK = asyncio.Lock()                       # Prevent duplicate scans from manual and scheduled tasks
FIRST_STAGE_KEEP = 50          # Number of candidates retained after stage-one screening
BATCH_SIZE = 80               # Yahoo Finance batch size to reduce request count
STRONG_BULL_SCORE = 2.0
PULLBACK_MIN_SCORE = 0.0

MIN_AVG_TURNOVER = 8_000_000  # Require at least NT$8 million in 20-day average turnover to filter out illiquid stocks

# ═════════════════════════════════════════════════════════════════════════════
#  Utility Functions and SQLite Cache
# ═════════════════════════════════════════════════════════════════════════════

def esc(text) -> str:
    """Escape Telegram MarkdownV2 special characters"""
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

def create_progress_bar(pct: float, length: int = 10) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(pct / 100 * length)
    return "█" * filled + "░" * (length - filled)

def format_volume(vol: float) -> str:
    if vol >= 1_000_000:
        return f"{vol / 1_000_000:.1f}M"
    if vol >= 1_000:
        return f"{vol / 1_000:.1f}K"
    return str(int(vol))

def db_init():
    """Initialize the SQLite database used to cache scan results"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scan_results (
            ticker     TEXT NOT NULL,
            name       TEXT NOT NULL,
            score      REAL NOT NULL,
            signals    TEXT NOT NULL,
            entry_text TEXT NOT NULL,
            category   TEXT NOT NULL,
            timestamp  INTEGER NOT NULL
        )
    """)
    # Store the last scan timestamp for each category.
    # Keep the update timestamp accurate even if a scan returns zero stocks.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scan_metadata (
            category   TEXT PRIMARY KEY,
            timestamp  INTEGER NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS market_status (
            id          INTEGER PRIMARY KEY CHECK (id = 1),
            above_ma20  INTEGER NOT NULL,
            total       INTEGER NOT NULL,
            timestamp   INTEGER NOT NULL
        )
    """)
    con.commit()
    con.close()
    print("✅ SQLite local cache initialized")

def db_write_category(rows: list[dict], category: str):
    ts = int(time.time())
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM scan_results WHERE category = ?", (category,))
    cur.executemany(
        "INSERT INTO scan_results (ticker, name, score, signals, entry_text, category, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(r["ticker"], r["name"], r["score"], json.dumps(r["signals"], ensure_ascii=False), r["entry_text"], category, ts) for r in rows]
    )
    cur.execute(
        "INSERT INTO scan_metadata (category, timestamp) VALUES (?, ?) "
        "ON CONFLICT(category) DO UPDATE SET timestamp = excluded.timestamp",
        (category, ts)
    )
    con.commit()
    con.close()

def db_read_category(category: str) -> tuple[list[dict], int]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT ticker, name, score, signals, entry_text, timestamp FROM scan_results WHERE category = ? ORDER BY score DESC",
        (category,)
    )
    rows = cur.fetchall()
    cur.execute("SELECT timestamp FROM scan_metadata WHERE category = ?", (category,))
    meta = cur.fetchone()
    con.close()

    ts = int(meta[0]) if meta else (int(rows[0][5]) if rows else 0)
    result = [
        {"ticker": r[0], "name": r[1], "score": r[2], "signals": json.loads(r[3]), "entry_text": r[4]}
        for r in rows
    ]
    return result, ts


def format_scan_time(ts: int) -> str:
    """Convert a SQLite Unix timestamp to Taiwan local time."""
    if not ts:
        return "Not updated yet"
    return datetime.datetime.fromtimestamp(ts, TW_TZ).strftime("%Y-%m-%d %H:%M:%S")


def db_write_market_status(above_ma20: int, total: int, ts: int | None = None):
    """Store market breadth from the latest full-market scan."""
    ts = int(ts or time.time())
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO market_status (id, above_ma20, total, timestamp) VALUES (1, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET above_ma20=excluded.above_ma20, total=excluded.total, timestamp=excluded.timestamp",
        (int(above_ma20), int(total), ts)
    )
    con.commit()
    con.close()


def db_read_market_status() -> dict:
    """Read the latest saved market breadth without re-querying Yahoo."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT above_ma20, total, timestamp FROM market_status WHERE id = 1")
    row = cur.fetchone()
    con.close()
    if not row:
        return {"above_ma20": 0, "total": 0, "time": "Not updated yet", "timestamp": 0}
    ts = int(row[2])
    return {
        "above_ma20": int(row[0]),
        "total": int(row[1]),
        "time": format_scan_time(ts),
        "timestamp": ts,
    }

def load_favs() -> dict:
    if os.path.exists(FAV_FILE):
        try:
            with open(FAV_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_favs(favs: dict):
    with open(FAV_FILE, "w", encoding="utf-8") as f:
        json.dump(favs, f, ensure_ascii=False)

def get_full_market_pool(force: bool = False) -> list[tuple[str, str, str]]:
    """
    Build the universe of Taiwan-listed and OTC common stocks.
    Use twstock's built-in symbol table first, avoiding one-by-one Yahoo lookups; rebuild once per day.
    Return format: (ticker, name, Yahoo suffix).
    If twstock is unavailable, fall back to QUICK_POOL.
    """
    global MARKET_POOL_CACHE, MARKET_POOL_CACHE_DATE
    today = datetime.datetime.now(TW_TZ).strftime("%Y-%m-%d")
    if MARKET_POOL_CACHE and MARKET_POOL_CACHE_DATE == today and not force:
        return MARKET_POOL_CACHE

    rows = []
    try:
        import twstock
        for code, info in twstock.codes.items():
            code = str(code).strip()
            if not re.fullmatch(r"\d{4}", code):
                continue

            market = str(getattr(info, "market", ""))
            kind = str(getattr(info, "type", getattr(info, "security_type", "")))
            name = str(getattr(info, "name", code))

            # Keep only listed/OTC common stocks and exclude ETFs, warrants, preferred shares, and other non-common-stock instruments.
            if market not in ("上市", "上櫃"):
                continue
            if kind and "股票" not in kind:
                continue

            suffix = ".TW" if market == "上市" else ".TWO"
            rows.append((code, name, suffix))
    except Exception as e:
        print(f"⚠️ Unable to build the full-market universe; falling back to QUICK_POOL: {e}")

    if not rows:
        # Fallback only when twstock is unavailable; fetch_stock_df will still try both .TW and .TWO.
        rows = [(code, name, "") for code, name in QUICK_POOL]

    # Deduplicate and sort by ticker while building a fast ticker -> name / Yahoo suffix lookup.
    dedup = {}
    for code, name, suffix in rows:
        dedup[code] = (code, name, suffix)
    MARKET_POOL_CACHE = sorted(dedup.values(), key=lambda x: x[0])
    MARKET_POOL_CACHE_DATE = today
    MARKET_LOOKUP.clear()
    MARKET_LOOKUP.update({code: (name, suffix) for code, name, suffix in MARKET_POOL_CACHE})
    print(f"✅ Full-market stock universe built: {len(MARKET_POOL_CACHE)} stocks")
    return MARKET_POOL_CACHE


def _extract_symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Extract a single stock's OHLCV data from a multi-ticker yfinance response."""
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
        needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if "Close" not in needed:
            return pd.DataFrame()
        df = df[needed].dropna(subset=["Close"])
        return df
    except Exception:
        return pd.DataFrame()


async def first_stage_market_scan(limit: int = FIRST_STAGE_KEEP) -> tuple[list[dict], int, int]:
    """
    Stage 1: fast full-market batch screening.
    Rank stocks using price trend, 20-day return, relative volume, and liquidity.
    Return candidate list, number above MA20, and valid sample count.
    """
    pool = get_full_market_pool()
    candidates = []
    above_ma20 = 0
    valid_total = 0

    for start in range(0, len(pool), BATCH_SIZE):
        chunk = pool[start:start + BATCH_SIZE]
        symbols = [code + suffix for code, _, suffix in chunk]
        try:
            raw = await asyncio.to_thread(
                yf.download,
                tickers=" ".join(symbols),
                period="3mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"⚠️ Full-market batch {start // BATCH_SIZE + 1} download failed: {e}")
            continue

        for code, name, suffix in chunk:
            symbol = code + suffix
            df = _extract_symbol_frame(raw, symbol)
            if df.empty or len(df) < 25:
                continue
            try:
                close = df["Close"].astype(float)
                volume = df["Volume"].fillna(0).astype(float) if "Volume" in df.columns else pd.Series(0, index=df.index)
                c = float(close.iloc[-1])
                ma5 = float(close.tail(5).mean())
                ma20 = float(close.tail(20).mean())
                ret5 = (c / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 else 0.0
                ret20 = (c / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 else 0.0
                vol20 = float(volume.tail(20).mean())
                vol_ratio = float(volume.iloc[-1] / vol20) if vol20 > 0 else 0.0
                avg_turnover = float((close.tail(20) * volume.tail(20)).mean())

                if not all(map(math.isfinite, [c, ma5, ma20, ret5, ret20, vol_ratio, avg_turnover])):
                    continue
                valid_total += 1
                if c > ma20:
                    above_ma20 += 1

                # Exclude illiquid stocks from stage two to reduce noisy signals and trading frictions.
                if avg_turnover < MIN_AVG_TURNOVER:
                    continue

                # Stage one is not the final trading score; it only forwards recently strengthening stocks to stage two.
                fast_score = 0.0
                fast_score += 2.0 if c > ma20 else -1.0
                fast_score += 1.0 if ma5 > ma20 else 0.0
                fast_score += max(-1.0, min(2.0, ret20 / 8.0))
                fast_score += max(-0.5, min(1.5, ret5 / 5.0))
                fast_score += max(0.0, min(1.5, (vol_ratio - 1.0)))

                candidates.append({
                    "ticker": code, "name": name, "suffix": suffix,
                    "fast_score": fast_score, "ret5": ret5, "ret20": ret20,
                    "vol_ratio": vol_ratio, "price": c,
                })
            except Exception:
                continue

        # Yield briefly to the event loop so the Telegram bot remains responsive during scanning.
        await asyncio.sleep(0.05)

    candidates.sort(key=lambda x: (x["fast_score"], x["ret20"], x["vol_ratio"]), reverse=True)
    return candidates[:limit], above_ma20, valid_total

# ═════════════════════════════════════════════════════════════════════════════
#  Ticker Resolution and Market Data Retrieval
# ═════════════════════════════════════════════════════════════════════════════

def clean_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Yahoo Finance data; symbols without valid OHLC data are treated as unavailable."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    needed = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if not needed or "Close" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["Close"])
    if "High" in df.columns:
        df = df[df["High"].notna()]
    if "Low" in df.columns:
        df = df[df["Low"].notna()]
    return df


def resolve_stock_name(stock_id: str) -> str:
    """Use the full-market lookup cache first to avoid repeated twstock queries."""
    if stock_id in MARKET_LOOKUP:
        return MARKET_LOOKUP[stock_id][0]
    try:
        import twstock
        if stock_id in twstock.codes:
            return twstock.codes[stock_id].name
    except Exception:
        pass
    for code, name in QUICK_POOL:
        if code == stock_id:
            return name
    return stock_id


def get_stock_suffix_candidates(stock_id: str) -> list[str]:
    """Use the correct Yahoo suffix when market type is known; otherwise try both .TW and .TWO."""
    if stock_id in MARKET_LOOKUP:
        suffix = MARKET_LOOKUP[stock_id][1]
        if suffix in (".TW", ".TWO"):
            return [suffix]
    return [".TW", ".TWO"]


async def fetch_stock_df(stock_id: str, interval: str = "1d", period: str = "6mo") -> pd.DataFrame:
    for suffix in get_stock_suffix_candidates(stock_id):
        symbol = f"{stock_id}{suffix}"
        try:
            df = await asyncio.to_thread(
                yf.download,
                symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            required_cols = [c for c in ["Close", "High", "Low"] if c in df.columns]
            if len(required_cols) < 3:
                continue
            df = df.dropna(subset=required_cols).copy()
            if "Volume" not in df.columns:
                df["Volume"] = 0.0
            else:
                df["Volume"] = df["Volume"].fillna(0)

            if len(df) >= 25:
                return df
        except Exception as e:
            print(f"⚠️ {symbol} data download failed: {e}")
    return pd.DataFrame()


async def fetch_candidate_frames(candidates: list[dict], period: str = "6mo") -> dict[str, pd.DataFrame]:
    """Batch-download stage-two candidates to avoid repeated Yahoo requests for the top 50 stocks."""
    result: dict[str, pd.DataFrame] = {}
    if not candidates:
        return result

    for start in range(0, len(candidates), BATCH_SIZE):
        chunk = candidates[start:start + BATCH_SIZE]
        symbols = [c["ticker"] + c["suffix"] for c in chunk if c.get("suffix") in (".TW", ".TWO")]
        if not symbols:
            continue
        try:
            raw = await asyncio.to_thread(
                yf.download,
                tickers=" ".join(symbols),
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"⚠️ Stage-two batch download failed: {e}")
            continue

        for cand in chunk:
            suffix = cand.get("suffix", "")
            if suffix not in (".TW", ".TWO"):
                continue
            symbol = cand["ticker"] + suffix
            df = _extract_symbol_frame(raw, symbol)
            if not df.empty and len(df) >= 25:
                result[cand["ticker"]] = df
        await asyncio.sleep(0)

    return result

# ═════════════════════════════════════════════════════════════════════════════
#  Technical Indicator Calculation (MA / Bollinger Bands / RSI / KD / MACD / ATR / Volume)
# ═════════════════════════════════════════════════════════════════════════════

def calculate_indicators(df: pd.DataFrame) -> dict:
    # Safety check: only use bars with valid OHLC data
    df = df.dropna(subset=["Close", "High", "Low"]).copy()
    if len(df) < 2:
        raise ValueError("Insufficient valid price data to calculate technical indicators")

    close = df["Close"].squeeze().astype(float)
    high = df["High"].squeeze().astype(float)
    low = df["Low"].squeeze().astype(float)
    volume = df["Volume"].squeeze().fillna(0).astype(float)

    # 1. Moving averages (MA5, MA20, MA60)
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(min(60, len(close))).mean()

    # 2. Bollinger Bands (20 periods, 2 standard deviations)
    std20 = close.rolling(20).std()
    bb_up = ma20 + 2 * std20
    bb_low = ma20 - 2 * std20

    # 3. RSI (14 periods)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # 4. KD stochastic indicator (9 periods)
    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9 + 1e-10) * 100.0
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()

    # 5. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    macd = dif.ewm(span=9, adjust=False).mean()
    osc = dif - macd

    # 6. ATR volatility (14 periods, used for stop-loss reference)
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # 7. Average volume (20 periods)
    vol_ma20 = volume.rolling(20).mean()

    return {
        "c_now": float(close.iloc[-1]),
        "c_prev": float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1]),
        "ma5": float(ma5.iloc[-1]),
        "ma20": float(ma20.iloc[-1]),
        "ma60": float(ma60.iloc[-1]),
        "bb_up": float(bb_up.iloc[-1]),
        "bb_low": float(bb_low.iloc[-1]),
        "rsi": float(rsi.iloc[-1]),
        "k_now": float(k.iloc[-1]),
        "d_now": float(d.iloc[-1]),
        "k_prev": float(k.iloc[-2]) if len(k) > 1 else float(k.iloc[-1]),
        "d_prev": float(d.iloc[-2]) if len(d) > 1 else float(d.iloc[-1]),
        "dif_now": float(dif.iloc[-1]),
        "macd_now": float(macd.iloc[-1]),
        "osc_now": float(osc.iloc[-1]),
        "atr": float(atr.iloc[-1]) if not math.isnan(float(atr.iloc[-1])) else float(close.iloc[-1]) * 0.02,
        "vol_now": float(volume.iloc[-1]),
        "vol_ma20": float(vol_ma20.iloc[-1]) if float(vol_ma20.iloc[-1]) > 0 else 1.0,
    }

# ═════════════════════════════════════════════════════════════════════════════
#  Multi-Factor Momentum Scoring System (-5 to +5)
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_score(ind: dict) -> dict:
    """
    Score stocks from -5 to +5 using moving averages, RSI, KD, MACD, Bollinger Bands, and volume.
    """
    score = 0.0
    bull_signals = []
    bear_signals = []

    c = ind["c_now"]
    ma5, ma20, ma60 = ind["ma5"], ind["ma20"], ind["ma60"]

    # 1. Moving-average alignment and MA20 position (weight ±1.5)
    if c > ma5 > ma20 > ma60:
        score += 1.5
        bull_signals.append("Bullish moving-average alignment (+1.5)")
    elif c > ma20:
        score += 1.0
        bull_signals.append("Price above MA20 (+1.0)")
    elif c < ma5 < ma20 < ma60:
        score -= 1.5
        bear_signals.append("Bearish moving-average alignment (-1.5)")
    else:
        score -= 1.0
        bear_signals.append("Price below MA20 (-1.0)")

    # 2. RSI strength and overbought/oversold conditions (weight ±1.0)
    rsi = ind["rsi"]
    if 55 <= rsi <= 75:
        score += 1.0
        bull_signals.append("RSI in strong bullish zone (+1.0)")
    elif rsi < 30:
        score += 1.0
        bull_signals.append("RSI oversold rebound signal (+1.0)")
    elif rsi > 80:
        score -= 1.0
        bear_signals.append("RSI overbought / overheated (-1.0)")
    elif rsi < 45:
        score -= 0.5
        bear_signals.append("RSI momentum weak (-0.5)")

    # 3. KD golden/death cross and bullish/bearish structure (weight ±1.0)
    k_n, d_n = ind["k_now"], ind["d_now"]
    k_p, d_p = ind["k_prev"], ind["d_prev"]
    if k_p < d_p and k_n > d_n:
        score += 1.0
        bull_signals.append("KD golden cross (+1.0)")
    elif k_n > d_n:
        score += 0.5
        bull_signals.append("KD bullish structure (+0.5)")
    elif k_p > d_p and k_n < d_n:
        score -= 1.0
        bear_signals.append("KD death cross (-1.0)")
    else:
        score -= 0.5
        bear_signals.append("KD bearish structure (-0.5)")

    # 4. MACD trend momentum (weight ±1.0)
    if ind["dif_now"] > ind["macd_now"]:
        score += 1.0
        bull_signals.append("MACD bullish momentum (+1.0)")
    else:
        score -= 1.0
        bear_signals.append("MACD bearish momentum (-1.0)")

    # 5. Volume and Bollinger Band breakout conditions (weight +1.0 / -0.5)
    vol_ratio = ind["vol_now"] / ind["vol_ma20"]
    if vol_ratio >= 1.5 and c > ind["c_prev"]:
        score += 1.0
        bull_signals.append(f"High-volume advance  {vol_ratio:.1f}× average volume (+1.0)")
    elif c < ind["bb_low"]:
        score += 0.5
        bull_signals.append("Price near lower Bollinger Band support (+0.5)")
    elif c > ind["bb_up"]:
        score -= 0.5
        bear_signals.append("Price stretched above upper Bollinger Band (-0.5)")

    final_score = max(-5.0, min(5.0, round(score, 1)))
    # Convert the -5 to +5 score into a 0% to 100% bullish-strength indicator
    bull_pct = (final_score + 5.0) / 10.0 * 100.0

    return {
        "score": final_score,
        "bull_pct": bull_pct,
        "bull_signals": bull_signals,
        "bear_signals": bear_signals,
        "vol_ratio": vol_ratio
    }

# ═════════════════════════════════════════════════════════════════════════════
#  Individual Stock Strategy Generation
# ═════════════════════════════════════════════════════════════════════════════

def build_trading_strategy(ind: dict, ev: dict) -> dict:
    """Convert the quantitative score into actionable guidance for new entries and existing positions."""
    c = ind["c_now"]
    ma5, ma20, ma60 = ind["ma5"], ind["ma20"], ind["ma60"]
    rsi, score, vol_ratio = ind["rsi"], ev["score"], ev["vol_ratio"]
    atr = max(ind["atr"], c * 0.01)
    stop = max(ma20, c - 2.0 * atr) if c > ma20 else c - 2.0 * atr
    target = max(ind["bb_up"], c * 1.06)
    buy_low, buy_high = min(ma5, ma20), max(ma5, ma20)

    if score >= 3.0 and c > ma20:
        stance = "🟢 Bullish / Hold"
        new_entry = f"Avoid chasing; consider scaling in after a pullback to {buy_low:.2f}–{buy_high:.2f} if price stabilizes."
        holding = f"Hold existing positions; near {target:.2f} consider partial profit-taking; reduce or exit below {stop:.2f}."
    elif score >= 1.5 and c > ma20:
        stance = "🟡 Buy on Pullback"
        new_entry = f"Wait for a pullback toward {buy_low:.2f}–{buy_high:.2f} and consider scaling in after stabilization."
        holding = f"May continue holding without chasing; reduce if price falls below MA20 {ma20:.2f}, with {stop:.2f} as the defensive level."
    elif score >= 0.5:
        stance = "🟡 Small Position / Wait for Confirmation"
        new_entry = f"Start with a small position; add only after holding above MA20 {ma20:.2f} with improving volume."
        holding = f"Hold and monitor; reduce exposure if price cannot hold {ma20:.2f}."
    elif score <= -2.5:
        stance = "🔴 Bearish / Avoid"
        new_entry = f"Avoid new entries until price reclaims MA20 {ma20:.2f} and KD/MACD improve."
        holding = f"Use rebounds mainly to reduce exposure; exit below {stop:.2f}."
    elif score <= -0.5:
        stance = "🟠 Weak / Wait"
        new_entry = f"Wait for stabilization and a move back above MA20 {ma20:.2f} before reassessing."
        holding = f"Control position size; reduce below {stop:.2f} and become more constructive only after reclaiming MA20."
    else:
        stance = "⚪ Range-Bound"
        new_entry = f"Consider entries only near {buy_low:.2f}–{buy_high:.2f}; reassess after a breakout."
        holding = f"Hold without adding; exit below {stop:.2f}; near {target:.2f} consider partial profit-taking."

    warnings = []
    if rsi >= 75: warnings.append("RSI is elevated; avoid chasing")
    if vol_ratio >= 2.0: warnings.append("Volume is unusually high; watch for next-session volatility")
    if c < ma60: warnings.append("Price remains below MA60; medium-term trend is weak")
    return {"stance": stance, "new_entry": new_entry, "holding": holding,
            "warning": "; ".join(warnings) if warnings else "No additional technical warnings"}

# ═════════════════════════════════════════════════════════════════════════════
#  Full Stock Analysis Report (Summary / Technical / Risk tabs)
# ═════════════════════════════════════════════════════════════════════════════

async def analyze_stock_report(stock_id: str, interval: str = "1d", tab: str = "summary", force: bool = False) -> str:
    cache_key = f"{stock_id}:{interval}:{tab}"
    if not force and cache_key in _report_cache:
        ts, text = _report_cache[cache_key]
        if time.time() - ts < 180:
            return text

    period_map = {"5m": "1mo", "15m": "1mo", "60m": "3mo", "1d": "6mo", "1wk": "2y"}
    period = period_map.get(interval, "6mo")

    stock_name = resolve_stock_name(stock_id)
    df = await fetch_stock_df(stock_id, interval=interval, period=period)
    if df.empty:
        return f"⚠️ No price data found for `{esc(stock_id)}`. Please verify the ticker."

    ind = calculate_indicators(df)
    ev = evaluate_score(ind)

    c_now = ind["c_now"]
    pct_ch = (c_now - ind["c_prev"]) / (ind["c_prev"] + 1e-10) * 100.0
    score = ev["score"]
    bull_pct = ev["bull_pct"]
    strategy = build_trading_strategy(ind, ev)
    bar = create_progress_bar(bull_pct)
    now_str = datetime.datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")

    # Stop-loss and target calculation using MA20, upper Bollinger Band, and 2× ATR
    stop_loss = max(ind["ma20"], c_now - 2.0 * ind["atr"]) if c_now > ind["ma20"] else c_now - 2.0 * ind["atr"]
    target_price = max(ind["bb_up"], c_now * 1.06)

    c_str = f"{c_now:.2f}"
    pct_str = f"{pct_ch:+.2f}%"
    score_str = f"{score:+.1f}"
    bull_pct_str = f"{bull_pct:.0f}%"
    stop_str = f"{stop_loss:.2f}"
    target_str = f"{target_price:.2f}"
    ma5_str = f"{ind['ma5']:.2f}"
    ma20_str = f"{ind['ma20']:.2f}"
    ma60_str = f"{ind['ma60']:.2f}"

    if tab == "summary":
        if score >= 2.5:
            action = "🟢 Strong Bullish Momentum"
        elif score >= 0.5:
            action = "🟡 Moderately Bullish / Buy on Pullback"
        elif score <= -2.5:
            action = "🔴 Bearish / Reduce or Avoid"
        else:
            action = "⚪ Range-Bound / Wait for Direction"

        trend_str = (
            "📈 Bullish moving-average alignment" if c_now > ind["ma20"] > ind["ma60"]
            else "📉 Bearish moving-average alignment" if c_now < ind["ma20"] < ind["ma60"]
            else "↔️ Mixed moving averages / consolidation"
        )

        report = (
            f"📊 *[{esc(stock_name)} \\({esc(stock_id)}\\)] Summary Score Report — {esc(interval)}*\n"
            f"🎯 Latest Price: `{esc(c_str)}` \\(`{esc(pct_str)}`\\)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 *Overall View*: {esc(action)}\n"
            f"🧮 *Multi-Factor Momentum Score*: `{esc(score_str)}` / 5\\.0\n"
            f"📊 *Bullish Strength*: `{bar}` `{esc(bull_pct_str)}`\n"
            f"🌊 *Moving-Average Structure*: {esc(trend_str)}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📝 *Trading Strategy*\n"
            f"  • New Entry: {esc(strategy['new_entry'])}\n"
            f"  • Existing Position: {esc(strategy['holding'])}\n"
            f"  • Risk Note: {esc(strategy['warning'])}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 *Reference Levels*\n"
            f"  • Support Zone \\(MA5/MA20\\): `{esc(ma5_str)}` \\~ `{esc(ma20_str)}`\n"
            f"  • Upper Bollinger Target: `{esc(target_str)}`\n"
            f"  • Suggested Stop Reference: `{esc(stop_str)}`\n"
            f"\n⏱ `{esc(now_str)}`"
        )

    elif tab == "technical":
        rsi_str = f"{ind['rsi']:.1f}"
        k_str = f"{ind['k_now']:.1f}"
        d_str = f"{ind['d_now']:.1f}"
        dif_str = f"{ind['dif_now']:.2f}"
        macd_str = f"{ind['macd_now']:.2f}"
        vol_r_str = f"{ev['vol_ratio']:.2f}x"

        bull_txt = "、".join(ev["bull_signals"]) if ev["bull_signals"] else "No clear bullish signals"
        bear_txt = "、".join(ev["bear_signals"]) if ev["bear_signals"] else "No clear bearish signals"

        report = (
            f"📐 *[{esc(stock_name)} \\({esc(stock_id)}\\)] Technical Indicator Details — {esc(interval)}*\n"
            f"🎯 Latest Price: `{esc(c_str)}` \\(`{esc(pct_str)}`\\)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 *Core Technical Indicators*\n"
            f"  • Moving averages: MA5 `{esc(ma5_str)}` \\| MA20 `{esc(ma20_str)}` \\| MA60 `{esc(ma60_str)}`\n"
            f"  • RSI \\(14\\): `{esc(rsi_str)}`\n"
            f"  • KD \\(9\\): K\\=`{esc(k_str)}` / D\\=`{esc(d_str)}`\n"
            f"  • MACD: DIF\\=`{esc(dif_str)}` / MACD\\=`{esc(macd_str)}`\n"
            f"  • Volume: Current `{esc(format_volume(ind['vol_now']))}` \\(Avg. `{esc(vol_r_str)}`\\)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔔 *Scoring Signals*\n"
            f"  🟢 Positive Signals: {esc(bull_txt)}\n"
            f"  🔴 Negative Signals: {esc(bear_txt)}\n"
            f"\n⏱ `{esc(now_str)}`"
        )

    else:  # tab == "risk"
        atr_str = f"{ind['atr']:.2f}"
        bb_up_str = f"{ind['bb_up']:.2f}"
        bb_low_str = f"{ind['bb_low']:.2f}"
        risk_pct = abs(c_now - stop_loss) / (c_now + 1e-10) * 100.0
        reward_pct = abs(target_price - c_now) / (c_now + 1e-10) * 100.0
        rr_ratio = reward_pct / max(risk_pct, 0.5)

        report = (
            f"🛡️ *[{esc(stock_name)} \\({esc(stock_id)}\\)] Stop-Loss and Risk Reference — {esc(interval)}*\n"
            f"🎯 Latest Price: `{esc(c_str)}` \\(`{esc(pct_str)}`\\)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⚙️ *Volatility and Bollinger Bands*\n"
            f"  • Upper Bollinger Band \\(Resistance\\): `{esc(bb_up_str)}`\n"
            f"  • Middle Bollinger Band \\(MA20\\): `{esc(ma20_str)}`\n"
            f"  • Lower Bollinger Band \\(Support\\): `{esc(bb_low_str)}`\n"
            f"  • ATR \\(14-period Average True Range\\): `{esc(atr_str)}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⚖️ *Risk / Reward Assessment*\n"
            f"  • Estimated Target: `{esc(target_str)}` \\(`+{esc(f'{reward_pct:.1f}%')}`\\)\n"
            f"  • Suggested Stop: `{esc(stop_str)}` \\(`-{esc(f'{risk_pct:.1f}%')}`\\)\n"
            f"  • Estimated Risk / Reward: `{esc(f'{rr_ratio:.1f}')} : 1`\n"
            f"\n⏱ `{esc(now_str)}`"
        )

    _report_cache[cache_key] = (time.time(), report)
    return report

# ═════════════════════════════════════════════════════════════════════════════
#  Batch Screening and SQLite Cache Update
# ═════════════════════════════════════════════════════════════════════════════

async def run_quick_scan() -> dict:
    """
    Two-stage full-market stock screening:
    1) Batch-screen listed and OTC stocks and retain the top FIRST_STAGE_KEEP candidates.
    2) Batch-download six months of data for candidates, run the full multi-factor model, and save results to SQLite.
    SCAN_LOCK prevents manual scans and the 30-minute scheduled scan from running simultaneously.
    """
    global global_market_status

    if SCAN_LOCK.locked():
        print("ℹ️ A scan is already running; this duplicate request was skipped.")
        rows, _ = db_read_category("explosive")
        pulls, _ = db_read_category("pullback")
        return {
            "explosive": len(rows), "pullback": len(pulls),
            "market_total": global_market_status.get("total", 0), "candidates": 0,
            "skipped": True,
        }

    async with SCAN_LOCK:
        bull_list = []
        pullback_list = []

        print("🔎 Stage 1: scanning Taiwan listed and OTC markets...")
        candidates, above_ma20_count, total_valid = await first_stage_market_scan(FIRST_STAGE_KEEP)
        print(f"✅ Stage 1 complete: {total_valid} valid stocks; retained {len(candidates)} candidates")

        # Batch-download six months of daily data for candidates to avoid 50 separate requests.
        frames = await fetch_candidate_frames(candidates, period="6mo")

        for cand in candidates:
            code, name = cand["ticker"], cand["name"]
            df = frames.get(code, pd.DataFrame())
            if df.empty:
                # Fall back to a single-symbol request only for rare batch-download misses.
                df = await fetch_stock_df(code, interval="1d", period="6mo")
            if df.empty:
                continue
            try:
                ind = calculate_indicators(df)
                ev = evaluate_score(ind)
            except Exception as e:
                print(f"⚠️ {code} indicator calculation failed: {e}")
                continue

            c_now = ind["c_now"]
            pct_ch = (c_now - ind["c_prev"]) / (ind["c_prev"] + 1e-10) * 100.0
            sigs_short = ev["bull_signals"][:2] if ev["bull_signals"] else ["Technical consolidation"]

            entry_text = (
                f"*{esc(name)} \\({esc(code)}\\)* Price:`{esc(f'{c_now:.2f}')}` \\(`{esc(f'{pct_ch:+.1f}%')}`\\)\n"
                f"   ├ 🧮 Full score: `{esc(f'{ev['score']:+.1f}')}` / 5\\.0 | RSI: `{esc(f'{ind['rsi']:.0f}')}`\n"
                f"   ├ ⚡ Stage-one rank: `{esc(f'{cand['fast_score']:.1f}')}` | 20-day: `{esc(f'{cand['ret20']:+.1f}%')}`\n"
                f"   └ 🔔 Signals: {esc(', '.join(sigs_short))}"
            )

            item = {
                "ticker": code, "name": name, "score": ev["score"],
                "signals": ev["bull_signals"], "entry_text": entry_text,
            }

            if ev["score"] >= 1.5 and c_now > ind["ma20"]:
                bull_list.append(item)
            elif PULLBACK_MIN_SCORE <= ev["score"] < STRONG_BULL_SCORE and c_now > ind["ma60"] and pct_ch < 0:
                pullback_list.append(item)

        # Use stage-one ranking as a secondary tie-breaker for more stable ordering.
        bull_list.sort(key=lambda x: x["score"], reverse=True)
        pullback_list.sort(key=lambda x: x["score"], reverse=True)

        db_write_category(bull_list, "explosive")
        db_write_category(pullback_list, "pullback")

        if total_valid > 0:
            scan_ts = int(time.time())
            global_market_status = {
                "above_ma20": above_ma20_count,
                "total": total_valid,
                "time": format_scan_time(scan_ts),
            }
            # The full-market scan already counts stocks above MA20, so breadth is saved directly without another /market scan.
            db_write_market_status(above_ma20_count, total_valid, scan_ts)

        return {
            "explosive": len(bull_list),
            "pullback": len(pullback_list),
            "market_total": total_valid,
            "candidates": len(candidates),
            "skipped": False,
        }

# ═════════════════════════════════════════════════════════════════════════════
#  Intraday Auto Update: Scan Every 30 Minutes
# ═════════════════════════════════════════════════════════════════════════════

AUTO_SCAN_MINUTES = 30


def is_tw_market_scan_time(now: datetime.datetime | None = None) -> bool:
    """Run automatic scans only during Taiwan market hours on weekdays to avoid unnecessary Yahoo Finance requests."""
    now = now or datetime.datetime.now(TW_TZ)
    if now.weekday() >= 5:  # Saturday and Sunday
        return False
    hhmm = now.hour * 60 + now.minute
    return 9 * 60 <= hhmm <= 13 * 60 + 35


async def auto_scan_loop():
    """Automatically update every 30 minutes during market hours; errors are handled inside the loop so the background task remains alive."""
    while True:
        try:
            now = datetime.datetime.now(TW_TZ)

            # Run only on weekdays during Taiwan market hours
            is_weekday = now.weekday() < 5
            minutes = now.hour * 60 + now.minute
            in_session = (9 * 60) <= minutes <= (13 * 60 + 35)

            if is_weekday and in_session:
                # Trigger around the hour and half-hour
                if now.minute in (0, 30):
                    try:
                        print(f"🔄 Automatic scan started: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                        await run_quick_scan()
                        print("✅ Automatic scan completed")
                    except Exception as e:
                        print(f"⚠️ Automatic scan failed: {e}")

                    # Avoid duplicate triggers within the same minute
                    await asyncio.sleep(61)
                    continue

            # Normal polling interval
            await asyncio.sleep(20)

        except asyncio.CancelledError:
            print("ℹ️ Automatic scan task stopped")
            raise
        except Exception as e:
            # Keep the background task alive even if time checks or other logic fail
            print(f"⚠️ Background auto-scan loop error: {e}")
            await asyncio.sleep(30)


def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    """Main menu shortcut buttons; market breadth is integrated into the scan center."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📈 TSMC (2330)", callback_data="tab:2330:1d:summary"),
            InlineKeyboardButton("📱 MediaTek (2454)", callback_data="tab:2454:1d:summary"),
            InlineKeyboardButton("⚡ Hon Hai (2317)", callback_data="tab:2317:1d:summary")
        ],
        [
            InlineKeyboardButton("📊 Scan Center + Market Breadth", callback_data="buy_menu")
        ],
        [
            InlineKeyboardButton("🔥 Strong Stocks", callback_data="buy:cache:explosive"),
            InlineKeyboardButton("🔻 Bullish Pullbacks", callback_data="buy:cache:pullback")
        ],
        [
            InlineKeyboardButton("❤️ My Watchlist (/fav)", callback_data="open_fav"),
            InlineKeyboardButton("📖 Scoring Rules (/point)", callback_data="open_point")
        ]
    ])

def get_back_main_keyboard() -> InlineKeyboardMarkup:
    """Shared navigation: return to main menu"""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_main")]
    ])


def get_back_scan_keyboard() -> InlineKeyboardMarkup:
    """Shared navigation: return to scan menu / main menu"""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 Back to Scan Menu", callback_data="buy_menu")],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")]
    ])


def get_check_keyboard(stock_id: str, interval: str = "1d", tab: str = "summary") -> InlineKeyboardMarkup:
    """Tabs, interval selector, and watchlist controls shown below individual stock analysis"""
    tabs = [
        ("📊 Summary", "summary"),
        ("📐 Technical", "technical"),
        ("🛡️ Risk", "risk")
    ]
    row1 = [
        InlineKeyboardButton(
            f"{'● ' if tab == t else ''}{label}",
            callback_data=f"tab:{stock_id}:{interval}:{t}"
        )
        for label, t in tabs
    ]
    row2 = [
        InlineKeyboardButton(
            f"{'●' if interval == iv else ''}{lbl}",
            callback_data=f"tab:{stock_id}:{iv}:{tab}"
        )
        for lbl, iv in [("5m", "5m"), ("15m", "15m"), ("60m", "60m"), ("Daily", "1d"), ("Weekly", "1wk")]
    ]
    row3 = [
        InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh:{stock_id}:{interval}:{tab}"),
        InlineKeyboardButton("❤️ Add to Watchlist", callback_data=f"addfav:{stock_id}"),
        InlineKeyboardButton("❌ Close", callback_data="close_panel")
    ]
    row4 = [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_main")]
    return InlineKeyboardMarkup([row1, row2, row3, row4])

def get_scan_center_text() -> str:
    """Scan center combining full-market results, market breadth, update time, and usage notes."""
    bull_rows, ts_bull = db_read_category("explosive")
    pull_rows, ts_pull = db_read_category("pullback")
    latest_ts = max(ts_bull, ts_pull)
    updated_text = format_scan_time(latest_ts)
    market_cache = db_read_market_status()

    total = int(market_cache.get("total", 0) or 0)
    above = int(market_cache.get("above_ma20", 0) or 0)
    if total > 0:
        pct = above / total * 100.0
        breadth_bar = create_progress_bar(pct)
        breadth_status = (
            "🟢 Healthy Bullish Breadth" if pct >= 50
            else "🟡 Mixed / Consolidating" if pct >= 30
            else "🔴 Weak / Bearish Breadth"
        )
        breadth_text = (
            f"🌡️ Market Breadth: {pct:.1f}% ({above}/{total} stocks above MA20)\n"
            f"📊 Breadth: {breadth_bar}  {breadth_status}\n"
            f"🕒 Breadth Updated: {market_cache.get('time', updated_text)}\n"
        )
    else:
        breadth_text = (
            "🌡️ Market Breadth: no data yet\n"
            "📊 After one full-market scan, the system will calculate the percentage of stocks above MA20.\n"
        )

    return (
        "📊 Full-Market Quantitative Scan Center\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🕒 Last Scan Update: {updated_text}\n"
        f"🔥 Strong Bullish: {len(bull_rows)} stocks | 🔻 Bullish Pullbacks：{len(pull_rows)} stocks\n"
        + breadth_text +
        f"🔄 Auto update: every {AUTO_SCAN_MINUTES} minutes during Taiwan market hours (on the hour / half hour)\n"
        "🖐️ Manual update: use “Run Full-Market Scan Now” to refresh both stock selections and market breadth.\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🌡️ [How to Read Market Breadth]\n"
        "• Market breadth = the percentage of stocks trading above MA20.\n"
        "• A higher percentage means broader participation and generally healthier market structure.\n"
        "• A low percentage suggests weak participation and possible reliance on a small number of large-cap stocks.\n"
        "• Market breadth uses the same market-wide dataset as the strong-stock scan, avoiding duplicate requests.\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🌐 [Stage 1: Fast Full-Market Screening]\n"
        "• Scan Taiwan listed and OTC stocks and rank them by trend, 5/20-day returns, relative volume, and liquidity.\n"
        "• The same scan also counts stocks above MA20 to generate market breadth.\n"
        "• The top 50 candidates proceed to stage-two technical scoring.\n\n"
        "🎯 [Stage 2: Full Technical Scoring and Classification]\n"
        "🔥 Strong Bullish Stocks\n"
        "• The stock must meet the bullish score threshold and trade above MA20.\n"
        "• Used to identify stocks with stronger current trend and momentum.\n\n"
        "🔻 Bullish Pullback Stocks\n"
        "• The medium-term structure remains above MA60 while price is experiencing a short-term pullback.\n"
        "• Used to identify pullback opportunities within an ongoing uptrend.\n\n"
        "ℹ️ Not all top-50 candidates are classified; stocks that do not meet the rules are excluded.\n\n"
        "Choose a category below or manually refresh the full-market data:"
    )


def get_buy_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📈 View Strong Bullish Stocks", callback_data="buy:cache:explosive"),
            InlineKeyboardButton("🔻 View Bullish Pullbacks", callback_data="buy:cache:pullback")
        ],
        [
            InlineKeyboardButton("🔄 Run Full-Market Scan Now", callback_data="buy:rescan")
        ],
        [
            InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_main")
        ]
    ])


def get_market_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Recalculate Market Breadth", callback_data="market:rescan")],
        [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_main")],
    ])


def format_market_breadth_text(s: dict, markdown: bool = False) -> str:
    """Format cached market breadth for Telegram display."""
    if not s or s.get("total", 0) <= 0:
        return (
            "🌡️ Taiwan Market Breadth\n"
            "🕒 Updated：Not updated yet\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "No market-breadth cache is available yet.\n"
            "Run a full-market scan first or use “Recalculate Market Breadth” below."
        )
    pct = s["above_ma20"] / max(s["total"], 1) * 100.0
    bar = create_progress_bar(pct)
    status_lbl = "🟢 Healthy Bullish Breadth" if pct >= 50 else "🟡 Mixed / Consolidating" if pct >= 30 else "🔴 Weak / Bearish Breadth"
    if markdown:
        return (
            f"🌡️ *Taiwan Market Breadth*\n"
            f"🕒 Updated: `{esc(s['time'])}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 Stocks Above MA20: `{esc(f'{pct:.1f}%')}` \\({s['above_ma20']}/{s['total']}\\)\n"
            f"📊 Breadth: `{bar}`\n"
            f"💡 Market Status: *{esc(status_lbl)}*\n"
            f"ℹ️ This page reads the latest full-market scan cache and does not trigger a duplicate scan."
        )
    return (
        "🌡️ Taiwan Market Breadth\n"
        f"🕒 Updated：{s['time']}\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Stocks Above MA20: {pct:.1f}% ({s['above_ma20']}/{s['total']})\n"
        f"📊 Breadth: {bar}\n"
        f"💡 Market Status：{status_lbl}\n"
        "ℹ️ This page reads the latest full-market scan cache and does not trigger a duplicate scan."
    )


async def calculate_market_breadth() -> dict:
    """Calculate Taiwan market breadth in batches, skipping symbols unavailable from Yahoo."""
    global global_market_status

    try:
        market_pool = get_full_market_pool()
    except Exception as e:
        print(f"⚠️ Failed to build market universe: {e}")
        market_pool = []

    if not market_pool:
        global_market_status = {
            "above_ma20": 0,
            "total": 0,
            "time": datetime.datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        }
        return global_market_status

    symbols = []
    for code, _name, suffix in market_pool:
        suffix = suffix if suffix in (".TW", ".TWO") else ".TW"
        symbols.append(f"{code}{suffix}")

    total_valid = 0
    above_ma20 = 0
    batch_size = BATCH_SIZE

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            raw = await asyncio.to_thread(
                yf.download,
                tickers=" ".join(batch),
                period="2mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"⚠️ Market-breadth batch download failed: {e}")
            continue

        for sym in batch:
            try:
                sdf = _extract_symbol_frame(raw, sym)
                sdf = clean_price_frame(sdf)
                if len(sdf) < 20:
                    continue

                close = pd.to_numeric(sdf["Close"], errors="coerce").dropna()
                if len(close) < 20:
                    continue

                c_now = float(close.iloc[-1])
                ma20 = float(close.tail(20).mean())

                if not (math.isfinite(c_now) and math.isfinite(ma20)):
                    continue

                total_valid += 1
                if c_now > ma20:
                    above_ma20 += 1
            except Exception:
                continue

        await asyncio.sleep(0)

    manual_ts = int(time.time())
    global_market_status = {
        "above_ma20": above_ma20,
        "total": total_valid,
        "time": format_scan_time(manual_ts),
    }
    db_write_market_status(above_ma20, total_valid, manual_ts)
    return global_market_status


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 Taiwan Stock Quantitative Analysis Bot\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🔍 /check 2330 — Individual multi-factor technical analysis\n"
        "📊 /buy — Full-market scan center (breadth / strong stocks / pullbacks)\n"
        "❤️ /fav — View watchlist\n"
        "📖 /point — View multi-factor scoring rules\n\n"
        "👇 Use the shortcut buttons below to get started:"
    )
    await update.message.reply_text(
        text,
        reply_markup=get_main_menu_keyboard()
    )


async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("💡 Enter a stock ticker, for example: /check 2330")
        return

    stock_id = context.args[0].strip()
    wait_msg = await update.message.reply_text(
        f"⏳ Calculating {stock_id} technical indicators and score..."
    )

    report = await analyze_stock_report(
        stock_id, "1d", "summary", force=True
    )

    await wait_msg.edit_text(
        report,
        parse_mode="MarkdownV2",
        reply_markup=get_check_keyboard(stock_id, "1d", "summary")
    )


async def cmd_buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        get_scan_center_text(),
        reply_markup=get_buy_keyboard()
    )


async def cmd_market(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Backward-compatible command: market breadth is now integrated into the scan center and no longer triggers a separate scan."""
    await update.message.reply_text(
        "🌡️ Market breadth is integrated into the Full-Market Quantitative Scan Center and shares the same scan dataset.\n\n"
        + get_scan_center_text(),
        reply_markup=get_buy_keyboard()
    )


def get_point_text() -> str:
    """Use plain text for scoring rules to avoid Telegram MarkdownV2 escaping issues."""
    return (
        "📖 [Multi-Factor Technical Momentum Scoring Rules (-5 to +5)]\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "1️⃣ Moving-Average Trend (MA5 / MA20 / MA60)\n"
        "  • Bullish moving-average alignment: +1.5 | Above MA20: +1.0\n"
        "  • Bearish moving-average alignment: -1.5 | Below MA20: -1.0\n\n"
        "2️⃣ RSI Relative Strength Index (14)\n"
        "  • RSI 55–75 strong zone: +1.0\n"
        "  • RSI below 30 oversold rebound zone: +1.0\n"
        "  • RSI above 80 overbought: -1.0\n"
        "  • RSI below 45 weak momentum: -0.5\n\n"
        "3️⃣ KD Stochastic Indicator (9)\n"
        "  • KD golden cross：+1.0\n"
        "  • K > D bullish structure: +0.5\n"
        "  • KD death cross：-1.0\n"
        "  • K < D bearish structure: -0.5\n\n"
        "4️⃣ MACD Trend Momentum (12, 26, 9)\n"
        "  • DIF > MACD：+1.0\n"
        "  • DIF < MACD：-1.0\n\n"
        "5️⃣ Volume and Bollinger Bands\n"
        "  • Volume ≥ 1.5x the 20-day average with a positive session: +1.0\n"
        "  • Below lower Bollinger Band: +0.5 (oversold rebound watch)\n"
        "  • Above upper Bollinger Band: -0.5 (overextension risk)"
    )


async def cmd_point(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_point_text(), reply_markup=get_back_main_keyboard())


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/add 2330`", parse_mode="Markdown")
        return
    sid = context.args[0].strip()
    cid = str(update.effective_chat.id)
    favs = load_favs()
    ul = favs.get(cid, [])
    if sid not in ul:
        ul.append(sid)
        favs[cid] = ul
        save_favs(favs)
    name = resolve_stock_name(sid)
    await update.message.reply_text(f"✅ Added {name} ({sid}) to your watchlist. Use /fav to view it anytime.")

async def cmd_remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/remove 2330`", parse_mode="Markdown")
        return
    sid = context.args[0].strip()
    cid = str(update.effective_chat.id)
    favs = load_favs()
    ul = favs.get(cid, [])
    if sid in ul:
        ul.remove(sid)
        favs[cid] = ul
        save_favs(favs)
        await update.message.reply_text(f"🗑️ Added {sid} from the watchlist.")
    else:
        await update.message.reply_text(f"ℹ️ {sid} is not in your watchlist.")

async def cmd_fav(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = str(update.effective_chat.id)
    ul = load_favs().get(cid, [])
    if not ul:
        await update.message.reply_text("📋 Your watchlist is empty. Use `/add 2330` or click “❤️ Add to Watchlist” below a stock report.", parse_mode="Markdown", reply_markup=get_back_main_keyboard())
        return
    buttons, row = [], []
    for sid in ul:
        name = resolve_stock_name(sid)
        row.append(InlineKeyboardButton(f"🔍 {name} ({sid})", callback_data=f"tab:{sid}:1d:summary"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_main")])
    await update.message.reply_text(
        f"❤️ *My Watchlist* \\(Total {len(ul)} stocks\\)\nSelect a stock below to view its latest analysis:",
        parse_mode="MarkdownV2",
        reply_markup=InlineKeyboardMarkup(buttons)
    )

# ═════════════════════════════════════════════════════════════════════════════
#  Interactive Button Callback Handler (CallbackQueryHandler)
# ═════════════════════════════════════════════════════════════════════════════

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data

    if data == "close_panel":
        await query.answer()
        try:
            await query.message.delete()
        except Exception:
            pass
        return

    if data.startswith(("tab:", "refresh:")):
        _, stock_id, interval, tab = data.split(":")
        force = data.startswith("refresh:")
        await query.answer("Recalculating..." if force else "Switching view...")
        report = await analyze_stock_report(stock_id, interval, tab, force=force)
        try:
            await query.edit_message_text(
                report,
                parse_mode="MarkdownV2",
                reply_markup=get_check_keyboard(stock_id, interval, tab)
            )
        except BadRequest:
            pass
        return

    if data.startswith("addfav:"):
        sid = data.split(":")[1]
        cid = str(update.effective_chat.id)
        favs = load_favs()
        ul = favs.get(cid, [])
        if sid not in ul:
            ul.append(sid)
            favs[cid] = ul
            save_favs(favs)
            await query.answer(f"✅ Added {sid} to your watchlist!", show_alert=True)
        else:
            await query.answer(f"ℹ️ {sid}  is already in your watchlist!", show_alert=True)
        return

    if data == "buy_menu":
        await query.answer()
        await query.edit_message_text(
            get_scan_center_text(),
            reply_markup=get_buy_keyboard()
        )
        return

    if data == "buy:rescan":
        await query.answer("Starting scan and refreshing cache...")
        await query.edit_message_text(
            "🚀 *Running Full-Market Scan*\n"
            "Stage 1: Fast screening of listed and OTC stocks\n"
            "Stage 2: Full technical scoring of candidates\n"
            "This page will update automatically when the scan is complete.",
            parse_mode="MarkdownV2"
        )
        try:
            counts = await run_quick_scan()
        except Exception as e:
            err = esc(str(e)[:300])
            await query.edit_message_text(
                f"⚠️ *Full-Market Scan Failed*\n`{err}`\n\nPlease try again later.",
                parse_mode="MarkdownV2",
                reply_markup=get_buy_keyboard()
            )
            return

        _, latest_ts = db_read_category("explosive")
        if not latest_ts:
            _, latest_ts = db_read_category("pullback")
        update_time = format_scan_time(latest_ts)

        result_text = (
            f"✅ *Scan complete and saved to the SQLite local cache!*\n"
            f"🕒 Data Updated: `{esc(update_time)}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🌐 Valid Market Sample: `{counts.get('market_total', 0)}` stocks\n"
            f"🎯 Stage-One Candidates: `{counts.get('candidates', 0)}` stocks\n"
            f"📈 Strong Bullish Stocks: `{counts['explosive']}`\n"
            f"🔻 Bullish Pullback Stocks: `{counts['pullback']}`\n\n"
            f"Select a button below to view the results:"
        )
        await query.edit_message_text(
            result_text,
            parse_mode="MarkdownV2",
            reply_markup=get_buy_keyboard()
        )
        return

    if data.startswith("buy:cache:"):
        cat = data.split(":")[2]
        rows, ts = db_read_category(cat)

        if not rows:
            # Callback queries must be acknowledged immediately and only once.
            await query.answer("No cached data yet. Building the first full-market scan...")
            await query.edit_message_text(
                "⏳ *Building Initial Full-Market Cache*\n"
                "Stage 1: Scanning listed and OTC stocks\n"
                "Stage 2: Full technical scoring of candidates\n"
                "Results will be displayed automatically when the scan is complete.",
                parse_mode="MarkdownV2"
            )
            try:
                await run_quick_scan()
                rows, ts = db_read_category(cat)
            except Exception as e:
                err = esc(str(e)[:300])
                await query.edit_message_text(
                    f"⚠️ *Full-Market Scan Failed*\n`{err}`\n\nPlease go back and try again.",
                    parse_mode="MarkdownV2",
                    reply_markup=get_back_scan_keyboard()
                )
                return
        else:
            await query.answer()

        title = "📈 Strong Bullish Stock List" if cat == "explosive" else "🔻 Bullish Pullback Stock List"
        update_time = format_scan_time(ts)
        if not rows:
            result_text = (
                f"*{esc(title)}*\n"
                f"🕒 Data Updated: `{esc(update_time)}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"No stocks currently meet the criteria."
            )
        else:
            lines = [f"*{i}\\.* {r['entry_text']}" for i, r in enumerate(rows[:10], 1)]
            result_text = (
                f"🏆 *{esc(title)}*\n"
                f"🕒 Data Updated: `{esc(update_time)}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                + "\n\n".join(lines)
            )

        back_kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Back to Scan Menu", callback_data="buy_menu")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")]
        ])
        await query.edit_message_text(
            result_text,
            parse_mode="MarkdownV2",
            reply_markup=back_kb
        )
        return


    if data == "open_fav":
        await query.answer()
        cid = str(query.message.chat.id)
        ul = load_favs().get(cid, [])
        if not ul:
            await query.edit_message_text(
                "📋 Your watchlist is empty. Use /add 2330 or click “❤️ Add to Watchlist” below a stock report.",
                reply_markup=get_back_main_keyboard()
            )
            return
        buttons, row = [], []
        for sid in ul:
            name = resolve_stock_name(sid)
            row.append(InlineKeyboardButton(f"🔍 {name} ({sid})", callback_data=f"tab:{sid}:1d:summary"))
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        buttons.append([InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_main")])
        await query.edit_message_text(
            f"❤️ *My Watchlist* \\(Total {len(ul)} stocks\\)\nSelect a stock below to view its latest analysis:",
            parse_mode="MarkdownV2",
            reply_markup=InlineKeyboardMarkup(buttons)
        )
        return

    # Backward compatibility: legacy market-breadth buttons now route to the integrated scan center.
    if data in ("open_market", "market:rescan"):
        await query.answer("Market breadth is integrated into the scan center")
        await query.edit_message_text(
            get_scan_center_text(),
            reply_markup=get_buy_keyboard()
        )
        return

    if data == "open_point":
        await query.answer()
        point_text = get_point_text()
        back_kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_main")]]
        )
        try:
            await query.edit_message_text(
                point_text,
                reply_markup=back_kb
            )
        except BadRequest:
            # If the original message cannot be edited, send a new one so the button still appears responsive.
            await query.message.reply_text(
                point_text,
                reply_markup=back_kb
            )
        return

    if data == "back_main":
        await query.answer()
        text = (
            "🤖 *Taiwan Stock Quantitative Analysis Bot*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔍 `/check 2330` — Individual multi-factor technical analysis\n"
            "📊 `/buy` — Scan center (market breadth, strong stocks, pullbacks)\n"
            r"❤️ `/fav` — View watchlist \(`/add 2330` to add\)" "\n"
            "📖 `/point` — View multi-factor scoring rules\n\n"
            "👇 *Use the shortcut buttons below to get started:*"
        )
        await query.edit_message_text(text, parse_mode="MarkdownV2", reply_markup=get_main_menu_keyboard())
        return

# ═════════════════════════════════════════════════════════════════════════════
#  Startup and Initialization
# ═════════════════════════════════════════════════════════════════════════════

async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Handle uncaught exceptions by logging them and responding to the user when possible."""
    err = context.error
    print(f"❌ Unhandled error: {type(err).__name__}: {err}")

    try:
        if isinstance(update, Update):
            if update.callback_query:
                try:
                    await update.callback_query.answer()
                except Exception:
                    pass
                try:
                    await update.callback_query.message.reply_text(
                        "⚠️ An error occurred. Please return to the main menu and try again."
                    )
                except Exception:
                    pass
            elif update.effective_message:
                await update.effective_message.reply_text(
                    "⚠️ An error occurred. Please try again later."
                )
    except Exception:
        pass


async def post_init(app: Application):
    db_init()
    commands = [
        BotCommand("start", "Open main menu and shortcuts"),
        BotCommand("check", "Individual stock analysis /check 2330"),
        BotCommand("buy", "Scan center: market breadth + strong stocks + pullbacks"),
        BotCommand("fav", "View my watchlist"),
        BotCommand("add", "Add to watchlist /add 2330"),
        BotCommand("remove", "Remove from watchlist /remove 2330"),
        BotCommand("market", "Market breadth (integrated into /buy scan center)"),
        BotCommand("point", "View multi-factor scoring rules (-5 to +5)"),
    ]
    await app.bot.set_my_commands(commands)
    global AUTO_SCAN_TASK
    AUTO_SCAN_TASK = asyncio.create_task(auto_scan_loop())
    print(f"✅ Telegram commands and SQLite cache initialized. Auto-scan interval during market hours: {AUTO_SCAN_MINUTES} minutes.")

def main():
    if not TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN was not found. Set the environment variable before starting the bot.")
        return

    print("🚀 Starting Taiwan Stock Quantitative Analysis Telegram Bot...")
    app = ApplicationBuilder().token(TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("check", cmd_check))
    app.add_handler(CommandHandler("buy", cmd_buy))
    app.add_handler(CommandHandler("market", cmd_market))
    app.add_handler(CommandHandler("point", cmd_point))
    app.add_handler(CommandHandler("add", cmd_add))
    app.add_handler(CommandHandler("remove", cmd_remove))
    app.add_handler(CommandHandler("fav", cmd_fav))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_error_handler(global_error_handler)

    print("✅ Bot is online. Open Telegram and enter /start to test the interface.")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
