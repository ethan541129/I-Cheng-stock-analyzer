"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   台股量化技術分析 Telegram Bot (Clean & Interactive Edition)                ║
║   核心指標: 均線 (MA) / 布林通道 / RSI / KD / MACD / 成交量                   ║
║   評分系統: 多因子動能加減分模型 (-5 至 +5 分) + SQLite 本地快取               ║
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

# 避免 yfinance 在新版 pandas 下大量輸出 inplace deprecation 警告
warnings.filterwarnings(
    "ignore",
    message=r".*inplace keyword.*deprecated.*",
)

# 降低 yfinance 對無效/下市/未收錄代號的 console 噪音
logging.getLogger("yfinance").setLevel(logging.CRITICAL)



# ═════════════════════════════════════════════════════════════════════════════
#  基本設定與常數
# ═════════════════════════════════════════════════════════════════════════════

TOKEN = "telegram_bot_token"
TW_TZ = timezone(timedelta(hours=8))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "scan_cache.db")
FAV_FILE = os.path.join(BASE_DIR, "favorites.json")

# 常用快捷股票清單：只用於主選單快捷查詢與股票名稱備援，不參與全市場強勢股掃描。
# 可依個人常查股票自行增刪。
QUICK_POOL = [
    # ETF / 大型權值
    ("0050", "元大台灣50"), ("0056", "元大高股息"),
    ("2330", "台積電"), ("2317", "鴻海"), ("2454", "聯發科"), ("2308", "台達電"),
    ("2382", "廣達"), ("2357", "華碩"), ("2379", "瑞昱"), ("2327", "國巨"),

    # 半導體 / 記憶體 / IC
    ("2303", "聯電"), ("2408", "南亞科"), ("2344", "華邦電"), ("2337", "旺宏"),
    ("8299", "群聯"), ("3034", "聯詠"), ("3661", "世芯-KY"), ("3443", "創意"),
    ("3711", "日月光投控"), ("2449", "京元電子"), ("6239", "力成"), ("8150", "南茂"),
    ("3583", "辛耘"), ("6187", "萬潤"), ("3131", "弘塑"), ("6643", "M31"),

    # AI / 伺服器 / 散熱 / PCB
    ("3231", "緯創"), ("6669", "緯穎"), ("3017", "奇鋐"), ("3324", "雙鴻"),
    ("2345", "智邦"), ("2383", "台光電"), ("3037", "欣興"), ("8046", "南電"),
    ("2368", "金像電"), ("3013", "晟銘電"), ("2356", "英業達"), ("2376", "技嘉"),

    # 網通 / 光通訊
    ("3081", "聯亞"), ("3363", "上詮"), ("4979", "華星光"), ("6442", "光聖"),
    ("3163", "波若威"), ("4908", "前鼎"),

    # 電力 / 重電 / 被動元件
    ("1519", "華城"), ("1513", "中興電"), ("1503", "士電"), ("1609", "大亞"),
    ("2492", "華新科"), ("3026", "禾伸堂"),

    # 航運
    ("2603", "長榮"), ("2609", "陽明"), ("2615", "萬海"), ("2618", "長榮航"),

    # 金融
    ("2881", "富邦金"), ("2882", "國泰金"), ("2891", "中信金"), ("2886", "兆豐金"),

    # 其他熱門電子 / 通路
    ("3702", "大聯大"), ("3036", "文曄"), ("8112", "至上"), ("8069", "元太")
]

AUTO_SCAN_TASK = None
_report_cache: dict = {}
global_market_status = {"above_ma20": 0, "total": 0, "time": "尚未更新"}

# 全市場掃描設定
MARKET_POOL_CACHE: list[tuple[str, str, str]] = []   # (代號, 名稱, Yahoo suffix)
MARKET_POOL_CACHE_DATE = ""
MARKET_LOOKUP: dict[str, tuple[str, str]] = {}  # code -> (name, suffix)
SCAN_LOCK = asyncio.Lock()                       # 防止手動與排程同時重複掃描
FIRST_STAGE_KEEP = 50          # 第一層快篩後保留候選檔數
BATCH_SIZE = 80               # Yahoo Finance 批次下載檔數，降低請求數
STRONG_BULL_SCORE = 2.0
PULLBACK_MIN_SCORE = 0.0

MIN_AVG_TURNOVER = 8_000_000  # 20日平均成交金額至少 800 萬，排除過度冷門股

# ═════════════════════════════════════════════════════════════════════════════
#  輔助工具與 SQLite 資料庫快取
# ═════════════════════════════════════════════════════════════════════════════

def esc(text) -> str:
    """跳脫 Telegram MarkdownV2 特殊字元"""
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
    """初始化 SQLite 資料庫（用於快取掃描結果）"""
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
    # 另外保存每個分類最後一次掃描時間。
    # 即使某次掃描結果為 0 檔，也能正確顯示「資料更新時間」。
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
    print("✅ SQLite 本地快取資料庫初始化完成")

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
    """將 SQLite Unix timestamp 轉成台灣時間顯示。"""
    if not ts:
        return "尚未更新"
    return datetime.datetime.fromtimestamp(ts, TW_TZ).strftime("%Y-%m-%d %H:%M:%S")


def db_write_market_status(above_ma20: int, total: int, ts: int | None = None):
    """保存最近一次全市場掃描得到的大盤廣度。"""
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
    """讀取最近一次已保存的大盤廣度，不重新抓 Yahoo。"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT above_ma20, total, timestamp FROM market_status WHERE id = 1")
    row = cur.fetchone()
    con.close()
    if not row:
        return {"above_ma20": 0, "total": 0, "time": "尚未更新", "timestamp": 0}
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
    取得台灣上市＋上櫃普通股清單。
    優先使用 twstock 內建代號表，不需要逐檔向 Yahoo 查詢；每天重建一次即可。
    回傳格式：(股票代號, 名稱, Yahoo suffix)。
    若 twstock 無法使用，退回 QUICK_POOL。
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

            # 只保留上市/上櫃股票，排除 ETF、權證、特別股等非普通股票商品。
            if market not in ("上市", "上櫃"):
                continue
            if kind and "股票" not in kind:
                continue

            suffix = ".TW" if market == "上市" else ".TWO"
            rows.append((code, name, suffix))
    except Exception as e:
        print(f"⚠️ 無法建立全市場股票池，暫用 QUICK_POOL：{e}")

    if not rows:
        # twstock 不可用時只做備援；實際抓價仍會由 fetch_stock_df 同時嘗試 .TW / .TWO。
        rows = [(code, name, "") for code, name in QUICK_POOL]

    # 去重並依代號排序，同時建立代號 -> 名稱 / Yahoo suffix 快速索引。
    dedup = {}
    for code, name, suffix in rows:
        dedup[code] = (code, name, suffix)
    MARKET_POOL_CACHE = sorted(dedup.values(), key=lambda x: x[0])
    MARKET_POOL_CACHE_DATE = today
    MARKET_LOOKUP.clear()
    MARKET_LOOKUP.update({code: (name, suffix) for code, name, suffix in MARKET_POOL_CACHE})
    print(f"✅ 全市場股票池建立完成：{len(MARKET_POOL_CACHE)} 檔")
    return MARKET_POOL_CACHE


def _extract_symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """從 yfinance 多股票批次資料中取出單一股票 OHLCV。"""
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
    第一層：全市場批次快篩。
    價格趨勢、20日漲幅、量比、流動性快速排名
    回傳候選清單、站上月線檔數、有效樣本數。
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
            print(f"⚠️ 全市場批次 {start // BATCH_SIZE + 1} 下載失敗：{e}")
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

                # 太冷門的股票不進第二層，避免訊號失真與成交困難。
                if avg_turnover < MIN_AVG_TURNOVER:
                    continue

                # 第一層不是最終買賣分數，只負責把「近期真的有轉強」的股票送進第二層。
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

        # 稍微讓出事件迴圈，避免 Telegram bot 長時間完全無回應。
        await asyncio.sleep(0.05)

    candidates.sort(key=lambda x: (x["fast_score"], x["ret20"], x["vol_ratio"]), reverse=True)
    return candidates[:limit], above_ma20, valid_total

# ═════════════════════════════════════════════════════════════════════════════
#  股票代號與資料抓取
# ═════════════════════════════════════════════════════════════════════════════

def clean_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """清理 Yahoo Finance 回傳資料；無有效 OHLC 的代號直接視為無資料。"""
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
    """優先使用已建立的全市場索引，避免每次查詢都重新走 twstock。"""
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
    """已知上市/上櫃別時只抓正確 suffix；未知時才同時嘗試 .TW / .TWO。"""
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
            print(f"⚠️ {symbol} 抓價失敗：{e}")
    return pd.DataFrame()


async def fetch_candidate_frames(candidates: list[dict], period: str = "6mo") -> dict[str, pd.DataFrame]:
    """第二階段候選股批次下載，避免前 50 檔逐檔向 Yahoo 重複發 request。"""
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
            print(f"⚠️ 第二階段批次下載失敗：{e}")
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
#  經典技術指標計算 (MA / 布林通道 / RSI / KD / MACD / ATR / 成交量)
# ═════════════════════════════════════════════════════════════════════════════

def calculate_indicators(df: pd.DataFrame) -> dict:
    # 再做一次防呆：只使用有有效 OHLC 價格的 K 棒
    df = df.dropna(subset=["Close", "High", "Low"]).copy()
    if len(df) < 2:
        raise ValueError("有效股價資料不足，無法計算技術指標")

    close = df["Close"].squeeze().astype(float)
    high = df["High"].squeeze().astype(float)
    low = df["Low"].squeeze().astype(float)
    volume = df["Volume"].squeeze().fillna(0).astype(float)

    # 1. 移動平均線 (MA5, MA20, MA60)
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(min(60, len(close))).mean()

    # 2. 布林通道 (20日, 2倍標準差)
    std20 = close.rolling(20).std()
    bb_up = ma20 + 2 * std20
    bb_low = ma20 - 2 * std20

    # 3. RSI (14日)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # 4. KD 指標 (9日)
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

    # 6. ATR 波動度 (14日，用於計算停損位)
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # 7. 成交量均量 (20日)
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
#  多因子動能評分系統 (-5 到 +5 分，直觀好懂)
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_score(ind: dict) -> dict:
    """
    依據均線、RSI、KD、MACD、布林通道與成交量進行加減分 (-5 ~ +5)
    """
    score = 0.0
    bull_signals = []
    bear_signals = []

    c = ind["c_now"]
    ma5, ma20, ma60 = ind["ma5"], ind["ma20"], ind["ma60"]

    # 1. 均線排列與月線位置 (權重 ±1.5)
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

    # 2. RSI 強弱與超買超賣 (權重 ±1.0)
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

    # 3. KD 指標金叉/死叉與多空 (權重 ±1.0)
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

    # 4. MACD 趨勢動能 (權重 ±1.0)
    if ind["dif_now"] > ind["macd_now"]:
        score += 1.0
        bull_signals.append("MACD 多頭柱狀 (+1.0)")
    else:
        score -= 1.0
        bear_signals.append("MACD 空頭柱狀 (-1.0)")

    # 5. 成交量與布林通道突破 (權重 +1.0 / -0.5)
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
    # 將 -5 ~ +5 換算成 0 ~ 100% 的多頭信心百分比
    bull_pct = (final_score + 5.0) / 10.0 * 100.0

    return {
        "score": final_score,
        "bull_pct": bull_pct,
        "bull_signals": bull_signals,
        "bear_signals": bear_signals,
        "vol_ratio": vol_ratio
    }

# ═════════════════════════════════════════════════════════════════════════════
#  個股操作策略生成
# ═════════════════════════════════════════════════════════════════════════════

def build_trading_strategy(ind: dict, ev: dict) -> dict:
    """把量化分數轉成「新進場 / 已持有」可執行策略。"""
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

    warnings = []
    if rsi >= 75: warnings.append("RSI 偏高，避免追價")
    if vol_ratio >= 2.0: warnings.append("成交量明顯放大，留意隔日震盪")
    if c < ma60: warnings.append("仍在季線下，中期趨勢偏弱")
    return {"stance": stance, "new_entry": new_entry, "holding": holding,
            "warning": "；".join(warnings) if warnings else "無額外技術面警示"}

# ═════════════════════════════════════════════════════════════════════════════
#  個股完整分析報告生成（支援「綜合 / 技術 / 風控」三大分頁）
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
        return f"⚠️ 查無 `{esc(stock_id)}` 的股價資料，請確認代號是否正確。"

    ind = calculate_indicators(df)
    ev = evaluate_score(ind)

    c_now = ind["c_now"]
    pct_ch = (c_now - ind["c_prev"]) / (ind["c_prev"] + 1e-10) * 100.0
    score = ev["score"]
    bull_pct = ev["bull_pct"]
    strategy = build_trading_strategy(ind, ev)
    bar = create_progress_bar(bull_pct)
    now_str = datetime.datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")

    # 停損與目標價計算 (利用 MA20、布林上軌與 2倍 ATR)
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
            action = "🟢 強勢偏多（動能充沛）"
        elif score >= 0.5:
            action = "🟡 溫和偏多（拉回逢低布局）"
        elif score <= -2.5:
            action = "🔴 弱勢偏空（建議避開或減碼）"
        else:
            action = "⚪ 區間盤整（觀望等待方向）"

        trend_str = (
            "📈 均線多頭排列" if c_now > ind["ma20"] > ind["ma60"]
            else "📉 均線空頭排列" if c_now < ind["ma20"] < ind["ma60"]
            else "↔️ 均線糾結整理"
        )

        report = (
            f"📊 *【{esc(stock_name)} \\({esc(stock_id)}\\)】綜合評分報告 — {esc(interval)}*\n"
            f"🎯 最新報價: `{esc(c_str)}` \\(`{esc(pct_str)}`\\)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 *綜合行動建議*: {esc(action)}\n"
            f"🧮 *多因子動能評分*: `{esc(score_str)}` / 5\\.0\n"
            f"📊 *多頭強度*: `{bar}` `{esc(bull_pct_str)}`\n"
            f"🌊 *均線格局*: {esc(trend_str)}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📝 *操作策略*\n"
            f"  • 新進場: {esc(strategy['new_entry'])}\n"
            f"  • 已持有: {esc(strategy['holding'])}\n"
            f"  • 注意: {esc(strategy['warning'])}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 *參考操作區間*\n"
            f"  • 支撐觀察 \\(MA5/MA20\\): `{esc(ma5_str)}` \\~ `{esc(ma20_str)}`\n"
            f"  • 布林上軌目標價: `{esc(target_str)}`\n"
            f"  • 建議停損參考位: `{esc(stop_str)}`\n"
            f"\n⏱ `{esc(now_str)}`"
        )

    elif tab == "technical":
        rsi_str = f"{ind['rsi']:.1f}"
        k_str = f"{ind['k_now']:.1f}"
        d_str = f"{ind['d_now']:.1f}"
        dif_str = f"{ind['dif_now']:.2f}"
        macd_str = f"{ind['macd_now']:.2f}"
        vol_r_str = f"{ev['vol_ratio']:.2f}倍"

        bull_txt = "、".join(ev["bull_signals"]) if ev["bull_signals"] else "無明顯看多訊號"
        bear_txt = "、".join(ev["bear_signals"]) if ev["bear_signals"] else "無明顯看空訊號"

        report = (
            f"📐 *【{esc(stock_name)} \\({esc(stock_id)}\\)】技術指標明細 — {esc(interval)}*\n"
            f"🎯 最新報價: `{esc(c_str)}` \\(`{esc(pct_str)}`\\)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 *核心技術指標*\n"
            f"  • 均線: MA5 `{esc(ma5_str)}` \\| MA20 `{esc(ma20_str)}` \\| MA60 `{esc(ma60_str)}`\n"
            f"  • RSI \\(14\\): `{esc(rsi_str)}`\n"
            f"  • KD \\(9\\): K\\=`{esc(k_str)}` / D\\=`{esc(d_str)}`\n"
            f"  • MACD: DIF\\=`{esc(dif_str)}` / MACD\\=`{esc(macd_str)}`\n"
            f"  • 成交量能: 當日 `{esc(format_volume(ind['vol_now']))}` \\(均量 `{esc(vol_r_str)}`\\)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔔 *多因子評分觸發項目*\n"
            f"  🟢 加分項: {esc(bull_txt)}\n"
            f"  🔴 扣分項: {esc(bear_txt)}\n"
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
            f"🛡️ *【{esc(stock_name)} \\({esc(stock_id)}\\)】停損與風控參考 — {esc(interval)}*\n"
            f"🎯 最新報價: `{esc(c_str)}` \\(`{esc(pct_str)}`\\)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⚙️ *波動度與布林通道*\n"
            f"  • 布林上軌 \\(壓力\\): `{esc(bb_up_str)}`\n"
            f"  • 布林中軌 \\(月線\\): `{esc(ma20_str)}`\n"
            f"  • 布林下軌 \\(支撐\\): `{esc(bb_low_str)}`\n"
            f"  • ATR \\(14日真實波動\\): `{esc(atr_str)}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⚖️ *風報比評估*\n"
            f"  • 預估目標價: `{esc(target_str)}` \\(`+{esc(f'{reward_pct:.1f}%')}`\\)\n"
            f"  • 建議停損價: `{esc(stop_str)}` \\(`-{esc(f'{risk_pct:.1f}%')}`\\)\n"
            f"  • 預估風報比: `{esc(f'{rr_ratio:.1f}')} : 1`\n"
            f"\n⏱ `{esc(now_str)}`"
        )

    _report_cache[cache_key] = (time.time(), report)
    return report

# ═════════════════════════════════════════════════════════════════════════════
#  批次掃描與 SQLite 快取更新
# ═════════════════════════════════════════════════════════════════════════════

async def run_quick_scan() -> dict:
    """
    全市場兩階段選股：
    1) 上市＋上櫃批次快篩，保留前 FIRST_STAGE_KEEP 檔。
    2) 候選股以批次方式抓 6 個月資料，再跑完整多因子評分並寫入 SQLite。
    SCAN_LOCK 可避免手動重掃與 30 分鐘排程撞在一起。
    """
    global global_market_status

    if SCAN_LOCK.locked():
        print("ℹ️ 已有掃描正在執行，本次略過重複掃描。")
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

        print("🔎 第一層：開始掃描上市＋上櫃全市場...")
        candidates, above_ma20_count, total_valid = await first_stage_market_scan(FIRST_STAGE_KEEP)
        print(f"✅ 第一層完成：有效 {total_valid} 檔，留下 {len(candidates)} 檔候選")

        # 一次批次抓候選股的完整 6 個月日 K，避免逐檔 50 次 request。
        frames = await fetch_candidate_frames(candidates, period="6mo")

        for cand in candidates:
            code, name = cand["ticker"], cand["name"]
            df = frames.get(code, pd.DataFrame())
            if df.empty:
                # 極少數批次遺漏時才單檔補抓。
                df = await fetch_stock_df(code, interval="1d", period="6mo")
            if df.empty:
                continue
            try:
                ind = calculate_indicators(df)
                ev = evaluate_score(ind)
            except Exception as e:
                print(f"⚠️ {code} 指標計算失敗：{e}")
                continue

            c_now = ind["c_now"]
            pct_ch = (c_now - ind["c_prev"]) / (ind["c_prev"] + 1e-10) * 100.0
            sigs_short = ev["bull_signals"][:2] if ev["bull_signals"] else ["技術面整理"]

            entry_text = (
                f"*{esc(name)} \\({esc(code)}\\)* 現價:`{esc(f'{c_now:.2f}')}` \\(`{esc(f'{pct_ch:+.1f}%')}`\\)\n"
                f"   ├ 🧮 完整評分: `{esc(f'{ev['score']:+.1f}')}` / 5\\.0 ｜ RSI: `{esc(f'{ind['rsi']:.0f}')}`\n"
                f"   ├ ⚡ 快篩排名分: `{esc(f'{cand['fast_score']:.1f}')}` ｜ 20日: `{esc(f'{cand['ret20']:+.1f}%')}`\n"
                f"   └ 🔔 訊號: {esc('、'.join(sigs_short))}"
            )

            item = {
                "ticker": code, "name": name, "score": ev["score"],
                "signals": ev["bull_signals"], "entry_text": entry_text,
            }

            if ev["score"] >= 1.5 and c_now > ind["ma20"]:
                bull_list.append(item)
            elif PULLBACK_MIN_SCORE <= ev["score"] < STRONG_BULL_SCORE and c_now > ind["ma60"] and pct_ch < 0:
                pullback_list.append(item)

        # 同分時再參考快篩順序，結果比較穩定。
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
            # 全市場掃描已經計算站上 MA20 的檔數，直接同步保存大盤廣度，避免 /market 再掃一次。
            db_write_market_status(above_ma20_count, total_valid, scan_ts)

        return {
            "explosive": len(bull_list),
            "pullback": len(pullback_list),
            "market_total": total_valid,
            "candidates": len(candidates),
            "skipped": False,
        }

# ═════════════════════════════════════════════════════════════════════════════
#  盤中自動更新：每 30 分鐘掃描一次
# ═════════════════════════════════════════════════════════════════════════════

AUTO_SCAN_MINUTES = 30


def is_tw_market_scan_time(now: datetime.datetime | None = None) -> bool:
    """僅在台股交易日盤中執行自動掃描，避免盤後持續打 Yahoo Finance。"""
    now = now or datetime.datetime.now(TW_TZ)
    if now.weekday() >= 5:  # 週六、週日
        return False
    hhmm = now.hour * 60 + now.minute
    return 9 * 60 <= hhmm <= 13 * 60 + 35


async def auto_scan_loop():
    """盤中每 30 分鐘自動更新；任何錯誤都在迴圈內處理，不讓背景 task 崩潰。"""
    while True:
        try:
            now = datetime.datetime.now(TW_TZ)

            # 只在週一到週五台股盤中執行
            is_weekday = now.weekday() < 5
            minutes = now.hour * 60 + now.minute
            in_session = (9 * 60) <= minutes <= (13 * 60 + 35)

            if is_weekday and in_session:
                # 對齊整點 / 半點附近執行
                if now.minute in (0, 30):
                    try:
                        print(f"🔄 自動掃描開始：{now.strftime('%Y-%m-%d %H:%M:%S')}")
                        await run_quick_scan()
                        print("✅ 自動掃描完成")
                    except Exception as e:
                        print(f"⚠️ 自動掃描失敗：{e}")

                    # 避免同一分鐘重複觸發
                    await asyncio.sleep(61)
                    continue

            # 一般輪詢間隔
            await asyncio.sleep(20)

        except asyncio.CancelledError:
            print("ℹ️ 自動掃描 task 已停止")
            raise
        except Exception as e:
            # 就算時間判斷或其他背景邏輯出錯，也不要讓 task 結束
            print(f"⚠️ 自動掃描背景迴圈錯誤：{e}")
            await asyncio.sleep(30)


def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    """主選單快捷按鈕；大盤廣度已整合進掃描中心。"""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📈 台積電 (2330)", callback_data="tab:2330:1d:summary"),
            InlineKeyboardButton("📱 聯發科 (2454)", callback_data="tab:2454:1d:summary"),
            InlineKeyboardButton("⚡ 鴻海 (2317)", callback_data="tab:2317:1d:summary")
        ],
        [
            InlineKeyboardButton("📊 掃描中心＋大盤廣度", callback_data="buy_menu")
        ],
        [
            InlineKeyboardButton("🔥 強勢股", callback_data="buy:cache:explosive"),
            InlineKeyboardButton("🔻 多頭回調", callback_data="buy:cache:pullback")
        ],
        [
            InlineKeyboardButton("❤️ 我的自選股 (/fav)", callback_data="open_fav"),
            InlineKeyboardButton("📖 評分規則 (/point)", callback_data="open_point")
        ]
    ])

def get_back_main_keyboard() -> InlineKeyboardMarkup:
    """通用：返回主選單"""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 返回主選單", callback_data="back_main")]
    ])


def get_back_scan_keyboard() -> InlineKeyboardMarkup:
    """通用：返回掃描選單 / 主選單"""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 返回掃描選單", callback_data="buy_menu")],
        [InlineKeyboardButton("🏠 主選單", callback_data="back_main")]
    ])


def get_check_keyboard(stock_id: str, interval: str = "1d", tab: str = "summary") -> InlineKeyboardMarkup:
    """個股分析下方的分頁、週期切換與自選按鈕"""
    tabs = [
        ("📊 綜合", "summary"),
        ("📐 技術", "technical"),
        ("🛡️ 風控", "risk")
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
        for lbl, iv in [("5分", "5m"), ("15分", "15m"), ("60分", "60m"), ("日K", "1d"), ("週K", "1wk")]
    ]
    row3 = [
        InlineKeyboardButton("🔄 重新整理", callback_data=f"refresh:{stock_id}:{interval}:{tab}"),
        InlineKeyboardButton("❤️ 加入自選", callback_data=f"addfav:{stock_id}"),
        InlineKeyboardButton("❌ 關閉", callback_data="close_panel")
    ]
    row4 = [InlineKeyboardButton("🔙 返回主選單", callback_data="back_main")]
    return InlineKeyboardMarkup([row1, row2, row3, row4])

def get_scan_center_text() -> str:
    """掃描中心：整合全市場掃描結果、大盤廣度、更新時間與使用說明。"""
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
            "🟢 多頭健康" if pct >= 50
            else "🟡 震盪整理" if pct >= 30
            else "🔴 空頭偏弱"
        )
        breadth_text = (
            f"🌡️ 大盤廣度：{pct:.1f}%（{above}/{total} 檔站上 MA20）\n"
            f"📊 廣度：{breadth_bar}  {breadth_status}\n"
            f"🕒 廣度更新：{market_cache.get('time', updated_text)}\n"
        )
    else:
        breadth_text = (
            "🌡️ 大盤廣度：尚未建立資料\n"
            "📊 完成一次全市場掃描後，會同步計算站上 MA20 的股票比例。\n"
        )

    return (
        "📊 全市場量化掃描中心\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🕒 最近掃描更新：{updated_text}\n"
        f"🔥 強勢多頭：{len(bull_rows)} 檔 ｜ 🔻 多頭回調：{len(pull_rows)} 檔\n"
        + breadth_text +
        f"🔄 自動更新：台股盤中每 {AUTO_SCAN_MINUTES} 分鐘一次（整點／半點）\n"
        "🖐️ 手動更新：按下方『立即全市場重新選股』，會一次同步更新選股結果與大盤廣度。\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🌡️【大盤廣度怎麼看？】\n"
        "• 大盤廣度 = 全市場中，股價站上 20 日均線（MA20）的股票比例。\n"
        "• 比例越高，代表上漲有較多股票共同參與，市場結構較健康。\n"
        "• 比例偏低，代表多數股票仍弱，可能只有少數權值股撐盤。\n"
        "• 本數據與強勢股掃描共用同一批全市場資料，不會為了看大盤廣度再重掃一次。\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🌐【第一層：全市場快速初篩】\n"
        "• 掃描上市／上櫃股票，依趨勢、5/20 日漲幅、量比與流動性排序。\n"
        "• 同一輪掃描同步統計站上 MA20 的檔數，直接產生上方大盤廣度。\n"
        "• 快篩後取前 50 檔進入第二階段完整技術評分。\n\n"
        "🎯【第二層：完整技術評分與分類】\n"
        "🔥 強勢多頭股\n"
        "• 綜合評分達多頭門檻，且股價站上 MA20。\n"
        "• 用來找目前趨勢與動能較強的標的。\n\n"
        "🔻 多頭回調股\n"
        "• 中期結構仍在 MA60 之上，但短線出現回檔。\n"
        "• 用來觀察多頭趨勢中的拉回機會。\n\n"
        "ℹ️ 前 50 檔候選不一定都會進入上述兩類；未達條件者不會列入清單。\n\n"
        "請選擇要查看的分類，或手動更新整個市場資料："
    )


def get_buy_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📈 查看強勢多頭股", callback_data="buy:cache:explosive"),
            InlineKeyboardButton("🔻 查看多頭回調股", callback_data="buy:cache:pullback")
        ],
        [
            InlineKeyboardButton("🔄 立即全市場重新選股", callback_data="buy:rescan")
        ],
        [
            InlineKeyboardButton("🔙 返回主選單", callback_data="back_main")
        ]
    ])


def get_market_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 重新計算大盤廣度", callback_data="market:rescan")],
        [InlineKeyboardButton("🔙 返回主選單", callback_data="back_main")],
    ])


def format_market_breadth_text(s: dict, markdown: bool = False) -> str:
    """將大盤廣度快取轉成 Telegram 顯示文字。"""
    if not s or s.get("total", 0) <= 0:
        return (
            "🌡️ 台股市場廣度指標\n"
            "🕒 更新時間：尚未更新\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "目前還沒有大盤廣度快取。\n"
            "請先執行一次全市場掃描，或按下方『重新計算大盤廣度』。"
        )
    pct = s["above_ma20"] / max(s["total"], 1) * 100.0
    bar = create_progress_bar(pct)
    status_lbl = "🟢 多頭健康" if pct >= 50 else "🟡 震盪整理" if pct >= 30 else "🔴 空頭偏弱"
    if markdown:
        return (
            f"🌡️ *台股市場廣度指標 \\(Market Breadth\\)*\n"
            f"🕒 更新時間: `{esc(s['time'])}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 站上 20 日月線比例: `{esc(f'{pct:.1f}%')}` \\({s['above_ma20']}/{s['total']}\\)\n"
            f"📊 廣度進度: `{bar}`\n"
            f"💡 市場狀態: *{esc(status_lbl)}*\n"
            f"ℹ️ 此頁直接讀取最近一次全市場掃描快取，不會重複掃描。"
        )
    return (
        "🌡️ 台股市場廣度指標\n"
        f"🕒 更新時間：{s['time']}\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 站上 20 日月線比例：{pct:.1f}% ({s['above_ma20']}/{s['total']})\n"
        f"📊 廣度進度：{bar}\n"
        f"💡 市場狀態：{status_lbl}\n"
        "ℹ️ 此頁直接讀取最近一次全市場掃描快取，不會重複掃描。"
    )


async def calculate_market_breadth() -> dict:
    """批次計算台股市場廣度，Yahoo 無資料的股票自動略過。"""
    global global_market_status

    try:
        market_pool = get_full_market_pool()
    except Exception as e:
        print(f"⚠️ 建立市場股票池失敗: {e}")
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
            print(f"⚠️ 大盤廣度批次下載失敗：{e}")
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
        "🤖 台股量化技術分析機器人\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🔍 /check 2330 — 個股多因子技術分析\n"
        "📊 /buy — 全市場掃描中心（含大盤廣度／強勢股／回調股）\n"
        "❤️ /fav — 查看自選股清單\n"
        "📖 /point — 查看多因子評分規則\n\n"
        "👇 可直接點擊下方快捷按鈕開始操作："
    )
    await update.message.reply_text(
        text,
        reply_markup=get_main_menu_keyboard()
    )


async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("💡 請輸入股票代號，例如：/check 2330")
        return

    stock_id = context.args[0].strip()
    wait_msg = await update.message.reply_text(
        f"⏳ 正在計算 {stock_id} 技術指標與評分..."
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
    """相容舊指令：大盤廣度已整合到掃描中心，不再單獨重掃。"""
    await update.message.reply_text(
        "🌡️ 大盤廣度已整合到『全市場量化掃描中心』，與強勢股／回調股共用同一批掃描資料。\n\n"
        + get_scan_center_text(),
        reply_markup=get_buy_keyboard()
    )


def get_point_text() -> str:
    """評分規則使用純文字，避免 Telegram MarkdownV2 特殊字元問題。"""
    return (
        "📖 【多因子技術動能評分規則（-5 至 +5 分）】\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "1️⃣ 均線趨勢（MA5 / MA20 / MA60）\n"
        "  • 均線多頭排列：+1.5 ｜ 站上月線：+1.0\n"
        "  • 均線空頭排列：-1.5 ｜ 跌破月線：-1.0\n\n"
        "2️⃣ RSI 相對強弱指標（14日）\n"
        "  • 位於 55～75 強勢區間：+1.0\n"
        "  • 低於 30 超跌反彈區：+1.0\n"
        "  • 高於 80 過熱超買區：-1.0\n"
        "  • 低於 45 動能偏弱：-0.5\n\n"
        "3️⃣ KD 隨機指標（9日）\n"
        "  • KD 黃金交叉：+1.0\n"
        "  • K > D 多頭格局：+0.5\n"
        "  • KD 死亡交叉：-1.0\n"
        "  • K < D 空頭整理：-0.5\n\n"
        "4️⃣ MACD 趨勢動能（12, 26, 9）\n"
        "  • DIF > MACD：+1.0\n"
        "  • DIF < MACD：-1.0\n\n"
        "5️⃣ 成交量能與布林通道\n"
        "  • 成交量達 20 日均量 1.5 倍以上且收紅：+1.0\n"
        "  • 跌破布林下軌：+0.5（超跌反彈觀察）\n"
        "  • 突破布林上軌：-0.5（乖離過大）"
    )


async def cmd_point(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_point_text(), reply_markup=get_back_main_keyboard())


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("用法：`/add 2330`", parse_mode="Markdown")
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
    await update.message.reply_text(f"✅ 已將 {name} ({sid}) 加入自選股！輸入 /fav 可隨時查看。")

async def cmd_remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("用法：`/remove 2330`", parse_mode="Markdown")
        return
    sid = context.args[0].strip()
    cid = str(update.effective_chat.id)
    favs = load_favs()
    ul = favs.get(cid, [])
    if sid in ul:
        ul.remove(sid)
        favs[cid] = ul
        save_favs(favs)
        await update.message.reply_text(f"🗑️ 已將 {sid} 從自選股移除。")
    else:
        await update.message.reply_text(f"ℹ️ {sid} 不在你的自選股清單中。")

async def cmd_fav(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = str(update.effective_chat.id)
    ul = load_favs().get(cid, [])
    if not ul:
        await update.message.reply_text("📋 目前自選股為空，請輸入 `/add 2330` 或在個股報告下方點擊「❤️ 加入自選」！", parse_mode="Markdown", reply_markup=get_back_main_keyboard())
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
    buttons.append([InlineKeyboardButton("🔙 返回主選單", callback_data="back_main")])
    await update.message.reply_text(
        f"❤️ *我的自選股清單* \\(共 {len(ul)} 檔\\)\n請點擊下方按鈕查看即時分析：",
        parse_mode="MarkdownV2",
        reply_markup=InlineKeyboardMarkup(buttons)
    )

# ═════════════════════════════════════════════════════════════════════════════
#  互動按鈕回調處理器 (CallbackQueryHandler)
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
        await query.answer("重新計算中..." if force else "切換顯示中...")
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
            await query.answer(f"✅ 已將 {sid} 加入自選股！", show_alert=True)
        else:
            await query.answer(f"ℹ️ {sid} 已經在自選股中囉！", show_alert=True)
        return

    if data == "buy_menu":
        await query.answer()
        await query.edit_message_text(
            get_scan_center_text(),
            reply_markup=get_buy_keyboard()
        )
        return

    if data == "buy:rescan":
        await query.answer("開始掃描更新快取...")
        await query.edit_message_text(
            "🚀 *正在執行全市場掃描*\n"
            "第一階段：上市櫃快速初篩\n"
            "第二階段：候選股完整技術評分\n"
            "完成後此頁會自動更新。",
            parse_mode="MarkdownV2"
        )
        try:
            counts = await run_quick_scan()
        except Exception as e:
            err = esc(str(e)[:300])
            await query.edit_message_text(
                f"⚠️ *全市場掃描失敗*\n`{err}`\n\n請稍後再試。",
                parse_mode="MarkdownV2",
                reply_markup=get_buy_keyboard()
            )
            return

        _, latest_ts = db_read_category("explosive")
        if not latest_ts:
            _, latest_ts = db_read_category("pullback")
        update_time = format_scan_time(latest_ts)

        result_text = (
            f"✅ *掃描完成並已存入 SQLite 本地快取！*\n"
            f"🕒 資料更新時間: `{esc(update_time)}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🌐 全市場有效樣本: `{counts.get('market_total', 0)}` 檔\n"
            f"🎯 第一層候選: `{counts.get('candidates', 0)}` 檔\n"
            f"📈 強勢多頭股: `{counts['explosive']}` 檔\n"
            f"🔻 多頭回調股: `{counts['pullback']}` 檔\n\n"
            f"請點擊下方按鈕查看結果："
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
            # callback 必須立刻回覆，且只回覆一次。
            await query.answer("目前尚無快取資料，正在建立首次全市場掃描...")
            await query.edit_message_text(
                "⏳ *正在建立首次全市場快取*\n"
                "第一階段：掃描上市櫃股票\n"
                "第二階段：候選股完整技術評分\n"
                "完成後此頁會自動顯示結果。",
                parse_mode="MarkdownV2"
            )
            try:
                await run_quick_scan()
                rows, ts = db_read_category(cat)
            except Exception as e:
                err = esc(str(e)[:300])
                await query.edit_message_text(
                    f"⚠️ *全市場掃描失敗*\n`{err}`\n\n請返回後再試一次。",
                    parse_mode="MarkdownV2",
                    reply_markup=get_back_scan_keyboard()
                )
                return
        else:
            await query.answer()

        title = "📈 強勢多頭股清單" if cat == "explosive" else "🔻 多頭回調股清單"
        update_time = format_scan_time(ts)
        if not rows:
            result_text = (
                f"*{esc(title)}*\n"
                f"🕒 資料更新時間: `{esc(update_time)}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"目前無符合條件之標的。"
            )
        else:
            lines = [f"*{i}\\.* {r['entry_text']}" for i, r in enumerate(rows[:10], 1)]
            result_text = (
                f"🏆 *{esc(title)}*\n"
                f"🕒 資料更新時間: `{esc(update_time)}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                + "\n\n".join(lines)
            )

        back_kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 返回掃描選單", callback_data="buy_menu")],
            [InlineKeyboardButton("🏠 主選單", callback_data="back_main")]
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
                "📋 目前自選股為空，請輸入 /add 2330 或在個股報告下方點擊「❤️ 加入自選」！",
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
        buttons.append([InlineKeyboardButton("🔙 返回主選單", callback_data="back_main")])
        await query.edit_message_text(
            f"❤️ *我的自選股清單* \\(共 {len(ul)} 檔\\)\n請點擊下方按鈕查看即時分析：",
            parse_mode="MarkdownV2",
            reply_markup=InlineKeyboardMarkup(buttons)
        )
        return

    # 相容舊版按鈕：大盤廣度已整合進掃描中心，舊按鈕一律導回掃描中心。
    if data in ("open_market", "market:rescan"):
        await query.answer("大盤廣度已整合到掃描中心")
        await query.edit_message_text(
            get_scan_center_text(),
            reply_markup=get_buy_keyboard()
        )
        return

    if data == "open_point":
        await query.answer()
        point_text = get_point_text()
        back_kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🔙 返回主選單", callback_data="back_main")]]
        )
        try:
            await query.edit_message_text(
                point_text,
                reply_markup=back_kb
            )
        except BadRequest:
            # 若原訊息不能編輯，就直接另發一則，避免按鈕看起來完全沒反應。
            await query.message.reply_text(
                point_text,
                reply_markup=back_kb
            )
        return

    if data == "back_main":
        await query.answer()
        text = (
            "🤖 *台股量化技術分析機器人*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔍 `/check 2330` — 個股多因子技術分析\n"
            "📊 `/buy` — 掃描中心（含大盤廣度、強勢股、回調股）\n"
            r"❤️ `/fav` — 查看自選股清單 \(`/add 2330` 新增\)" "\n"
            "📖 `/point` — 查看多因子評分加減分規則\n\n"
            "👇 *可直接點擊下方快捷按鈕開始操作：*"
        )
        await query.edit_message_text(text, parse_mode="MarkdownV2", reply_markup=get_main_menu_keyboard())
        return

# ═════════════════════════════════════════════════════════════════════════════
#  啟動與初始化
# ═════════════════════════════════════════════════════════════════════════════

async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """避免未處理例外只印 traceback；同時記錄並盡量回覆使用者。"""
    err = context.error
    print(f"❌ 未處理錯誤: {type(err).__name__}: {err}")

    try:
        if isinstance(update, Update):
            if update.callback_query:
                try:
                    await update.callback_query.answer()
                except Exception:
                    pass
                try:
                    await update.callback_query.message.reply_text(
                        "⚠️ 操作發生錯誤，請返回主選單後再試一次。"
                    )
                except Exception:
                    pass
            elif update.effective_message:
                await update.effective_message.reply_text(
                    "⚠️ 操作發生錯誤，請稍後再試。"
                )
    except Exception:
        pass


async def post_init(app: Application):
    db_init()
    commands = [
        BotCommand("start", "開啟主選單與快捷按鈕"),
        BotCommand("check", "個股分析 /check 2330"),
        BotCommand("buy", "掃描中心：大盤廣度＋強勢股＋回調股"),
        BotCommand("fav", "查看我的自選股清單"),
        BotCommand("add", "新增自選股 /add 2330"),
        BotCommand("remove", "移除自選股 /remove 2330"),
        BotCommand("market", "大盤廣度（已整合至 /buy 掃描中心）"),
        BotCommand("point", "查看多因子評分規則 (-5~+5)"),
    ]
    await app.bot.set_my_commands(commands)
    global AUTO_SCAN_TASK
    AUTO_SCAN_TASK = asyncio.create_task(auto_scan_loop())
    print(f"✅ Telegram 選單指令與 SQLite 快取綁定完成！盤中每 {AUTO_SCAN_MINUTES} 分鐘自動更新一次。")

def main():
    if not TOKEN:
        print("❌ 找不到 TELEGRAM_BOT_TOKEN，請先設定環境變數再啟動。")
        return

    print("🚀 啟動台股量化分析 Telegram Bot (精簡互動版)...")
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

    print("✅ 機器人已上線！請至 Telegram 輸入 /start 測試按鈕")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
