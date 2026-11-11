# I-Cheng Stock Bot

I-Cheng Stock Bot is a Python project for analyzing Taiwan and U.S. stocks. I started it as a Telegram bot and later built a Streamlit dashboard for interactive charts and market scans.

## What you can do

### Taiwan stocks

- Look up a stock and view its candlestick chart with 5-, 20-, and 60-period moving averages.
- Check RSI, KD, MACD, ATR, and Bollinger Bands alongside a technical score from -5 to +5.
- Read entry and holding guidance, with reference prices for stops and targets.
- Scan the Taiwan market in two stages to find strong stocks and bullish pullbacks. Market breadth shows the share of stocks trading above their 20-day moving averages.

### U.S. stocks

- Look up tickers such as AAPL, NVDA, MU, and TSLA using the same technical scoring rules.
- Follow major U.S. indexes through SPY, QQQ, DIA, and IWM.
- Scan a selected group of actively traded U.S. stocks for strength.

### Dashboard and watchlist

- Keep Taiwan and U.S. stocks in one watchlist.
- Choose how much price history to show on the dashboard.

## How the technical score works

The score ranges from -5 to +5. It combines the moving-average trend, RSI momentum, KD crossovers, MACD direction, trading volume, and the price's position relative to the Bollinger Bands.

The dashboard uses the score to describe the setup as bullish, neutral, weak, or bearish and flag when a pullback may offer a better entry.

## Optional AI analysis

Unlock AI access with the password, then select **Generate AI analysis**. The dashboard calls the OpenAI API on request to keep API usage down.

The short-term view covers the trend, possible entries, how to manage an existing position, and support and risk levels. The longer-term view looks at the trend and any available company fundamentals. The summary gives an overall bullish, neutral, or bearish view.

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- yfinance
- twstock
- SQLite
- python-telegram-bot
- OpenAI API

## Project structure

```text
I-Cheng-stock-bot/
├── streamlit_app.py
├── telegram_bot.py
├── requirements.txt
└── README.md
```

`streamlit_app.py` runs the web dashboard, and `telegram_bot.py` runs the Telegram bot. Both use similar technical scoring logic.

## Run locally

Install the dependencies.

```bash
pip install -r requirements.txt
```

Start the dashboard.

```bash
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Upload `streamlit_app.py`, `requirements.txt`, and `README.md` to GitHub.
2. Connect the repository to Streamlit Community Cloud.
3. Set the main file path to `streamlit_app.py`.
4. Deploy the app.

Keep API keys and passwords in Streamlit Secrets. Use these entries to configure AI access.

```toml
OPENAI_API_KEY = "your-api-key"
AI_ACCESS_PASSWORD = "your-private-password"
OPENAI_MODEL = "gpt-6-luna"
```

## Why I built this

I wanted to practice building a tool around real market data. Writing the scoring rules in Python gave me a way to turn technical indicators into something I could read at a glance. Building both the dashboard and the Telegram bot gave me practice designing charts and deciding what information fits in a chat.

I also learned how to cache market scans, add an AI feature with password protection, and deploy a working app.

## Notes

- Yahoo Finance may return delayed data or have gaps for some tickers.
- The first Taiwan market scan may take longer than later scans.
- This project is for education, research, and portfolio demonstrations only. It does not provide financial advice.
