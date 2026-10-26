# I-Cheng Stock Bot

A Python-based stock analysis project that combines quantitative technical analysis, market screening, interactive visualization, and optional AI-assisted insights.

This project started as a Telegram stock analysis bot and was later expanded into a Streamlit web dashboard. The goal is to make stock data easier to understand through technical indicators, scoring models, market screening, and a simple visual interface.

## Project Overview

The platform currently supports both Taiwan and U.S. stocks.

### Taiwan Stock Features
- Individual stock analysis
- Candlestick charts with MA5 / MA20 / MA60
- RSI, KD, MACD, ATR, and Bollinger Bands
- Multi-factor technical score from -5 to +5
- Entry, holding, stop-loss, and target-price references
- Full-market two-stage stock screening
- Bullish pullback screening
- Market breadth analysis

### U.S. Stock Features
- Individual stock analysis using ticker symbols such as AAPL, NVDA, MU, and TSLA
- Technical analysis using the same scoring framework
- U.S. market overview using major ETFs such as SPY, QQQ, DIA, and IWM
- Popular-stock strength radar for selected high-liquidity U.S. stocks

### Shared Features
- Combined Taiwan and U.S. watchlist
- K-line history with selectable time ranges
- Streamlit dashboard interface
- Telegram bot interface
- Optional AI analysis for short-term and long-term views
- Password-protected AI access to control API usage

## Quantitative Scoring Model

The system evaluates multiple technical signals and converts them into a score between -5 and +5.

Main factors include:
- Moving-average trend
- RSI momentum
- KD crossovers
- MACD direction
- Trading volume
- Bollinger Band position

The score is then translated into a more readable market view such as bullish, neutral, pullback, weak, or bearish.

## AI Analysis

AI is used as an optional analysis layer rather than an automatic feature.

It can summarize:
- Short-term trend
- Entry and holding ideas
- Important support and risk levels
- Medium- to long-term trend
- Available fundamental information
- Overall bullish / neutral / bearish view

The AI function only runs when the user manually clicks the AI analysis button, helping reduce unnecessary API usage.

## Technologies Used

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

## Project Structure

```text
I-Cheng-stock-bot/
├── streamlit_app.py
├── telegram_bot.py
├── requirements.txt
└── README.md
```

- `streamlit_app.py` provides the web dashboard.
- `telegram_bot.py` provides the Telegram interface.
- Both versions use similar quantitative analysis logic.

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

## Deployment

The web version can be deployed through Streamlit Community Cloud.

Basic deployment steps:
1. Upload `streamlit_app.py`, `requirements.txt`, and `README.md` to GitHub.
2. Connect the GitHub repository to Streamlit Community Cloud.
3. Set the main file path to `streamlit_app.py`.
4. Deploy the application.

Sensitive information such as API keys and passwords should be stored in Streamlit Secrets instead of GitHub.

Example:

```toml
OPENAI_API_KEY = "your-api-key"
AI_ACCESS_PASSWORD = "your-private-password"
OPENAI_MODEL = "gpt-6-luna"
```

## Why I Built This Project

I built this project to practice combining financial data analysis, Python programming, visualization, API integration, and simple product design in one application.

The project also helped me learn how to:
- Work with real-world market data
- Design a technical scoring model
- Build both web and chatbot interfaces
- Handle market screening and caching
- Integrate AI safely into an application
- Deploy a working project online

## Notes

- Yahoo Finance data may occasionally be delayed or unavailable for some symbols.
- Full-market Taiwan screening may take longer on the first run.
- The U.S. market radar currently uses a selected group of liquid stocks rather than the entire U.S. market.
- This project is for educational, research, and portfolio purposes only and does not provide financial advice.
