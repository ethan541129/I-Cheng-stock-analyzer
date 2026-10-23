# 台股量化技術分析 Dashboard

此網站版由原本的 Telegram 股票分析機器人改寫而成，保留原本多因子評分邏輯，並使用 Streamlit 做視覺化展示。

## 網站功能
- 個股分析：K 線、MA5 / MA20 / MA60、RSI、KD、MACD、ATR、布林通道
- 多因子評分：-5 ～ +5 分
- 操作策略：新進場 / 已持有 / 停損 / 目標價
- 全市場兩階段強勢股掃描
- 多頭回調股
- 大盤廣度
- 評分規則頁面

## 本機執行
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 部署到 Streamlit Community Cloud
1. 把 `streamlit_app.py`、`requirements.txt`、`README.md` 上傳到 GitHub repository。
2. 登入 Streamlit Community Cloud。
3. 選擇該 GitHub repository。
4. Main file path 指定 `streamlit_app.py`。
5. Deploy。

## 注意
- Yahoo Finance 對部分台股可能沒有資料。
- 全市場掃描第一次執行較慢。
- 本專案僅供課程與技術分析展示，不構成投資建議。
