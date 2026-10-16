import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import datetime

def load_stock_metadata(csv_path: str = "stocks_default.csv") -> pd.DataFrame:
    """
    載入股票代號與名稱對照表
    支援欄位: symbol, name, industry, market
    """
    try:
        df = pd.read_csv(csv_path)
        # 確保必要欄位存在
        if "symbol" not in df.columns:
            raise ValueError("CSV 必須包含 'symbol' 欄位")
        if "name" not in df.columns:
            df["name"] = df["symbol"]
        return df
    except Exception as e:
        print(f"Error loading stock metadata from {csv_path}: {e}")
        # 返回最小預設
        return pd.DataFrame([
            {"symbol": "2330.TW", "name": "台積電", "industry": "半導體", "market": "TWSE"},
            {"symbol": "2317.TW", "name": "鴻海", "industry": "其他電子", "market": "TWSE"},
            {"symbol": "2454.TW", "name": "聯發科", "industry": "半導體", "market": "TWSE"},
            {"symbol": "2308.TW", "name": "台達電", "industry": "電子零組件", "market": "TWSE"},
            {"symbol": "2382.TW", "name": "廣達", "industry": "電腦週邊", "market": "TWSE"},
        ])

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算技術指標:
    - 均線: 5MA, 10MA, 20MA, 60MA
    - 量能: 5MA成交量 (Vol_MA5)
    - 漲跌幅、K線紅黑 (Is_Red)
    注意: 輸入 df['Volume'] 已轉換為「張」(Lots)
    """
    if df is None or len(df) < 5:
        return df

    df = df.copy()

    # 確保按日期由舊到新排序
    df.sort_index(ascending=True, inplace=True)

    # 均線計算
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA60"] = df["Close"].rolling(window=60).mean()

    # 成交量均線 (以張為單位)
    df["Vol_MA5"] = df["Volume"].rolling(window=5).mean()

    # 每日漲跌幅 (%) 與 K線實體漲跌
    df["Change_Pct"] = df["Close"].pct_change() * 100
    df["Is_Red"] = df["Close"] > df["Open"]

    return df

def fetch_stock_data(symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    """
    使用 yfinance 抓取單檔台股日 K 線資料
    並將成交量單位轉換為「張」(股數 / 1000)
    """
    try:
        # yfinance ticker
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, auto_adjust=False)

        if df is None or df.empty or len(df) < 20:
            return None

        # 整理欄位
        # 移除時區資訊以利比較與繪圖
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 檢查欄位
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                return None

        # 台股成交量單位轉換: 將股數 (Shares) 除以 1,000 轉換為「張」(Lots)
        df["Volume"] = df["Volume"] / 1000.0

        # 計算指標
        df = calculate_technical_indicators(df)

        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def fetch_multiple_stocks(symbols: List[str], period: str = "6mo", progress_callback=None) -> Dict[str, pd.DataFrame]:
    """
    批次抓取多檔股票歷史資料
    """
    data_dict = {}
    total = len(symbols)
    for idx, sym in enumerate(symbols):
        df = fetch_stock_data(sym, period=period)
        if df is not None and not df.empty:
            data_dict[sym] = df
        if progress_callback:
            progress_callback(idx + 1, total, sym)
    return data_dict
