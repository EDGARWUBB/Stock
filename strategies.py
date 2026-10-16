import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

def filter_liquidity(
    df: pd.DataFrame, 
    min_vol_ma5: float = 1000.0, 
    min_price: float = 15.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    基礎流動性過濾（必要條件）:
    1. 5 日平均成交量 大於 1,000 張 (min_vol_ma5)
    2. 當日收盤價 大於 15 元 (min_price)
    """
    if df is None or len(df) < 5:
        return False, {"reason": "資料筆數不足 5 日"}

    last_row = df.iloc[-1]
    vol_ma5 = last_row.get("Vol_MA5", 0.0)
    close_price = last_row.get("Close", 0.0)

    pass_vol = vol_ma5 >= min_vol_ma5
    pass_price = close_price >= min_price

    details = {
        "close_price": round(float(close_price), 2),
        "vol_ma5": round(float(vol_ma5), 1),
        "pass_vol": pass_vol,
        "pass_price": pass_price
    }

    if pass_vol and pass_price:
        return True, details
    else:
        reasons = []
        if not pass_vol:
            reasons.append(f"5日均量 {vol_ma5:.1f} 未達 {min_vol_ma5} 張")
        if not pass_price:
            reasons.append(f"收盤價 {close_price:.2f} 低於 {min_price} 元")
        details["reason"] = ", ".join(reasons)
        return False, details


def strategy_a_breakdown_reversal(
    df: pd.DataFrame,
    lookback_window: int = 25,
    exclude_recent: int = 3,
    vol_multiplier: float = 1.5
) -> Tuple[bool, Dict[str, Any]]:
    """
    策略 A：帶量破底翻（假跌破真反轉）
    - 前期低點定義：計算過去第 5 天到第 25 天的區間最低價（排除最近 3 天，稱為 Prev_Low）。
    - 破底動作：最近 3 個交易日內，曾出現最低價小於 Prev_Low。
    - 翻揚確認：今日收盤價重新站回 Prev_Low 之上，且今日為紅 K（收盤價 大於 開盤價）。
    - 量能確認：今日成交量 大於 5 日均量的 1.5 倍（確保有買盤實質承接，排除無量反彈）。
    """
    if df is None or len(df) < lookback_window + 5:
        return False, {"reason": f"資料天數不足 {lookback_window + 5} 天"}

    # 取過去區間（排除最近 exclude_recent 天）
    # 最近 exclude_recent 天對應 iloc[-exclude_recent:]
    # 前期區間對應 iloc[-(lookback_window) : -(exclude_recent)]
    prev_window = df.iloc[-(lookback_window) : -(exclude_recent)]
    if prev_window.empty:
        return False, {"reason": "前期計算區間為空"}

    prev_low = float(prev_window["Low"].min())

    # 最近 exclude_recent 天（包含今日）
    recent_window = df.iloc[-exclude_recent:]
    recent_min_low = float(recent_window["Low"].min())

    # 破底動作：最近 3 個交易日內曾出現最低價小於 Prev_Low
    had_breakdown = recent_min_low < prev_low

    # 今日狀況
    today = df.iloc[-1]
    today_close = float(today["Close"])
    today_open = float(today["Open"])
    today_vol = float(today["Volume"])
    today_vol_ma5 = float(today.get("Vol_MA5", 0.0))

    # 翻揚確認：今日收盤站回 Prev_Low 之上，且為紅 K
    reclaimed = today_close >= prev_low
    is_red = today_close > today_open

    # 量能確認：今日成交量 > 5日均量 * 1.5
    vol_ratio = (today_vol / today_vol_ma5) if today_vol_ma5 > 0 else 0.0
    vol_confirmed = vol_ratio >= vol_multiplier

    details = {
        "prev_low": round(prev_low, 2),
        "recent_min_low": round(recent_min_low, 2),
        "today_close": round(today_close, 2),
        "today_open": round(today_open, 2),
        "today_vol": round(today_vol, 1),
        "today_vol_ma5": round(today_vol_ma5, 1),
        "vol_ratio": round(vol_ratio, 2),
        "had_breakdown": had_breakdown,
        "reclaimed": reclaimed,
        "is_red": is_red,
        "vol_confirmed": vol_confirmed
    }

    if had_breakdown and reclaimed and is_red and vol_confirmed:
        return True, details
    return False, details


def strategy_b_consolidation_breakout(
    df: pd.DataFrame,
    consolidation_days: int = 10,
    max_amplitude_pct: float = 8.0,
    vol_multiplier: float = 2.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    策略 B：區間盤整帶量突破
    - 盤整定義：過去 10 個交易日（不含今日）的最高價與最低價振幅小於 8%。
    - 均線保護：今日收盤價位於 20MA 與 60MA 之上，且 20MA 斜率向上（今日 20MA 大於 昨日 20MA）。
    - 突破確認：今日收盤價創下近 11 日新高，且今日成交量 大於 5 日均量的 2 倍。
    """
    if df is None or len(df) < max(consolidation_days + 15, 65):
        return False, {"reason": "資料天數不足以計算 60MA 或盤整區間"}

    # 過去 consolidation_days 個交易日（不含今日）
    # 對應 iloc[-(consolidation_days + 1) : -1]
    range_df = df.iloc[-(consolidation_days + 1) : -1]
    if len(range_df) < consolidation_days:
        return False, {"reason": "盤整區間天數不足"}

    range_high = float(range_df["High"].max())
    range_low = float(range_df["Low"].min())

    if range_low <= 0:
        return False, {"reason": "盤整區間低價異常"}

    # 振幅定義: (最高 - 最低) / 最低 * 100%
    amplitude_pct = ((range_high - range_low) / range_low) * 100.0
    is_consolidating = amplitude_pct <= max_amplitude_pct

    # 今日與昨日
    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    today_close = float(today["Close"])
    today_vol = float(today["Volume"])
    today_vol_ma5 = float(today.get("Vol_MA5", 0.0))
    today_ma20 = float(today.get("MA20", 0.0))
    yest_ma20 = float(yesterday.get("MA20", 0.0))
    today_ma60 = float(today.get("MA60", 0.0))

    # 均線保護：今日收盤價位於 20MA 與 60MA 之上，且 20MA 斜率向上
    ma_protected = (today_close > today_ma20) and (today_close > today_ma60) and (today_ma20 > yest_ma20)

    # 突破確認：今日收盤價創下近 11 日新高（大於前 10 天所有收盤價及最高價）
    is_new_high = today_close >= range_high or today_close > float(range_df["Close"].max())

    # 量能確認：今日成交量 > 5日均量 * 2.0
    vol_ratio = (today_vol / today_vol_ma5) if today_vol_ma5 > 0 else 0.0
    vol_confirmed = vol_ratio >= vol_multiplier

    details = {
        "range_high": round(range_high, 2),
        "range_low": round(range_low, 2),
        "amplitude_pct": round(amplitude_pct, 2),
        "today_close": round(today_close, 2),
        "today_ma20": round(today_ma20, 2),
        "today_ma60": round(today_ma60, 2),
        "vol_ratio": round(vol_ratio, 2),
        "is_consolidating": is_consolidating,
        "ma_protected": ma_protected,
        "is_new_high": is_new_high,
        "vol_confirmed": vol_confirmed
    }

    if is_consolidating and ma_protected and is_new_high and vol_confirmed:
        return True, details
    return False, details


def strategy_c_pullback_support(
    df: pd.DataFrame,
    lookback_surge_days: int = 15,
    surge_gain_pct: float = 5.0,
    surge_vol_mult: float = 2.0,
    ma_bias_pct: float = 2.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    策略 C：強勢股量縮回測支撐（下飄旗／回測買點）
    - 強勢背景：60MA 斜率向上，且過去 15 日內曾出現單日漲幅大於 5% 且帶量（大於 5 日均量 2 倍）的攻擊紅 K。
    - 量縮整理：最近 3 至 5 日股價拉回接近 10MA 或 20MA（收盤價與均線乖離率在正負 2% 內），且回檔期間成交量萎縮（小於 5 日均量）。
    - 轉強訊號：今日收紅 K、成交量較昨日放大，且收盤價突破昨日最高價。
    """
    if df is None or len(df) < max(lookback_surge_days + 15, 65):
        return False, {"reason": "資料天數不足以計算 60MA 或攻擊回測"}

    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    today_close = float(today["Close"])
    today_open = float(today["Open"])
    today_high = float(today["High"])
    today_vol = float(today["Volume"])
    yest_close = float(yesterday["Close"])
    yest_high = float(yesterday["High"])
    yest_vol = float(yesterday["Volume"])

    today_ma60 = float(today.get("MA60", 0.0))
    yest_ma60 = float(yesterday.get("MA60", 0.0))

    # 1. 強勢背景：60MA 斜率向上
    ma60_up = today_ma60 > yest_ma60

    # 過去 15 日內曾出現單日漲幅大於 5% 且帶量（> 5日均量 2 倍）的攻擊紅 K
    # 不含今日：iloc[-(lookback_surge_days + 1) : -1]
    surge_window = df.iloc[-(lookback_surge_days + 1) : -1]
    surge_mask = (
        (surge_window["Change_Pct"] >= surge_gain_pct) &
        (surge_window["Volume"] >= surge_window["Vol_MA5"] * surge_vol_mult) &
        (surge_window["Close"] > surge_window["Open"])
    )
    had_surge_k = bool(surge_mask.any())

    # 2. 量縮整理：最近 3 至 5 日拉回接近 10MA 或 20MA（乖離率在正負 2% 內）
    # 檢查近 3~5 日（不含今日的轉強日）是否有回測 10MA 或 20MA
    pullback_window = df.iloc[-5:-1]
    near_ma_flag = False
    for _, row in pullback_window.iterrows():
        c = float(row["Close"])
        ma10 = float(row.get("MA10", 0.0))
        ma20 = float(row.get("MA20", 0.0))
        if ma10 > 0:
            bias_10 = abs(c - ma10) / ma10 * 100.0
            if bias_10 <= ma_bias_pct:
                near_ma_flag = True
                break
        if ma20 > 0:
            bias_20 = abs(c - ma20) / ma20 * 100.0
            if bias_20 <= ma_bias_pct:
                near_ma_flag = True
                break

    # 回檔期間成交量萎縮：回測期間至少有交易日成交量小於該日 5 日均量，或昨日量縮
    vol_shrunk = bool((pullback_window["Volume"] < pullback_window["Vol_MA5"]).any())

    # 3. 轉強訊號：今日收紅 K、成交量較昨日放大，且收盤價突破昨日最高價
    is_red = today_close > today_open
    vol_increased = today_vol > yest_vol
    break_yest_high = today_close > yest_high

    rebound_signal = is_red and vol_increased and break_yest_high

    details = {
        "today_close": round(today_close, 2),
        "yest_high": round(yest_high, 2),
        "today_vol": round(today_vol, 1),
        "yest_vol": round(yest_vol, 1),
        "ma60_up": ma60_up,
        "had_surge_k": had_surge_k,
        "near_ma_flag": near_ma_flag,
        "vol_shrunk": vol_shrunk,
        "is_red": is_red,
        "vol_increased": vol_increased,
        "break_yest_high": break_yest_high
    }

    if ma60_up and had_surge_k and near_ma_flag and vol_shrunk and rebound_signal:
        return True, details
    return False, details


def scan_single_stock(
    symbol: str, 
    df: pd.DataFrame, 
    name: str = "", 
    industry: str = "",
    min_vol_ma5: float = 1000.0,
    min_price: float = 15.0,
    enable_strat_a: bool = True,
    enable_strat_b: bool = True,
    enable_strat_c: bool = True,
    strat_a_params: Optional[Dict] = None,
    strat_b_params: Optional[Dict] = None,
    strat_c_params: Optional[Dict] = None
) -> Optional[Dict[str, Any]]:
    """
    對單檔股票執行基礎流動性過濾與各項策略檢驗
    """
    if df is None or len(df) < 60:
        return None

    # 1. 基礎流動性過濾（必要條件）
    liq_pass, liq_info = filter_liquidity(df, min_vol_ma5=min_vol_ma5, min_price=min_price)
    if not liq_pass:
        return None

    last_row = df.iloc[-1]
    signals = []
    strat_details = {}

    # 策略 A
    if enable_strat_a:
        p = strat_a_params or {}
        pass_a, info_a = strategy_a_breakdown_reversal(df, **p)
        if pass_a:
            signals.append("策略A: 帶量破底翻")
            strat_details["A"] = info_a

    # 策略 B
    if enable_strat_b:
        p = strat_b_params or {}
        pass_b, info_b = strategy_b_consolidation_breakout(df, **p)
        if pass_b:
            signals.append("策略B: 盤整帶量突破")
            strat_details["B"] = info_b

    # 策略 C
    if enable_strat_c:
        p = strat_c_params or {}
        pass_c, info_c = strategy_c_pullback_support(df, **p)
        if pass_c:
            signals.append("策略C: 量縮回測轉強")
            strat_details["C"] = info_c

    if signals:
        return {
            "symbol": symbol,
            "name": name if name else symbol,
            "industry": industry,
            "close": round(float(last_row["Close"]), 2),
            "change_pct": round(float(last_row.get("Change_Pct", 0.0)), 2),
            "volume_lots": round(float(last_row["Volume"]), 1),
            "vol_ma5": round(float(last_row.get("Vol_MA5", 0.0)), 1),
            "ma20": round(float(last_row.get("MA20", 0.0)), 2),
            "ma60": round(float(last_row.get("MA60", 0.0)), 2),
            "signals": signals,
            "signal_count": len(signals),
            "details": strat_details,
            "last_date": str(df.index[-1].strftime("%Y-%m-%d"))
        }

    return None
