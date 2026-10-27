import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

def filter_liquidity(
    df: pd.DataFrame, 
    min_vol_ma5: float = 800.0, 
    min_price: float = 15.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    基礎流動性過濾（必要條件）:
    1. 5 日平均成交量 大於 min_vol_ma5 (張)
    2. 當日收盤價 大於 min_price (元)
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


def strategy_neckline_pullback(
    df: pd.DataFrame,
    lookback_window: int = 35,
    min_breakout_ago: int = 2,
    max_breakout_ago: int = 12,
    min_breakout_gain: float = 1.0,
    vol_breakout_mult: float = 1.15,
    pullback_tolerance_min: float = -3.0,
    pullback_tolerance_max: float = 3.5,
    vol_shrink_pct: float = 75.0,
    max_pullback_ago_days: int = 3,
    max_today_neck_pct: float = 3.0,
    today_vol_mult_max: float = 1.8,
    require_red_or_up: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    【核心策略：突破頸線量縮回測】(及時黃金買點版)
    1. 前期頸線峰位 (Neckline)：在過去第 10 至 40 日區間尋找最高價頸線峰位 Neckline_Price。
    2. 帶量突破 (Breakout)：過去 2 至 12 日內，曾帶量（>= 1.15x 5MA量）突破頸線（High > Neckline + 1%）。
    3. 量縮回測 (Low-Volume Retest)：
       - 突破後回測低點落在頸線附近（-3.0% 至 +3.5%）。
       - 回測最低點必須發生在「近 3 個交易日內」（時效性確認，排除 1 週前早已回測完畢、現已噴出的標的）。
       - 回測期間成交量顯著萎縮（<= 突破日 75% 或 < 5MA量）。
    4. 支撐確立與即時買點 (Rebound / Entry Point)：
       - 今日收盤價穩守於頸線之上（>= Neckline * 0.985）。
       - 今日收盤價距離頸線幅度 <= +3.0%（確保買在頸線甜蜜點，而非已漲 5%~10% 後追高）。
       - 今日成交量未失控爆量（<= 1.8x 5MA均量）。
       - 今日收紅K或高於昨日收盤，確認回測有撐轉強。
    """
    if df is None or len(df) < lookback_window + 15:
        return False, {"reason": "歷史資料長度不足以檢驗型態"}

    n = len(df)
    today = df.iloc[-1]
    today_close = float(today["Close"])
    today_open = float(today["Open"])
    today_vol = float(today["Volume"])
    today_vol_ma5 = float(today.get("Vol_MA5", 0))
    yest_close = float(df.iloc[-2]["Close"]) if len(df) > 1 else today_close

    # 若今日成交量過度爆量（> today_vol_mult_max 倍 5MA），代表當日已大幅爆量衝高，非量縮回測進場點
    if today_vol_ma5 > 0 and (today_vol / today_vol_ma5) > today_vol_mult_max:
        return False, {"reason": f"今日爆量 ({today_vol/today_vol_ma5:.2f}倍 > {today_vol_mult_max:.1f}倍)，非量縮回測甜蜜點"}

    best_match = None

    # 搜尋突破日 (從最近往前搜尋)
    for b_ago in range(min_breakout_ago, max_breakout_ago + 1):
        b_idx = n - b_ago - 1
        if b_idx < lookback_window:
            continue

        b_row = df.iloc[b_idx]
        b_high = float(b_row["High"])
        b_vol = float(b_row["Volume"])
        b_vol_ma5 = float(b_row.get("Vol_MA5", 0))

        # 前期觀察區間: 突破日之前的 lookback_window 天
        pre_start = max(0, b_idx - lookback_window)
        pre_end = b_idx - 2  # 避開突破前夕
        if pre_end <= pre_start:
            continue

        pre_df = df.iloc[pre_start : pre_end]
        neckline_price = float(pre_df["High"].max())
        neck_idx = pre_df["High"].idxmax()
        neck_date = str(neck_idx)[:10]

        # 1. 突破條件檢驗
        breakout_gain = ((b_high - neckline_price) / neckline_price) * 100.0
        vol_ratio = (b_vol / b_vol_ma5) if b_vol_ma5 > 0 else 1.0

        if b_high > neckline_price and breakout_gain >= min_breakout_gain and vol_ratio >= vol_breakout_mult:
            # 找到有效突破日！檢驗後續拉回走勢
            after_df = df.iloc[b_idx + 1 :]
            after_valid_vol = after_df[after_df["Volume"] > 0]

            pullback_min_low = float(after_df["Low"].min())
            pullback_low_idx = after_df["Low"].idxmin()
            pullback_low_date = str(pullback_low_idx)[:10]
            pullback_min_vol = float(after_valid_vol["Volume"].min()) if not after_valid_vol.empty else float(after_df["Volume"].min())

            # 關鍵時效檢驗 1: 回測最低點必須發生在近 max_pullback_ago_days 日內
            days_since_pullback = len(after_df) - after_df.index.get_loc(pullback_low_idx) - 1
            if days_since_pullback > max_pullback_ago_days:
                continue  # 回測已是太久以前的事（例如精誠已是7天前），跳過

            # 關鍵時效檢驗 2: 今日收盤價距離頸線不能超過 max_today_neck_pct (例如 <= +3.0%)
            dist_today_neck = ((today_close - neckline_price) / neckline_price) * 100.0
            if dist_today_neck > max_today_neck_pct:
                continue  # 今日已遠離頸線噴出，排除追價

            # 2. 回測幅度檢驗 (回測低點距離頸線的幅度)
            dist_to_neck = ((pullback_min_low - neckline_price) / neckline_price) * 100.0
            is_near_neck = (pullback_tolerance_min <= dist_to_neck <= pullback_tolerance_max)

            # 3. 量縮檢驗 (回測期間成交量萎縮，無沉重賣壓)
            is_vol_shrunk = (pullback_min_vol <= b_vol * (vol_shrink_pct / 100.0)) or (today_vol <= today_vol_ma5 * 1.2)

            # 4. 今日守穩頸線支撐
            is_supported = today_close >= neckline_price * 0.985

            # 5. 今日轉強確認
            is_turning = (today_close >= today_open) or (today_close >= yest_close) if require_red_or_up else True

            if is_near_neck and is_vol_shrunk and is_supported and is_turning:
                details = {
                    "neckline_price": round(neckline_price, 2),
                    "neckline_date": neck_date,
                    "breakout_date": str(b_row.name)[:10],
                    "breakout_high": round(b_high, 2),
                    "breakout_vol": round(b_vol, 0),
                    "breakout_vol_ratio": round(vol_ratio, 2),
                    "pullback_min_low": round(pullback_min_low, 2),
                    "pullback_low_date": pullback_low_date,
                    "days_since_pullback": days_since_pullback,
                    "dist_to_neck_pct": round(dist_to_neck, 2),
                    "dist_today_neck_pct": round(dist_today_neck, 2),
                    "pullback_min_vol": round(pullback_min_vol, 0),
                    "vol_shrink_ratio": round((pullback_min_vol / b_vol) * 100, 1),
                    "today_close": round(today_close, 2),
                    "is_supported": is_supported,
                    "is_turning": is_turning
                }
                best_match = details
                break

    if best_match:
        return True, best_match
    return False, {"reason": "未完全滿足突破頸線與及時量縮回測條件"}


def check_market_filter(
    market_df: pd.DataFrame,
    mode: str = "any"  # "any" (季線之上 或 5MA向上), "both" (季線之上 且 5MA向上)
) -> Tuple[bool, Dict[str, Any]]:
    """
    大盤環境濾網：加權指數位於季線之上或短期 5MA 向上
    """
    if market_df is None or len(market_df) < 65:
        return True, {"status": "無大盤資料，預設放行"}

    today = market_df.iloc[-1]
    yest = market_df.iloc[-2]
    close = float(today["Close"])
    ma5_today = float(today.get("MA5", 0.0))
    ma5_yest = float(yest.get("MA5", 0.0))
    ma60_today = float(today.get("MA60", 0.0))

    above_ma60 = close >= ma60_today
    ma5_up = ma5_today >= ma5_yest

    if mode == "both":
        is_safe = above_ma60 and ma5_up
    else:
        is_safe = above_ma60 or ma5_up

    details = {
        "market_close": round(close, 2),
        "ma60": round(ma60_today, 2),
        "above_ma60": above_ma60,
        "ma5_today": round(ma5_today, 2),
        "ma5_yest": round(ma5_yest, 2),
        "ma5_up": ma5_up,
        "is_safe": is_safe
    }
    return is_safe, details


def scan_single_stock(
    symbol: str, 
    df: pd.DataFrame, 
    name: str = "", 
    industry: str = "",
    min_vol_ma5: float = 800.0,
    min_price: float = 15.0,
    strat_params: Optional[Dict[str, Any]] = None,
    market_df: Optional[pd.DataFrame] = None,
    require_market_filter: bool = False,
    market_filter_mode: str = "any",
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    對單檔股票執行篩選：流動性檢查 + 突破頸線量縮回測型態檢驗
    """
    if df is None or len(df) < 50:
        return None

    # 0. 大盤環境濾網
    if require_market_filter and market_df is not None:
        is_safe, _ = check_market_filter(market_df, mode=market_filter_mode)
        if not is_safe:
            return None

    # 1. 流動性過濾
    liq_pass, _ = filter_liquidity(df, min_vol_ma5=min_vol_ma5, min_price=min_price)
    if not liq_pass:
        return None

    # 2. 突破頸線量縮回測策略檢驗
    params = strat_params or {}
    matched, details = strategy_neckline_pullback(df, **params)
    if not matched:
        return None

    last_row = df.iloc[-1]
    return {
        "symbol": symbol,
        "name": name,
        "industry": industry,
        "close": float(last_row["Close"]),
        "change_pct": float(last_row.get("Change_Pct", 0.0)),
        "volume_lots": float(last_row["Volume"]),
        "vol_ma5": float(last_row.get("Vol_MA5", 0.0)),
        "signals": ["突破頸線量縮回測"],
        "details": details,
        "last_date": str(last_row.name)[:10]
    }
