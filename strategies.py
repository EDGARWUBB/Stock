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
    vol_multiplier: float = 1.5,
    max_stop_loss_pct: float = 7.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    策略 A：帶量破底翻（假跌破真反轉）
    - 前期低點定義：計算過去第 5 天到第 25 天的區間最低價（排除最近 3 天，稱為 Prev_Low）。
    - 破底動作：最近 3 個交易日內，曾出現最低價小於 Prev_Low。
    - 翻揚確認：今日收盤價重新站回 Prev_Low 之上，且今日為紅 K（收盤價 大於 開盤價）。
    - 量能確認：今日成交量 大於 5 日均量的 1.5 倍（確保有買盤實質承接，排除無量反彈）。
    - 停損空間過濾（控制風險報酬比）：今日收盤價距離近 3 日破底最低價的幅度需小於 7%
      公式：(Close - Min_Low_3D) / Close < 0.07，避免當日過度拉抬導致追價風險過高。
    """
    if df is None or len(df) < lookback_window + 5:
        return False, {"reason": f"資料天數不足 {lookback_window + 5} 天"}

    # 取過去區間（排除最近 exclude_recent 天）
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

    # 停損空間過濾：(Close - Min_Low_3D) / Close < 0.07
    stop_loss_pct = ((today_close - recent_min_low) / today_close * 100.0) if today_close > 0 else 0.0
    pass_stop_loss = ((today_close - recent_min_low) / today_close) < (max_stop_loss_pct / 100.0)

    details = {
        "prev_low": round(prev_low, 2),
        "recent_min_low": round(recent_min_low, 2),
        "today_close": round(today_close, 2),
        "today_open": round(today_open, 2),
        "today_vol": round(today_vol, 1),
        "today_vol_ma5": round(today_vol_ma5, 1),
        "vol_ratio": round(vol_ratio, 2),
        "stop_loss_pct": round(stop_loss_pct, 2),
        "had_breakdown": had_breakdown,
        "reclaimed": reclaimed,
        "is_red": is_red,
        "vol_confirmed": vol_confirmed,
        "pass_stop_loss": pass_stop_loss
    }

    if had_breakdown and reclaimed and is_red and vol_confirmed and pass_stop_loss:
        return True, details
    return False, details


def strategy_a2_reversal_pullback(
    df: pd.DataFrame,
    lookback_trigger_days: int = 5,
    max_distance_pct: float = 3.0,
    lookback_window: int = 25,
    exclude_recent: int = 3,
    vol_multiplier: float = 1.5
) -> Tuple[bool, Dict[str, Any]]:
    """
    子模式「策略 A-2：破底翻後量縮回測買點」：
    - 過去 3 至 5 日內曾觸發「帶量破底翻」
    - 且今日股價拉回靠近 Prev_Low（距離在 0% 至 3% 之間，公式：0 <= (Close - Prev_Low) / Prev_Low <= 0.03）
    - 今日成交量小於 5 日均量（量縮籌碼安定，方便直接捕捉二次回測買點）
    """
    if df is None or len(df) < lookback_window + 10:
        return False, {"reason": "資料天數不足以檢驗回測買點"}

    today = df.iloc[-1]
    today_close = float(today["Close"])
    today_vol = float(today["Volume"])
    today_vol_ma5 = float(today.get("Vol_MA5", 0.0))

    # 1. 檢驗過去 3 至 5 日內是否曾觸發「帶量破底翻」
    found_reversal = False
    reversal_day_idx = None
    prev_low = None

    for i in range(1, lookback_trigger_days + 1):
        sub_df = df.iloc[:-i]
        triggered, info = strategy_a_breakdown_reversal(
            sub_df, 
            lookback_window=lookback_window,
            exclude_recent=exclude_recent,
            vol_multiplier=vol_multiplier,
            max_stop_loss_pct=12.0 # 允許觸發當天空間
        )
        if triggered:
            found_reversal = True
            reversal_day_idx = i
            prev_low = info["prev_low"]
            break

    if not found_reversal or prev_low is None:
        return False, {"reason": f"過去 {lookback_trigger_days} 日內未曾觸發帶量破底翻"}

    # 2. 今日股價拉回靠近 Prev_Low（距離在 0% 至 3% 之間）
    distance_pct = ((today_close - prev_low) / prev_low) * 100.0 if prev_low > 0 else 999.0
    near_support = (0.0 <= distance_pct <= max_distance_pct)

    # 3. 今日成交量小於 5 日均量
    vol_shrunk = today_vol < today_vol_ma5

    details = {
        "prev_low": round(prev_low, 2),
        "today_close": round(today_close, 2),
        "distance_pct": round(distance_pct, 2),
        "today_vol": round(today_vol, 1),
        "today_vol_ma5": round(today_vol_ma5, 1),
        "reversal_ago_days": reversal_day_idx,
        "found_reversal": found_reversal,
        "near_support": near_support,
        "vol_shrunk": vol_shrunk
    }

    if found_reversal and near_support and vol_shrunk:
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


def strategy_d_double_bottom(
    df: pd.DataFrame,
    l1_start_day: int = 30,
    l1_end_day: int = 16,
    l2_start_day: int = 10,
    l2_end_day: int = 2,
    min_interval_days: int = 7,
    min_neckline_gain_pct: float = 6.0,
    min_l2_ratio: float = -0.015,
    max_l2_ratio: float = 0.035,
    d1_max_distance_pct: float = 4.0,
    d2_vol_multiplier: float = 1.8,
    enable_d1: bool = True,
    enable_d2: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    策略 D：標準 W 底（雙重底）頸線突破與右腳轉強
    時間區間切片法拆解過去 25 至 35 個交易日：
    1. 左腳 (L1)：過去第 16 天至第 30 天區間最低價 L1_Price 與日期 L1_Date
    2. 中間頸線峰位 (Neckline)：L1_Date 之後至過去第 5 天之間最高價，且空間幅度 >= 6%
    3. 右腳 (L2)：過去第 2 天至第 10 天區間最低價 L2_Price，兩腳間隔 >= 7 天，L2 與 L1 差距在 -1.5% 至 +3.5% 之間
    4. 量能結構：L2 成交量 < L1 成交量，或 L2 成交量 < 20 日均量 (量縮無沉重賣壓)
    5. 觸發訊號：
       - 模式 D-1 (右腳潛伏轉折)：今日收盤站上 5MA、收紅 K、距離 L2 幅度在 4% 內
       - 模式 D-2 (帶量突破頸線)：今日收盤突破 Neckline、收紅 K、成交量 >= 5 日均量 1.8 倍
    """
    if df is None or len(df) < max(l1_start_day + 10, 40):
        return False, {"reason": "資料天數不足以切片檢驗 W 底型態"}

    n_len = len(df)

    # 1. 左腳 (L1)：過去第 16 天至第 30 天 (iloc[-l1_start_day : -(l1_end_day - 1)])
    # 注意: 若 l1_end_day=16, 則切片為 iloc[-30 : -15], 包含第 30 天前到第 16 天前
    sub_l1 = df.iloc[-l1_start_day : -(l1_end_day - 1)]
    if sub_l1.empty:
        return False, {"reason": "L1 觀察區間為空"}

    l1_idx = sub_l1["Low"].idxmin()
    l1_price = float(sub_l1.loc[l1_idx, "Low"])
    l1_vol = float(sub_l1.loc[l1_idx, "Volume"])
    l1_pos = df.index.get_loc(l1_idx)
    l1_date = sub_l1.loc[l1_idx].name

    # 3. 右腳 (L2)：過去第 2 天至第 10 天 (iloc[-l2_start_day : -(l2_end_day - 1)])
    # 若 l2_end_day=2, 則切片為 iloc[-10 : -1], 包含第 10 天前至第 2 天前
    sub_l2 = df.iloc[-l2_start_day : -(l2_end_day - 1)]
    if sub_l2.empty:
        return False, {"reason": "L2 觀察區間為空"}

    l2_idx = sub_l2["Low"].idxmin()
    l2_price = float(sub_l2.loc[l2_idx, "Low"])
    l2_vol = float(sub_l2.loc[l2_idx, "Volume"])
    l2_pos = df.index.get_loc(l2_idx)
    l2_date = sub_l2.loc[l2_idx].name

    # 兩腳時間間隔過濾：必須相隔至少 min_interval_days 天 (預設 7 天)
    interval_days = l2_pos - l1_pos
    if interval_days < min_interval_days:
        return False, {"reason": f"兩腳間隔 {interval_days} 天未達 {min_interval_days} 天"}

    # 左低右高 / 雙底支撐確認：L2 與 L1 差距在 min_l2_ratio (-1.5%) 至 max_l2_ratio (+3.5%)
    l2_diff_ratio = (l2_price - l1_price) / l1_price if l1_price > 0 else 999.0
    if not (min_l2_ratio <= l2_diff_ratio <= max_l2_ratio):
        return False, {"reason": f"兩腳價差幅度 {l2_diff_ratio*100:.2f}% 不在容許範圍內"}

    # 2. 中間頸線峰位 (Neckline Peak)：在 L1_Date 之後、至過去第 5 天之間的最高價
    # 切片範圍: 從 l1_pos + 1 至 n_len - 4
    neck_start = l1_pos + 1
    neck_end = n_len - 4
    if neck_start >= neck_end:
        return False, {"reason": "頸線峰位觀察區間不足"}

    sub_neckline = df.iloc[neck_start:neck_end]
    neck_idx = sub_neckline["High"].idxmax()
    neckline_price = float(sub_neckline.loc[neck_idx, "High"])
    neck_pos = df.index.get_loc(neck_idx)
    neckline_date = sub_neckline.loc[neck_idx].name

    # 確保右腳在頸線之後
    if l2_pos <= neck_pos:
        return False, {"reason": "右腳必須位於中間頸線峰位之後"}

    # 空間幅度過濾：Neckline_Price 必須比 L1_Price 高出至少 6%
    neckline_gain_pct = ((neckline_price - l1_price) / l1_price) * 100.0 if l1_price > 0 else 0.0
    if neckline_gain_pct < min_neckline_gain_pct:
        return False, {"reason": f"頸線反彈幅度 {neckline_gain_pct:.1f}% 未達 {min_neckline_gain_pct}% 門檻"}

    # 4. 量能結構過濾：右腳最低點成交量 < 左腳成交量，或 < 20 日均量
    l2_row = df.loc[l2_idx]
    l2_vol_ma20 = float(l2_row.get("Vol_MA20", 0.0))
    vol_ok = (l2_vol < l1_vol) or (l2_vol_ma20 > 0 and l2_vol < l2_vol_ma20)
    if not vol_ok:
        return False, {"reason": f"右腳成交量 ({l2_vol:.1f}) 未量縮於左腳 ({l1_vol:.1f}) 或 20MA均量"}

    # W 底等幅測距目標價
    min_bottom = min(l1_price, l2_price)
    target_price = round(neckline_price + (neckline_price - min_bottom), 2)

    # 5. 今日狀況與觸發訊號檢驗
    today = df.iloc[-1]
    today_close = float(today["Close"])
    today_open = float(today["Open"])
    today_vol = float(today["Volume"])
    today_ma5 = float(today.get("MA5", 0.0))
    today_vol_ma5 = float(today.get("Vol_MA5", 0.0))
    is_red = today_close > today_open

    triggered_sub_modes = []

    # 模式 D-1（右腳潛伏轉折買點）：站上 5MA、收紅 K、距離 L2 幅度在 4% 內、尚未大幅突破頸線
    dist_l2_pct = ((today_close - l2_price) / l2_price) * 100.0 if l2_price > 0 else 999.0
    is_d1_triggered = (
        (today_close >= l2_price) and
        (today_close > today_ma5) and
        is_red and
        (0.0 <= dist_l2_pct <= d1_max_distance_pct) and
        (today_close <= neckline_price)
    )
    if enable_d1 and is_d1_triggered:
        triggered_sub_modes.append("模式D-1: 右腳轉折")

    # 模式 D-2（標準帶量突破頸線買點）：收盤正式突破 Neckline、收紅 K、成交量 >= 5 日均量 1.8 倍
    vol_ratio = (today_vol / today_vol_ma5) if today_vol_ma5 > 0 else 0.0
    is_d2_triggered = (
        (today_close > neckline_price) and
        is_red and
        (vol_ratio >= d2_vol_multiplier)
    )
    if enable_d2 and is_d2_triggered:
        triggered_sub_modes.append("模式D-2: 帶量突破頸線")

    details = {
        "l1_price": round(l1_price, 2),
        "l1_date": str(l1_date.strftime("%Y-%m-%d")) if hasattr(l1_date, "strftime") else str(l1_date)[:10],
        "l1_vol": round(l1_vol, 1),
        "l2_price": round(l2_price, 2),
        "l2_date": str(l2_date.strftime("%Y-%m-%d")) if hasattr(l2_date, "strftime") else str(l2_date)[:10],
        "l2_vol": round(l2_vol, 1),
        "neckline_price": round(neckline_price, 2),
        "neckline_date": str(neckline_date.strftime("%Y-%m-%d")) if hasattr(neckline_date, "strftime") else str(neckline_date)[:10],
        "target_price": target_price,
        "neckline_gain_pct": round(neckline_gain_pct, 2),
        "l2_diff_pct": round(l2_diff_ratio * 100.0, 2),
        "interval_days": interval_days,
        "vol_ok": vol_ok,
        "sub_modes": triggered_sub_modes,
        "today_close": round(today_close, 2),
        "today_vol": round(today_vol, 1),
        "vol_ratio": round(vol_ratio, 2)
    }

    if triggered_sub_modes:
        return True, details

    return False, details


def check_market_filter(
    market_df: pd.DataFrame,
    mode: str = "any"  # "any" (季線之上 或 5MA向上), "both" (季線之上 且 5MA向上), "above_ma60", "ma5_up"
) -> Tuple[bool, Dict[str, Any]]:
    """
    大盤環境濾網：
    - 大盤指數 (如 ^TWII) 位於季線 (60MA) 之上
    - 或大盤 5MA 斜率向上 (今日 5MA > 昨日 5MA)
    條件滿足時才啟動做多策略，大幅降低在系統性崩跌時接刀的風險。
    """
    if market_df is None or len(market_df) < 65:
        return True, {"status": "無法取得完整大盤季線資料，預設放行"}

    today = market_df.iloc[-1]
    yest = market_df.iloc[-2]

    close = float(today["Close"])
    ma5_today = float(today.get("MA5", 0.0))
    ma5_yest = float(yest.get("MA5", 0.0))
    ma20_today = float(today.get("MA20", 0.0))
    ma60_today = float(today.get("MA60", 0.0))
    change_pct = float(today.get("Change_Pct", 0.0))

    above_ma60 = close >= ma60_today
    ma5_up = ma5_today >= ma5_yest

    details = {
        "index_close": round(close, 2),
        "index_change_pct": round(change_pct, 2),
        "ma5": round(ma5_today, 2),
        "ma20": round(ma20_today, 2),
        "ma60": round(ma60_today, 2),
        "above_ma60": above_ma60,
        "ma5_up": ma5_up,
        "bias_ma60_pct": round((close - ma60_today) / ma60_today * 100.0, 2) if ma60_today > 0 else 0.0
    }

    if mode == "both":
        is_safe = above_ma60 and ma5_up
        details["condition_text"] = "季線之上 且 5MA 向上"
    elif mode == "above_ma60":
        is_safe = above_ma60
        details["condition_text"] = "季線之上"
    elif mode == "ma5_up":
        is_safe = ma5_up
        details["condition_text"] = "5MA 向上"
    else:  # "any"
        is_safe = above_ma60 or ma5_up
        details["condition_text"] = "季線之上 或 5MA 向上"

    details["is_safe"] = is_safe
    return is_safe, details


def scan_single_stock(
    symbol: str, 
    df: pd.DataFrame, 
    name: str = "", 
    industry: str = "",
    min_vol_ma5: float = 1000.0,
    min_price: float = 15.0,
    enable_strat_a: bool = True,
    enable_strat_a2: bool = True,
    enable_strat_b: bool = True,
    enable_strat_c: bool = True,
    enable_strat_d: bool = True,
    strat_a_params: Optional[Dict] = None,
    strat_a2_params: Optional[Dict] = None,
    strat_b_params: Optional[Dict] = None,
    strat_c_params: Optional[Dict] = None,
    strat_d_params: Optional[Dict] = None,
    market_df: Optional[pd.DataFrame] = None,
    require_market_filter: bool = False,
    market_filter_mode: str = "any"
) -> Optional[Dict[str, Any]]:
    """
    對單檔股票執行基礎流動性過濾與各項策略檢驗
    """
    if df is None or len(df) < 60:
        return None

    # 0. 大盤環境濾網（選用）
    if require_market_filter and market_df is not None:
        is_market_safe, _ = check_market_filter(market_df, mode=market_filter_mode)
        if not is_market_safe:
            return None

    # 1. 基礎流動性過濾（必要條件）
    liq_pass, liq_info = filter_liquidity(df, min_vol_ma5=min_vol_ma5, min_price=min_price)
    if not liq_pass:
        return None

    last_row = df.iloc[-1]
    signals = []
    strat_details = {}

    # 策略 A: 帶量破底翻
    if enable_strat_a:
        p = strat_a_params or {}
        pass_a, info_a = strategy_a_breakdown_reversal(df, **p)
        if pass_a:
            signals.append("策略A: 帶量破底翻")
            strat_details["A"] = info_a

    # 策略 A-2: 破底翻後量縮回測買點
    if enable_strat_a2:
        p2 = strat_a2_params or {}
        pass_a2, info_a2 = strategy_a2_reversal_pullback(df, **p2)
        if pass_a2:
            signals.append("策略A-2: 破底翻量縮回測")
            strat_details["A2"] = info_a2

    # 策略 B: 盤整帶量突破
    if enable_strat_b:
        p = strat_b_params or {}
        pass_b, info_b = strategy_b_consolidation_breakout(df, **p)
        if pass_b:
            signals.append("策略B: 盤整帶量突破")
            strat_details["B"] = info_b

    # 策略 C: 強勢股量縮回測支撐
    if enable_strat_c:
        p = strat_c_params or {}
        pass_c, info_c = strategy_c_pullback_support(df, **p)
        if pass_c:
            signals.append("策略C: 量縮回測轉強")
            strat_details["C"] = info_c

    # 策略 D: 標準 W 底（雙重底）
    if enable_strat_d:
        pd_ = strat_d_params or {}
        pass_d, info_d = strategy_d_double_bottom(df, **pd_)
        if pass_d:
            sub_modes = info_d.get("sub_modes", ["W底型態"])
            for sm in sub_modes:
                signals.append(f"策略D: {sm}")
            strat_details["D"] = info_d

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
