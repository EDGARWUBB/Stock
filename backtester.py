import os
import sys

# Ensure UTF-8 output encoding
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import pandas as pd
import numpy as np
import json
import datetime
from data_loader import load_stock_metadata, fetch_stock_data, fetch_market_index, calculate_technical_indicators
from strategies import filter_liquidity, strategy_neckline_pullback, check_market_filter

def run_backtest(period: str = "1y", holding_periods = [3, 5, 10, 20]):
    print("=" * 75)
    print("【突破頸線量縮回測策略】歷史回測檢驗 (過去 1 年台股行情真實數據)")
    print("=" * 75)

    # 1. 載入大盤資料 (^TWII)
    print(">> 1. 載入加權指數 (^TWII) 作為大盤環境多空濾網...")
    market_df = fetch_market_index("^TWII", period=period)
    if market_df is not None:
        print(f"   ✔ 加權指數資料抓取完成: 共 {len(market_df)} 個交易日")
    else:
        print("   ⚠ 無法取得大盤指數，預設全放行")

    # 2. 載入股票池 (54 檔精選權值與熱門活躍股，含微星 2377 等)
    meta = load_stock_metadata("stocks_default.csv")
    symbols = meta["symbol"].tolist()
    names = dict(zip(meta["symbol"], meta["name"]))
    print(f">> 2. 載入股票池: 共 {len(symbols)} 檔代表性標的...")

    trade_records = []
    max_hold = max(holding_periods)

    print(">> 3. 開始逐日滑動窗口檢驗「突破頸線量縮回測」訊號...")

    for s_idx, symbol in enumerate(symbols):
        stock_name = names.get(symbol, symbol)
        try:
            df = fetch_stock_data(symbol, period=period)
        except Exception:
            continue

        if df is None or len(df) < 55:
            continue

        df = calculate_technical_indicators(df)
        n_bars = len(df)

        # 逐日進行回溯 (確保至少有 45 天數據計算頸線，最後留 max_hold 天結算)
        for t in range(45, n_bars - max_hold):
            sub_df = df.iloc[: t + 1]
            trade_date = sub_df.index[-1].strftime("%Y-%m-%d")
            entry_price = float(sub_df["Close"].iloc[-1])

            # 基礎流動性過濾 (5MA均量 >= 600張, 收盤價 >= 15元)
            liq_pass, _ = filter_liquidity(sub_df, min_vol_ma5=600.0, min_price=15.0)
            if not liq_pass:
                continue

            # 大盤環境檢驗
            market_safe = True
            if market_df is not None:
                m_sub = market_df[market_df.index <= sub_df.index[-1]]
                if len(m_sub) >= 60:
                    market_safe, _ = check_market_filter(m_sub, mode="any")

            # 檢驗「突破頸線量縮回測」策略
            matched, details = strategy_neckline_pullback(
                sub_df,
                lookback_window=35,
                min_breakout_ago=2,
                max_breakout_ago=12,
                min_breakout_gain=1.0,
                vol_breakout_mult=1.15,
                pullback_tolerance_min=-3.0,
                pullback_tolerance_max=4.5,
                vol_shrink_pct=75.0,
                require_red_or_up=True
            )

            if matched:
                record = {
                    "symbol": symbol,
                    "name": stock_name,
                    "date": trade_date,
                    "entry_price": entry_price,
                    "neckline_price": details.get("neckline_price", 0.0),
                    "dist_to_neck_pct": details.get("dist_to_neck_pct", 0.0),
                    "market_safe": market_safe
                }

                # 統計持股 N 日後的報酬與勝負
                for h in holding_periods:
                    exit_price = float(df["Close"].iloc[t + h])
                    ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
                    record[f"ret_{h}d"] = round(ret_pct, 2)
                    record[f"win_{h}d"] = ret_pct > 0.0

                trade_records.append(record)

    trades_df = pd.DataFrame(trade_records)

    # 避免同一檔股票連續幾天觸發重複計入，進行 5 日去重 (若 5 日內同檔觸發僅計入首次進場)
    if not trades_df.empty:
        trades_df["date_dt"] = pd.to_datetime(trades_df["date"])
        trades_df.sort_values(by=["symbol", "date_dt"], inplace=True)
        filtered_records = []
        for sym, group in trades_df.groupby("symbol"):
            last_date = None
            for _, row in group.iterrows():
                if last_date is None or (row["date_dt"] - last_date).days > 5:
                    filtered_records.append(row)
                    last_date = row["date_dt"]
        trades_df = pd.DataFrame(filtered_records).drop(columns=["date_dt"])

    total_trades = len(trades_df)
    print(f"\n✔ 回測掃描完成！共觸發 {total_trades} 次獨立進場訊號。")

    if total_trades == 0:
        print("未產生交易紀錄。")
        return {}

    # 4. 統計分析各持股天期之績效
    summary_results = {
        "strategy_name": "突破頸線量縮回測",
        "total_trades": total_trades,
        "holding_metrics": {}
    }

    print("\n" + "=" * 75)
    print("【突破頸線量縮回測】持股週期績效彙整表")
    print("=" * 75)
    print(f"{'持股天數':<8} | {'勝率(%)':<9} | {'平均報酬(%)':<11} | {'盈虧比':<8} | {'最大獲利(%)':<11} | {'最大虧損(%)':<11}")
    print("-" * 75)

    for h in holding_periods:
        ret_col = f"ret_{h}d"
        win_col = f"win_{h}d"

        rets = trades_df[ret_col]
        wins = trades_df[win_col]

        win_rate = (wins.sum() / len(wins)) * 100.0
        avg_ret = rets.mean()
        
        pos_rets = rets[rets > 0]
        neg_rets = rets[rets < 0]
        avg_win = pos_rets.mean() if len(pos_rets) > 0 else 0.0
        avg_loss = abs(neg_rets.mean()) if len(neg_rets) > 0 else 1.0
        pl_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        max_gain = rets.max()
        max_loss = rets.min()

        summary_results["holding_metrics"][f"{h}d"] = {
            "win_rate": round(win_rate, 1),
            "avg_ret": round(avg_ret, 2),
            "pl_ratio": round(pl_ratio, 2),
            "max_gain": round(max_gain, 1),
            "max_loss": round(max_loss, 1)
        }

        print(f"持有 {h:2d} 天  | {win_rate:6.1f}%   | {avg_ret:+8.2f}%    | {pl_ratio:6.2f}   | {max_gain:+8.1f}%    | {max_loss:+8.1f}%")

    # 5. 大盤環境濾網效益分析 (持有 10 天為基準)
    mkt_safe_trades = trades_df[trades_df["market_safe"] == True]
    if len(mkt_safe_trades) > 0:
        safe_wins = mkt_safe_trades["win_10d"]
        safe_rets = mkt_safe_trades["ret_10d"]
        safe_win_rate = (safe_wins.sum() / len(safe_wins)) * 100.0
        safe_avg_ret = safe_rets.mean()
        summary_results["market_filter_impact"] = {
            "all_win_rate_10d": round(summary_results["holding_metrics"]["10d"]["win_rate"], 1),
            "safe_win_rate_10d": round(safe_win_rate, 1),
            "win_rate_boost": round(safe_win_rate - summary_results["holding_metrics"]["10d"]["win_rate"], 1),
            "safe_avg_ret_10d": round(safe_avg_ret, 2)
        }

        print("\n" + "=" * 75)
        print("【大盤環境濾網效益 (以持有 10 天為例)】")
        print("=" * 75)
        print(f"- 全部訊號 10D 勝率: {summary_results['holding_metrics']['10d']['win_rate']}% | 平均報酬: {summary_results['holding_metrics']['10d']['avg_ret']:+.2f}%")
        print(f"- 濾網安全 10D 勝率: {safe_win_rate:.1f}% | 平均報酬: {safe_avg_ret:+.2f}%")
        print(f"- 勝率提升幅度: +{safe_win_rate - summary_results['holding_metrics']['10d']['win_rate']:.1f}%")

    # 6. 取樣輸出近期經典交易紀錄 (含微星等)
    print("\n" + "=" * 75)
    print("【近期經典觸發範例紀錄】")
    print("=" * 75)
    sample_trades = trades_df.tail(8)
    for _, row in sample_trades.iterrows():
        print(f"📅 {row['date']} | {row['symbol']} {row['name']} | 進場價: {row['entry_price']:.1f}元 | 頸線: {row['neckline_price']:.1f}元 | 5D報酬: {row['ret_5d']:+.2f}% | 10D報酬: {row['ret_10d']:+.2f}%")

    # 儲存回測結果至 JSON
    with open("backtest_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    print("\n✔ 回測結果已儲存至 backtest_results.json！")

    return summary_results

if __name__ == "__main__":
    run_backtest(period="1y")
