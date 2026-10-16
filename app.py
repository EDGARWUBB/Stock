import streamlit as st
import pandas as pd
import numpy as np
import datetime
from data_loader import load_stock_metadata, fetch_stock_data, calculate_technical_indicators
from strategies import (
    filter_liquidity,
    strategy_a_breakdown_reversal,
    strategy_b_consolidation_breakout,
    strategy_c_pullback_support,
    scan_single_stock
)
from chart_plotter import plot_stock_kline

# 設定頁面佈局與標題
st.set_page_config(
    page_title="台股短線技術面選股工具",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂 CSS 增強視覺效果
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #e74c3c;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .strategy-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 5px;
    }
    .badge-a { background-color: #ffeaa7; color: #d63031; border: 1px solid #fdcb6e; }
    .badge-b { background-color: #dfe6e9; color: #0984e3; border: 1px solid #b2bec3; }
    .badge-c { background-color: #e8f8f5; color: #00b894; border: 1px solid #55efc4; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_stock_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """快取股票歷史日K線資料"""
    return fetch_stock_data(symbol, period=period)


def main():
    st.title("📊 台股短線技術面選股工具")
    st.caption("基於量價結構、型態學與均線濾網的自動化技術面即時篩選系統")

    # ============================
    # 側邊欄：參數設定
    # ============================
    st.sidebar.header("⚙️ 篩選參數設定")

    # 1. 股票池設定
    st.sidebar.subheader("1. 股票資料來源")
    stock_source = st.sidebar.radio(
        "股票池來源",
        ["內建熱門與台灣50成分股", "上傳自訂 CSV", "手動輸入代號清單"],
        index=0
    )

    metadata_df = pd.DataFrame()
    if stock_source == "內建熱門與台灣50成分股":
        default_meta = load_stock_metadata("stocks_default.csv")
        sub_category = st.sidebar.selectbox(
            "選擇類別分類",
            ["全部股票 (53檔)", "台灣50核心股", "電子與半導體", "航運與傳產", "金融保險"]
        )
        if sub_category == "台灣50核心股":
            metadata_df = default_meta.head(25)
        elif sub_category == "電子與半導體":
            metadata_df = default_meta[default_meta["industry"].isin(["半導體", "電腦週邊", "電子零組件", "光電業", "通信網路"])]
        elif sub_category == "航運與傳產":
            metadata_df = default_meta[default_meta["industry"].isin(["航運業", "鋼鐵工業", "塑膠工業", "食品工業", "貿易百貨"])]
        elif sub_category == "金融保險":
            metadata_df = default_meta[default_meta["industry"] == "金融保險"]
        else:
            metadata_df = default_meta

    elif stock_source == "上傳自訂 CSV":
        uploaded_file = st.sidebar.file_uploader("上傳 CSV (需包含 symbol 欄位)", type=["csv"])
        if uploaded_file is not None:
            try:
                metadata_df = pd.read_csv(uploaded_file)
                if "symbol" not in metadata_df.columns:
                    st.sidebar.error("CSV 必須包含 'symbol' 欄位（例如 2330.TW）")
                    metadata_df = pd.DataFrame()
                if "name" not in metadata_df.columns:
                    metadata_df["name"] = metadata_df["symbol"]
            except Exception as e:
                st.sidebar.error(f"檔案解析失敗: {e}")
        else:
            st.sidebar.info("請上傳 CSV 檔案，預設暫時使用台灣 50 代表股")
            metadata_df = load_stock_metadata("stocks_default.csv").head(10)

    else:
        custom_input = st.sidebar.text_area(
            "輸入股票代號 (逗號或換行分隔)",
            value="2330.TW, 2317.TW, 2454.TW, 2603.TW, 3035.TW, 2002.TW, 2890.TW"
        )
        symbols = [s.strip().upper() for s in custom_input.replace("\n", ",").split(",") if s.strip()]
        metadata_df = pd.DataFrame({
            "symbol": symbols,
            "name": symbols,
            "industry": ["自訂"] * len(symbols)
        })

    st.sidebar.markdown(f"**當前股票池數量：** `{len(metadata_df)}` 檔")

    # 2. 基礎流動性過濾
    st.sidebar.subheader("2. 基礎流動性過濾（必要）")
    min_vol_ma5 = st.sidebar.slider(
        "5 日均量下限 (張)",
        min_value=300,
        max_value=5000,
        value=1000,
        step=100,
        help="短線交易需避開流動性陷阱，預設大於 1,000 張"
    )
    min_price = st.sidebar.slider(
        "收盤價下限 (元)",
        min_value=5.0,
        max_value=50.0,
        value=15.0,
        step=1.0,
        help="排除雞蛋水餃股，預設大於 15 元"
    )

    # 3. 策略開關與參數設定
    st.sidebar.subheader("3. 短線策略開關與參數")

    # 策略 A
    enable_strat_a = st.sidebar.checkbox("策略 A：帶量破底翻 (假跌破真反轉)", value=True)
    strat_a_params = {}
    if enable_strat_a:
        with st.sidebar.expander("策略 A 細部參數"):
            strat_a_params["lookback_window"] = st.slider("前期低點回溯天數 (第5~N天)", 15, 40, 25)
            strat_a_params["exclude_recent"] = st.slider("排除最近天數", 2, 5, 3)
            strat_a_params["vol_multiplier"] = st.slider("今日爆量倍數 (大於5日均量N倍)", 1.0, 3.0, 1.5, step=0.1)

    # 策略 B
    enable_strat_b = st.sidebar.checkbox("策略 B：區間盤整帶量突破", value=True)
    strat_b_params = {}
    if enable_strat_b:
        with st.sidebar.expander("策略 B 細部參數"):
            strat_b_params["consolidation_days"] = st.slider("盤整觀察天數", 5, 20, 10)
            strat_b_params["max_amplitude_pct"] = st.slider("盤整區間最大振幅 (%)", 4.0, 15.0, 8.0, step=0.5)
            strat_b_params["vol_multiplier"] = st.slider("突破成交量倍數 (大於5日均量N倍)", 1.2, 3.5, 2.0, step=0.1)

    # 策略 C
    enable_strat_c = st.sidebar.checkbox("策略 C：強勢股量縮回測支撐", value=True)
    strat_c_params = {}
    if enable_strat_c:
        with st.sidebar.expander("策略 C 細部參數"):
            strat_c_params["lookback_surge_days"] = st.slider("前期攻擊紅K回溯天數", 10, 25, 15)
            strat_c_params["surge_gain_pct"] = st.slider("攻擊紅K單日漲幅門檻 (%)", 3.0, 8.0, 5.0, step=0.5)
            strat_c_params["surge_vol_mult"] = st.slider("攻擊紅K量能倍數 (大於5日均量N倍)", 1.5, 3.0, 2.0, step=0.1)
            strat_c_params["ma_bias_pct"] = st.slider("回測均線乖離率容許範圍 (±%)", 1.0, 4.0, 2.0, step=0.5)

    # 交易日回溯模擬選擇 (便於回測特定日或檢視歷史訊號)
    st.sidebar.subheader("4. 掃描基準日")
    backtest_offset = st.sidebar.selectbox(
        "選擇掃描基準日",
        [
            ("最新交易日 (即時最新)", 0),
            ("1 個交易日前", 1),
            ("2 個交易日前", 2),
            ("3 個交易日前", 3),
            ("4 個交易日前", 4),
            ("5 個交易日前", 5),
            ("6 個交易日前", 6),
            ("7 個交易日前", 7)
        ],
        format_func=lambda x: x[0]
    )[1]

    run_scan_btn = st.sidebar.button("🚀 開始執行選股掃描", type="primary", use_container_width=True)

    # ============================
    # 執行掃描邏輯
    # ============================
    if "scan_results" not in st.session_state or run_scan_btn:
        with st.spinner("正在抓取歷史行情與計算技術指標，請稍候..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = []
            scanned_count = 0
            liquidity_pass_count = 0
            symbols_list = metadata_df["symbol"].tolist()
            names_dict = dict(zip(metadata_df["symbol"], metadata_df["name"]))
            industry_dict = dict(zip(metadata_df["symbol"], metadata_df.get("industry", [""] * len(metadata_df))))

            total = len(symbols_list)
            for idx, symbol in enumerate(symbols_list):
                status_text.text(f"掃描中 ({idx+1}/{total}): {symbol} {names_dict.get(symbol, '')}")
                progress_bar.progress((idx + 1) / total)

                df = get_cached_stock_data(symbol, period="6mo")
                if df is None or len(df) < 65:
                    continue

                scanned_count += 1

                # 若選擇歷史基準日
                if backtest_offset > 0:
                    if len(df) > backtest_offset + 60:
                        df_target = df.iloc[:-backtest_offset].copy()
                        # 重新確認均線
                        df_target = calculate_technical_indicators(df_target)
                    else:
                        continue
                else:
                    df_target = df

                # 基礎流動性檢查
                liq_pass, _ = filter_liquidity(df_target, min_vol_ma5=min_vol_ma5, min_price=min_price)
                if liq_pass:
                    liquidity_pass_count += 1

                # 策略檢查
                res = scan_single_stock(
                    symbol=symbol,
                    df=df_target,
                    name=names_dict.get(symbol, symbol),
                    industry=industry_dict.get(symbol, "-"),
                    min_vol_ma5=min_vol_ma5,
                    min_price=min_price,
                    enable_strat_a=enable_strat_a,
                    enable_strat_b=enable_strat_b,
                    enable_strat_c=enable_strat_c,
                    strat_a_params=strat_a_params,
                    strat_b_params=strat_b_params,
                    strat_c_params=strat_c_params
                )
                if res:
                    results.append(res)

            status_text.empty()
            progress_bar.empty()

            st.session_state["scan_results"] = results
            st.session_state["scanned_count"] = scanned_count
            st.session_state["liquidity_pass_count"] = liquidity_pass_count
            st.session_state["last_scan_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ============================
    # 主畫面呈現
    # ============================
    results = st.session_state.get("scan_results", [])
    scanned_count = st.session_state.get("scanned_count", 0)
    liquidity_pass_count = st.session_state.get("liquidity_pass_count", 0)
    last_scan_time = st.session_state.get("last_scan_time", "-")

    # 頂部統計指標卡片
    strat_a_hits = sum(1 for r in results if any("策略A" in s for s in r["signals"]))
    strat_b_hits = sum(1 for r in results if any("策略B" in s for s in r["signals"]))
    strat_c_hits = sum(1 for r in results if any("策略C" in s for s in r["signals"]))

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("掃描總檔數", f"{scanned_count} 檔")
    with m2:
        st.metric("通過流動性過濾", f"{liquidity_pass_count} 檔")
    with m3:
        st.metric("🔥 帶量破底翻 (A)", f"{strat_a_hits} 檔")
    with m4:
        st.metric("🚀 盤整突破 (B)", f"{strat_b_hits} 檔")
    with m5:
        st.metric("🎯 回測轉強 (C)", f"{strat_c_hits} 檔")

    st.markdown("---")

    # 篩選結果總覽
    st.subheader(f"📋 策略觸發精選名單 (共 {len(results)} 檔)")

    selected_symbol = None

    if results:
        # 轉換為 DataFrame 呈現
        table_rows = []
        for r in results:
            table_rows.append({
                "股票代號": r["symbol"],
                "名稱": r["name"],
                "產業類別": r["industry"],
                "最新收盤價(元)": f"{r['close']:.2f}",
                "漲跌幅(%)": f"{r['change_pct']:+.2f}%",
                "成交量(張)": f"{r['volume_lots']:,.0f}",
                "5日均量(張)": f"{r['vol_ma5']:,.0f}",
                "觸發訊號標籤": " | ".join(r["signals"]),
                "資料基準日": r["last_date"]
            })

        res_df = pd.DataFrame(table_rows)

        col_tbl, col_dl = st.columns([0.8, 0.2])
        with col_dl:
            csv_data = res_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 匯出選股清單 CSV",
                data=csv_data,
                file_name=f"tw_stock_signals_{datetime.date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.dataframe(
            res_df,
            use_container_width=True,
            hide_index=True
        )

        # 股票選擇器（連動圖表）
        st.markdown("### 🔍 個股深度 K 線與技術面分析")
        stock_options = [f"{r['symbol']} {r['name']} ({' | '.join(r['signals'])})" for r in results]
        selected_option = st.selectbox("選擇要檢視技術線圖的股票：", stock_options, index=0)
        selected_symbol = selected_option.split(" ")[0]

    else:
        st.info("💡 目前設定條件下未篩選出觸發訊號的股票。您可以：\n"
                "1. 在左側側邊欄微調參數（例如調降成交量爆量倍數、微幅放寬振幅百分比）\n"
                "2. 切換「掃描基準日」回溯檢視過去交易日（例如 1~4 個交易日前有航運、中鋼、統一等觸發）\n"
                "3. 下方仍可手動選擇任一股票直接檢視技術走勢圖！")

        # 允許直接檢視任一股票
        all_symbols = metadata_df["symbol"].tolist()
        stock_options = [f"{s} {metadata_df[metadata_df['symbol'] == s]['name'].values[0]}" for s in all_symbols]
        selected_option = st.selectbox("🔍 選擇任一檔股票檢視 K 線走勢：", stock_options, index=0)
        selected_symbol = selected_option.split(" ")[0]

    # ============================
    # 個股詳細圖表呈現
    # ============================
    if selected_symbol:
        st.markdown("---")
        stock_name = metadata_df[metadata_df["symbol"] == selected_symbol]["name"].values[0] if selected_symbol in metadata_df["symbol"].values else selected_symbol

        # 找出該股票觸發的訊號
        matched_result = next((r for r in results if r["symbol"] == selected_symbol), None)
        signals = matched_result["signals"] if matched_result else []

        # 取得完整資料
        full_df = get_cached_stock_data(selected_symbol, period="6mo")

        if full_df is not None and not full_df.empty:
            # 依基準日調整
            if backtest_offset > 0 and len(full_df) > backtest_offset + 60:
                display_df = full_df.iloc[:-backtest_offset].copy()
                display_df = calculate_technical_indicators(display_df)
            else:
                display_df = full_df

            # 繪製 Plotly 圖表
            fig = plot_stock_kline(
                df=display_df,
                symbol=selected_symbol,
                name=stock_name,
                signals=signals,
                days=60
            )
            st.plotly_chart(fig, use_container_width=True)

            # 技術指標數據看板
            latest_k = display_df.iloc[-1]
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("最新收盤價", f"{latest_k['Close']:.2f} 元", f"{latest_k['Change_Pct']:+.2f}%")
            with c2:
                st.metric("當日成交量", f"{latest_k['Volume']:,.0f} 張")
            with c3:
                st.metric("5MA 週線", f"{latest_k['MA5']:.2f} 元")
            with c4:
                st.metric("20MA 月線", f"{latest_k['MA20']:.2f} 元")
            with c5:
                st.metric("60MA 季線", f"{latest_k['MA60']:.2f} 元")

            # 策略詳細觸發條件拆解
            if matched_result and matched_result.get("details"):
                st.markdown("#### 🎯 策略觸發指標條件詳情")
                details = matched_result["details"]

                if "A" in details:
                    with st.expander("📌 策略 A：帶量破底翻 檢驗細節", expanded=True):
                        info = details["A"]
                        st.markdown(f"""
                        - **前期區間低點 (Prev_Low)：** `{info['prev_low']} 元`
                        - **近期最低點 (Recent Low)：** `{info['recent_min_low']} 元` (破底動作：{'✅ 曾破底' if info['had_breakdown'] else '❌'})
                        - **今日收盤價：** `{info['today_close']} 元` (重新站回：{'✅ 是' if info['reclaimed'] else '❌'}，紅K實體：{'✅ 是' if info['is_red'] else '❌'})
                        - **成交量能比：** 今日 `{info['today_vol']:,.0f} 張` / 5日均量 `{info['today_vol_ma5']:,.0f} 張` = **{info['vol_ratio']} 倍** (量能確認：{'✅ 達標' if info['vol_confirmed'] else '❌'})
                        """)

                if "B" in details:
                    with st.expander("📌 策略 B：區間盤整帶量突破 檢驗細節", expanded=True):
                        info = details["B"]
                        st.markdown(f"""
                        - **10 日盤整區間：** 最高 `{info['range_high']} 元` / 最低 `{info['range_low']} 元`
                        - **區間振幅：** **`{info['amplitude_pct']}%`** (小於門檻：{'✅ 符合盤整' if info['is_consolidating'] else '❌'})
                        - **突破確認：** 今日收盤 `{info['today_close']} 元` (創 11 日新高：{'✅ 突破' if info['is_new_high'] else '❌'})
                        - **均線支撐與扣抵：** 20MA=`{info['today_ma20']} 元`，60MA=`{info['today_ma60']} 元` (多頭排列與斜率向上：{'✅ 是' if info['ma_protected'] else '❌'})
                        - **爆量倍數：** **`{info['vol_ratio']} 倍`** 5日均量 (帶量確認：{'✅ 達標' if info['vol_confirmed'] else '❌'})
                        """)

                if "C" in details:
                    with st.expander("📌 策略 C：強勢股量縮回測支撐 檢驗細節", expanded=True):
                        info = details["C"]
                        st.markdown(f"""
                        - **強勢波段背景：** 過去 15 日曾出現單日漲幅>5% 且帶量 2 倍紅 K：{'✅ 曾有主力攻擊' if info['had_surge_k'] else '❌'}
                        - **均線生命線：** 60MA 季線斜率向上：{'✅ 季線多頭保護' if info['ma60_up'] else '❌'}
                        - **回測支撐與量縮：** 拉回接近 10MA/20MA 乖離內：{'✅ 支撐測試' if info['near_ma_flag'] else '❌'}，回測量能萎縮：{'✅ 籌碼安定' if info['vol_shrunk'] else '❌'}
                        - **轉強訊號確認：** 今日收紅 K ({'✅' if info['is_red'] else '❌'})、成交量較昨日放大 ({'✅' if info['vol_increased'] else '❌'})、突破昨日高點 `{info['yest_high']} 元` ({'✅' if info['break_yest_high'] else '❌'})
                        """)
        else:
            st.warning("查無此股票之歷史 K 線資料")


if __name__ == "__main__":
    main()
