import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import os
from data_loader import load_stock_metadata, fetch_stock_data, fetch_market_index, calculate_technical_indicators
from strategies import (
    filter_liquidity,
    strategy_a_breakdown_reversal,
    strategy_b_consolidation_breakout,
    strategy_c_pullback_support,
    check_market_filter,
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
    .market-banner-safe {
        background: linear-gradient(90deg, #e8f8f5 0%, #d1f2eb 100%);
        border: 1px solid #a3e4d7;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 20px;
    }
    .market-banner-warn {
        background: linear-gradient(90deg, #fef9e7 0%, #fdebd0 100%);
        border: 1px solid #f9e79f;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 20px;
    }
    .market-banner-danger {
        background: linear-gradient(90deg, #fdedec 0%, #fadbd8 100%);
        border: 1px solid #f5b7b1;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_stock_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """快取股票歷史日K線資料"""
    return fetch_stock_data(symbol, period=period)


@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_market_data(ticker: str = "^TWII", period: str = "6mo") -> pd.DataFrame:
    """快取台股大盤指數資料"""
    return fetch_market_index(ticker=ticker, period=period)


def main():
    st.title("📊 台股短線技術面選股工具")
    st.caption("基於量價結構、大盤環境濾網、型態學與均線的自動化即時篩選與回測系統")

    # ============================
    # 頂部大盤環境濾網看板 (Market Regime Bar)
    # ============================
    market_df = get_cached_market_data("^TWII", period="6mo")
    market_is_safe = True
    market_info = {}

    if market_df is not None and not market_df.empty:
        market_is_safe, market_info = check_market_filter(market_df, mode="any")
        latest_idx = market_df.iloc[-1]
        idx_close = latest_idx["Close"]
        idx_change = latest_idx.get("Change_Pct", 0.0)
        idx_ma5 = latest_idx.get("MA5", 0.0)
        idx_ma20 = latest_idx.get("MA20", 0.0)
        idx_ma60 = latest_idx.get("MA60", 0.0)

        # 判斷多空燈號
        is_above_ma60 = idx_close >= idx_ma60
        is_ma5_up = market_info.get("ma5_up", False)

        if is_above_ma60 and is_ma5_up:
            banner_class = "market-banner-safe"
            icon = "🟢"
            status_title = "大盤處於【強勢多頭安全環境】"
            advice = "加權指數位於季線 (60MA) 之上且 5MA 斜率向上，主力做多意願濃厚，允許積極執行順勢突破與破底翻買進！"
        elif is_above_ma60 or is_ma5_up:
            banner_class = "market-banner-warn"
            icon = "🟡"
            status_title = "大盤處於【震盪整理轉折環境】"
            advice = "加權指數處於季線之上但短線震盪，或正嘗試突破季線，短線做多建議嚴控部位成數與停損點。"
        else:
            banner_class = "market-banner-danger"
            icon = "🔴"
            status_title = "大盤處於【系統性空頭警戒環境】"
            advice = "加權指數跌破季線 (60MA) 且 5MA 走空向下，系統性風險較高，強烈建議保守觀望，嚴禁盲目接刀追價！"

        st.markdown(f"""
        <div class="{banner_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h4 style="margin: 0; color: #2C3E50;">{icon} {status_title}</h4>
                    <p style="margin: 5px 0 0 0; font-size: 13px; color: #555;">{advice}</p>
                </div>
                <div style="text-align: right; font-size: 13px;">
                    <b>加權指數 (^TWII)：</b> <span style="font-size: 16px; font-weight: bold; color: {'#E74C3C' if idx_change >= 0 else '#2ECC71'};">{idx_close:,.2f} ({idx_change:+.2f}%)</span><br>
                    <span style="color: #666;">5MA: {idx_ma5:,.1f} | 20MA: {idx_ma20:,.1f} | 60MA(季線): {idx_ma60:,.1f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ============================
    # 建立多功能 Tab 分頁
    # ============================
    tab1, tab2, tab3 = st.tabs([
        "🚀 即時短線選股與走勢圖", 
        "📊 策略半年歷史回測績效 (3D/5D/10D)", 
        "📘 策略規則手冊與 XQ XS 語法對照"
    ])

    # ============================
    # 側邊欄：參數設定
    # ============================
    st.sidebar.header("⚙️ 篩選與環境設定")

    # 大盤濾網開關
    st.sidebar.subheader("0. 大盤環境濾網")
    enable_market_filter = st.sidebar.checkbox(
        "啟用大盤濾網 (大盤弱勢時自動過濾做多)",
        value=False,
        help="當加權指數跌破季線且 5MA 下彎時，系統將自動暫停或過濾做多訊號，降低系統性崩跌風險。"
    )
    market_filter_mode = st.sidebar.selectbox(
        "大盤安全門檻",
        [("季線之上 或 5MA向上 (標準推薦)", "any"), ("季線之上 且 5MA向上 (極嚴格)", "both")],
        format_func=lambda x: x[0]
    )[1]

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
            strat_a_params["max_stop_loss_pct"] = st.slider(
                "停損空間上限 (%)", 
                3.0, 15.0, 7.0, step=0.5, 
                help="今日收盤價距離近 3 日最低點幅度需 < 7%，避免過度拉抬導致追價風險過高"
            )

    # 策略 A-2
    enable_strat_a2 = st.sidebar.checkbox("策略 A-2：破底翻後量縮回測買點 (二次回測)", value=True)
    strat_a2_params = {}
    if enable_strat_a2:
        with st.sidebar.expander("策略 A-2 細部參數"):
            strat_a2_params["lookback_trigger_days"] = st.slider("破底翻回溯天數", 3, 7, 5)
            strat_a2_params["max_distance_pct"] = st.slider("拉回靠近前低幅度門檻 (%)", 1.0, 5.0, 3.0, step=0.5, help="收盤價拉回至前低 0% 至 N% 區間")

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

    # 交易日回溯模擬選擇
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
                        df_target = calculate_technical_indicators(df_target)
                    else:
                        continue
                else:
                    df_target = df

                # 基準日對應之大盤
                sub_market = None
                if market_df is not None and not market_df.empty:
                    if backtest_offset > 0 and len(market_df) > backtest_offset + 60:
                        sub_market = market_df.iloc[:-backtest_offset].copy()
                        sub_market = calculate_technical_indicators(sub_market)
                    else:
                        sub_market = market_df

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
                    enable_strat_a2=enable_strat_a2,
                    enable_strat_b=enable_strat_b,
                    enable_strat_c=enable_strat_c,
                    strat_a_params=strat_a_params,
                    strat_a2_params=strat_a2_params,
                    strat_b_params=strat_b_params,
                    strat_c_params=strat_c_params,
                    market_df=sub_market,
                    require_market_filter=enable_market_filter,
                    market_filter_mode=market_filter_mode
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
    # TAB 1: 即時短線選股與走勢圖
    # ============================
    with tab1:
        results = st.session_state.get("scan_results", [])
        scanned_count = st.session_state.get("scanned_count", 0)
        liquidity_pass_count = st.session_state.get("liquidity_pass_count", 0)

        strat_a_hits = sum(1 for r in results if any(s == "策略A: 帶量破底翻" for s in r["signals"]))
        strat_a2_hits = sum(1 for r in results if any("策略A-2" in s for s in r["signals"]))
        strat_b_hits = sum(1 for r in results if any("策略B" in s for s in r["signals"]))
        strat_c_hits = sum(1 for r in results if any("策略C" in s for s in r["signals"]))

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("掃描總檔數", f"{scanned_count} 檔")
        with m2:
            st.metric("通過流動性", f"{liquidity_pass_count} 檔")
        with m3:
            st.metric("🔥 帶量破底翻 (A)", f"{strat_a_hits} 檔")
        with m4:
            st.metric("🔄 破底翻回測 (A-2)", f"{strat_a2_hits} 檔")
        with m5:
            st.metric("🚀 盤整突破 (B)", f"{strat_b_hits} 檔")
        with m6:
            st.metric("🎯 回測轉強 (C)", f"{strat_c_hits} 檔")

        st.markdown("---")
        st.subheader(f"📋 策略觸發精選名單 (共 {len(results)} 檔)")

        selected_symbol = None

        if results:
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

            st.dataframe(res_df, use_container_width=True, hide_index=True)

            st.markdown("### 🔍 個股深度 K 線與技術面分析")
            stock_options = [f"{r['symbol']} {r['name']} ({' | '.join(r['signals'])})" for r in results]
            selected_option = st.selectbox("選擇要檢視技術線圖的股票：", stock_options, index=0)
            selected_symbol = selected_option.split(" ")[0]

        else:
            st.info("💡 目前設定條件下未篩選出觸發訊號的股票。您可以：\n"
                    "1. 在左側側邊欄微調參數（例如調降成交量爆量倍數、微幅放寬振幅百分比）\n"
                    "2. 切換「掃描基準日」回溯檢視過去交易日（例如 1~4 個交易日前有航運、中鋼、統一等觸發）\n"
                    "3. 下方仍可手動選擇任一股票直接檢視技術走勢圖！")

            all_symbols = metadata_df["symbol"].tolist()
            if all_symbols:
                stock_options = [f"{s} {metadata_df[metadata_df['symbol'] == s]['name'].values[0]}" for s in all_symbols]
                selected_option = st.selectbox("🔍 選擇任一檔股票檢視 K 線走勢：", stock_options, index=0)
                selected_symbol = selected_option.split(" ")[0]

        if selected_symbol:
            st.markdown("---")
            stock_name = metadata_df[metadata_df["symbol"] == selected_symbol]["name"].values[0] if selected_symbol in metadata_df["symbol"].values else selected_symbol

            matched_result = next((r for r in results if r["symbol"] == selected_symbol), None)
            signals = matched_result["signals"] if matched_result else []

            full_df = get_cached_stock_data(selected_symbol, period="6mo")

            if full_df is not None and not full_df.empty:
                if backtest_offset > 0 and len(full_df) > backtest_offset + 60:
                    display_df = full_df.iloc[:-backtest_offset].copy()
                    display_df = calculate_technical_indicators(display_df)
                else:
                    display_df = full_df

                fig = plot_stock_kline(
                    df=display_df,
                    symbol=selected_symbol,
                    name=stock_name,
                    signals=signals,
                    days=60
                )
                st.plotly_chart(fig, use_container_width=True)

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
                            - **停損空間控制：** 距近 3 日最低點幅度 **`{info.get('stop_loss_pct', 0.0)}%`** (小於 7% 門檻：{'✅ 風險可控' if info.get('pass_stop_loss', True) else '❌ 過度追價'})
                            """)

                    if "A2" in details:
                        with st.expander("📌 策略 A-2：破底翻後量縮回測 檢驗細節", expanded=True):
                            info = details["A2"]
                            st.markdown(f"""
                            - **前期支撐低點 (Prev_Low)：** `{info['prev_low']} 元`
                            - **今日收盤價：** `{info['today_close']} 元`
                            - **距前低幅度：** **`{info['distance_pct']}%`** (0% 至 3% 內：{'✅ 完美回測支撐' if info['near_support'] else '❌'})
                            - **量能萎縮確認：** 今日 `{info['today_vol']:,.0f} 張` < 5日均量 `{info['today_vol_ma5']:,.0f} 張` ({'✅ 量縮無拋壓' if info['vol_shrunk'] else '❌'})
                            - **破底翻起漲點：** 距今約 `{info['reversal_ago_days']} 個交易日前`
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

    # ============================
    # TAB 2: 半年歷史回測績效報表
    # ============================
    with tab2:
        st.subheader("📈 三大短線策略過去半年歷史回測績效 (53 檔台股真實數據)")
        st.caption("統計訊號觸發後持有 3 天、5 天與 10 天之勝率、平均報酬率與盈虧比，並對比「大盤環境濾網」前後差異。")

        # 讀取回測報告數據
        bt_file = "backtest_results.json"
        if os.path.exists(bt_file):
            try:
                with open(bt_file, "r", encoding="utf-8") as f:
                    bt_data = json.load(f)

                # 呈現績效總表
                rows = []
                for s_name, data in bt_data.items():
                    if "3d" in data:
                        rows.append({
                            "策略項目": s_name,
                            "總觸發筆數": data["count"],
                            "3D 勝率": f"{data['3d']['win_rate']}%",
                            "3D 平均報酬": f"{data['3d']['avg_ret']:+.2f}%",
                            "5D 勝率": f"{data['5d']['win_rate']}%",
                            "5D 平均報酬": f"{data['5d']['avg_ret']:+.2f}%",
                            "10D 勝率": f"{data['10d']['win_rate']}%",
                            "10D 平均報酬": f"{data['10d']['avg_ret']:+.2f}%",
                            "10D 盈虧比": f"{data['10d']['pl_ratio']:.2f}",
                            "最大單筆獲利": f"+{data['10d']['max_gain']:.1f}%"
                        })
                perf_df = pd.DataFrame(rows)
                st.dataframe(perf_df, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("🛡️ 大盤環境濾網效益分析 (加權指數季線與 5MA)")
                b1, b2, b3 = st.columns(3)
                strat_b_stats = bt_data.get("策略B: 盤整帶量突破", {})
                if strat_b_stats and "10d" in strat_b_stats:
                    s10 = strat_b_stats["10d"]
                    with b1:
                        st.metric("策略 B 未濾網 10D 勝率", f"{s10['win_rate']}%", f"{s10['avg_ret']:+.2f}% 均報酬")
                    with b2:
                        st.metric("策略 B 啟用大盤濾網 10D 勝率", f"{s10['mkt_win_rate']}%", f"{s10['mkt_avg_ret']:+.2f}% 均報酬", delta_color="normal")
                    with b3:
                        delta_win = s10['mkt_win_rate'] - s10['win_rate']
                        st.metric("勝率提升幅度", f"+{delta_win:.1f}%", "大幅降低系統性風險")

                st.info("💡 **回測結論摘要**：\n"
                        "1. **策略 B (盤整帶量突破)** 是穩定性最佳的策略，10D 勝率達 72.7%，啟用大盤濾網後勝率更激增至 **88.9%**，平均報酬率達 **+12.16%**！\n"
                        "2. **策略 A (帶量破底翻)** 具有爆發性獲利潛力，10D 平均報酬率高達 **+12.12%**，盈虧比達 **37.93**，屬於典型的大賺小賠波段反轉型態。\n"
                        "3. **策略 C (量縮回測轉強)** 提供最高的交易頻率，適合於大盤震盪時尋求回測支撐的防守型買點。")

            except Exception as e:
                st.error(f"讀取回測報告失敗: {e}")
        else:
            st.warning("尚未產出歷史回測檔案，可手動執行 backtester.py 生成報告。")

    # ============================
    # TAB 3: 策略規範與 XQ XS 語法對照手冊
    # ============================
    with tab3:
        st.subheader("📘 台股量化策略規範與 XQ 全球贏家 XS 語法對照")
        st.markdown(r"""
        本專案已模組化為 Antigravity Workspace Skill (`tw-stock-strategy`)，支援直接將策略轉換為 XQ XS 選股腳本或量化模組。
        
        #### 1. 台股資料清洗標準
        - **成交量換算**：API 股數自動除以 1,000 轉換為「張」。
        - **基礎流動性過濾**：5 日均量 $\ge 1,000$ 張，收盤價 $\ge 15$ 元。
        - **大盤濾網**：加權指數 (`^TWII`) 需在季線 (60MA) 之上或 5MA 向上。

        #### 2. XQ 全球贏家 XS 選股腳本實作範例 (策略 C：量縮回測轉強)
        ```pascal
        // ==========================================
        // 策略 C: 強勢股量縮回測支撐 (XQ XS 選股腳本)
        // ==========================================
        input: minVol(1000, "5日均量門檻(張)");
        input: minPrice(15, "收盤價下限(元)");
        input: surgeDays(15, "攻擊紅K回溯天數");
        input: biasPct(2.0, "均線乖離率容許(%)");

        settotalbar(80);

        // 1. 基礎流動性過濾
        condition1 = Average(volume, 5) >= minVol and close >= minPrice;

        // 2. 60MA 季線向上
        condition2 = Average(close, 60) > Average(close, 60)[1];

        // 3. 過去 15 日曾出現單日漲幅 >= 5% 且成交量 >= 2 倍 5MA 之攻擊紅 K
        variable: hadSurge(false), i(0);
        hadSurge = false;
        for i = 1 to surgeDays begin
            if (close[i] - close[i+1]) / close[i+1] * 100 >= 5.0 
               and volume[i] >= Average(volume[i], 5) * 2.0 
               and close[i] > open[i] then
            begin
                hadSurge = true;
            end;
        end;

        // 4. 最近回測 10MA 或 20MA (乖離在正負 2% 內)
        variable: nearMA(false);
        nearMA = false;
        if absvalue(close[1] - Average(close[1], 10)) / Average(close[1], 10) * 100 <= biasPct 
           or absvalue(close[1] - Average(close[1], 20)) / Average(close[1], 20) * 100 <= biasPct then
        begin
            nearMA = true;
        end;

        // 5. 今日轉強: 收紅K、量增、收盤價突破昨日高點
        condition3 = close > open and volume > volume[1] and close > high[1];

        if condition1 and condition2 and hadSurge and nearMA and condition3 then ret = 1;
        ```
        """)


if __name__ == "__main__":
    main()
