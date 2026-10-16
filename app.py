import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import os
# 載入自訂模組並加入相容性保護 (防止雲端未同步更新所有檔案時拋出 ImportError)
import yfinance as yf
from data_loader import load_stock_metadata, fetch_stock_data, calculate_technical_indicators

try:
    from data_loader import resample_to_weekly
except ImportError:
    def resample_to_weekly(df):
        return df

try:
    from data_loader import fetch_market_index
except ImportError:
    def fetch_market_index(ticker: str = "^TWII", period: str = "6mo"):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, auto_adjust=False)
            if df is None or df.empty or len(df) < 20:
                return None
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.sort_index(ascending=True, inplace=True)
            df["MA5"] = df["Close"].rolling(window=5).mean()
            df["MA10"] = df["Close"].rolling(window=10).mean()
            df["MA20"] = df["Close"].rolling(window=20).mean()
            df["MA60"] = df["Close"].rolling(window=60).mean()
            df["Change_Pct"] = df["Close"].pct_change() * 100
            df["Is_Red"] = df["Close"] > df["Open"]
            return df
        except Exception:
            return None

try:
    from strategies import (
        filter_liquidity,
        strategy_a_breakdown_reversal,
        strategy_a2_reversal_pullback,
        strategy_b_consolidation_breakout,
        strategy_c_pullback_support,
        check_market_filter,
        scan_single_stock
    )
except ImportError:
    from strategies import (
        filter_liquidity,
        strategy_a_breakdown_reversal,
        strategy_b_consolidation_breakout,
        strategy_c_pullback_support,
        scan_single_stock
    )
    def check_market_filter(market_df, mode="any"):
        if market_df is None or len(market_df) < 65:
            return True, {}
        today = market_df.iloc[-1]
        yest = market_df.iloc[-2]
        close = float(today["Close"])
        ma5_today = float(today.get("MA5", 0.0))
        ma5_yest = float(yest.get("MA5", 0.0))
        ma60_today = float(today.get("MA60", 0.0))
        above_ma60 = close >= ma60_today
        ma5_up = ma5_today >= ma5_yest
        is_safe = (above_ma60 or ma5_up) if mode == "any" else (above_ma60 and ma5_up)
        return is_safe, {"above_ma60": above_ma60, "ma5_up": ma5_up}

    def strategy_a2_reversal_pullback(df, **kwargs):
        return False, {"reason": "模組更新中"}

from chart_plotter import plot_stock_kline
import importlib
import inspect

try:
    import data_loader
    import strategies
    import chart_plotter
    importlib.reload(data_loader)
    importlib.reload(strategies)
    importlib.reload(chart_plotter)
except Exception:
    pass

def safe_scan_single_stock(**kwargs):
    """防禦性呼叫，避免模組在快取或版本不一致時因參數不符而拋出 TypeError"""
    try:
        sig = inspect.signature(scan_single_stock)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_kw:
            valid_kwargs = kwargs
        else:
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return scan_single_stock(**valid_kwargs)
    except Exception:
        return scan_single_stock(**kwargs)

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
        [
            "🔥 全市場熱門活躍股 (約 525 檔，日量>=500張，推薦)",
            "🌐 全市場上市櫃所有股票 (全市場 1,979 檔)",
            "⚡ 內建精選與台灣50核心 (53 檔，極速秒開)",
            "📂 上傳自訂 CSV",
            "✍️ 手動輸入代號清單"
        ],
        index=0
    )

    metadata_df = pd.DataFrame()

    if "全市場熱門活躍股" in stock_source:
        if os.path.exists("stocks_active_hot.csv"):
            metadata_df = pd.read_csv("stocks_active_hot.csv")
        else:
            metadata_df = load_stock_metadata("stocks_default.csv")
        
        limit_count = st.sidebar.slider("掃描活躍股票數量上限", min_value=50, max_value=len(metadata_df), value=min(300, len(metadata_df)), step=50)
        metadata_df = metadata_df.head(limit_count)

    elif "全市場上市櫃所有股票" in stock_source:
        if os.path.exists("stocks_all_market.csv"):
            all_df = pd.read_csv("stocks_all_market.csv")
        else:
            all_df = load_stock_metadata("stocks_default.csv")

        market_filter_sel = st.sidebar.selectbox("市場類別篩選", ["全部上市櫃 (1,979檔)", "僅上市股票 (TWSE)", "僅上櫃股票 (TPEx)"])
        if "僅上市" in market_filter_sel:
            all_df = all_df[all_df["market"].str.contains("TWSE", na=False)]
        elif "僅上櫃" in market_filter_sel:
            all_df = all_df[all_df["market"].str.contains("TPEx", na=False)]

        scan_scope = st.sidebar.select_slider(
            "掃描檔數上限 (依流動性排序，避免超時)",
            options=[100, 250, 500, 1000, "全部掃描"],
            value=250,
            help="短線交易需優先過濾低成交量殭屍股，建議先掃描成交量居前之個股"
        )
        if scan_scope != "全部掃描":
            all_df.sort_values(by="latest_vol_lots", ascending=False, inplace=True)
            metadata_df = all_df.head(int(scan_scope))
        else:
            metadata_df = all_df

    elif "內建精選與台灣50核心" in stock_source:
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

    elif "上傳自訂 CSV" in stock_source:
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

    with st.sidebar.expander("🔄 證交所/櫃買全市場名單更新"):
        st.caption("自動連線台灣證券交易所 (TWSE) 與櫃買中心 (TPEx) OpenAPI 抓取最新普通股。")
        if st.button("立即連線更新全市場名單", use_container_width=True):
            with st.spinner("正在連線證交所與櫃買中心下載中..."):
                try:
                    from data_loader import update_full_market_stocks
                    res_df = update_full_market_stocks(save_csv=True)
                    st.success(f"更新完成！共取得 {len(res_df)} 檔上市櫃股票。")
                    st.rerun()
                except Exception as e:
                    st.error(f"更新失敗: {e}")

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

    # 策略 D
    enable_strat_d = st.sidebar.checkbox("策略 D：標準 W 底（雙重底）型態", value=True)
    strat_d_params = {}
    if enable_strat_d:
        with st.sidebar.expander("策略 D 細部參數"):
            strat_d_params["enable_d1"] = st.checkbox("啟用 模式 D-1 (右腳潛伏轉折買點)", value=True)
            strat_d_params["enable_d2"] = st.checkbox("啟用 模式 D-2 (帶量突破頸線買點)", value=True)
            strat_d_params["min_neckline_gain_pct"] = st.slider("頸線反彈最小幅度 (%)", 4.0, 15.0, 6.0, step=0.5, help="頸線與左腳 L1 價格差距需 >= 此門檻")
            strat_d_params["min_interval_days"] = st.slider("雙底最小間隔天數", 5, 15, 7, help="左腳與右腳之間需相隔至少 N 天")
            strat_d_params["d1_max_distance_pct"] = st.slider("D-1 距離右腳上限 (%)", 1.0, 6.0, 4.0, step=0.5, help="今日收盤價距離 L2 幅度上限")
            strat_d_params["d2_vol_multiplier"] = st.slider("D-2 突破成交量倍數", 1.2, 3.0, 1.8, step=0.1, help="突破日成交量相對於 5 日均量倍數")
            min_l2 = st.slider("右腳比左腳最低允許 (%)", -5.0, 0.0, -1.5, step=0.5)
            max_l2 = st.slider("右腳比左腳最高墊高 (%)", 1.0, 8.0, 3.5, step=0.5)
            strat_d_params["min_l2_ratio"] = min_l2 / 100.0
            strat_d_params["max_l2_ratio"] = max_l2 / 100.0

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
                res = safe_scan_single_stock(
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
                    enable_strat_d=enable_strat_d,
                    strat_a_params=strat_a_params,
                    strat_a2_params=strat_a2_params,
                    strat_b_params=strat_b_params,
                    strat_c_params=strat_c_params,
                    strat_d_params=strat_d_params,
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
        strat_d_hits = sum(1 for r in results if any("策略D" in s for s in r["signals"]))

        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
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
        with m7:
            st.metric("👑 W底雙重底 (D)", f"{strat_d_hits} 檔")

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

            # K 線圖控制列 (支援手勢縮放、日K / 週K 切換、預設視角切換)
            chart_ctrl1, chart_ctrl2, chart_ctrl3 = st.columns([0.3, 0.45, 0.25])
            with chart_ctrl1:
                kline_freq = st.radio(
                    "📊 K 線週期",
                    ["日 K 線", "週 K 線"],
                    index=0,
                    horizontal=True,
                    help="日 K 適合短線進出；週 K 適合判定大格局趨勢與支撐壓力"
                )
            with chart_ctrl2:
                range_options = ["近 60 根", "近 120 根", "近半年", "全部歷史"]
                selected_range = st.radio(
                    "🔍 初始顯示視野",
                    range_options,
                    index=0,
                    horizontal=True,
                    help="設定初始圖表縮放區間，圖表下方亦可直接用手指/滑鼠拖曳滑桿進行連續無段縮放！"
                )
            with chart_ctrl3:
                st.caption("🖐️ **手勢操作技巧**：\n- 雙指捏合／滾輪：無段縮放\n- 手指滑動／拖曳：平移時間\n- 底部導航滑桿：雙向拉動視窗")

            is_weekly = (kline_freq == "週 K 線")
            fetch_period = "2y" if is_weekly else "1y"

            full_df = get_cached_stock_data(selected_symbol, period=fetch_period)

            if full_df is not None and not full_df.empty:
                # 處理基準日偏移 (回溯模擬)
                if backtest_offset > 0 and len(full_df) > backtest_offset + 30:
                    raw_df = full_df.iloc[:-backtest_offset].copy()
                else:
                    raw_df = full_df.copy()

                if is_weekly:
                    display_df = resample_to_weekly(raw_df)
                    timeframe_val = "weekly"
                    if selected_range == "近 60 根":
                        display_days = 60
                    elif selected_range == "近 120 根":
                        display_days = 120
                    elif selected_range == "近半年":
                        display_days = 26
                    else:
                        display_days = len(display_df)
                else:
                    display_df = calculate_technical_indicators(raw_df)
                    timeframe_val = "daily"
                    if selected_range == "近 60 根":
                        display_days = 60
                    elif selected_range == "近 120 根":
                        display_days = 120
                    elif selected_range == "近半年":
                        display_days = 130
                    else:
                        display_days = len(display_df)

                w_bottom_info = matched_result.get("details", {}).get("D") if (matched_result and matched_result.get("details")) else None

                fig = plot_stock_kline(
                    df=display_df,
                    symbol=selected_symbol,
                    name=stock_name,
                    signals=signals,
                    days=display_days,
                    w_bottom_info=w_bottom_info,
                    timeframe=timeframe_val
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
                        "displaylogo": False,
                        "toImageButtonOptions": {"format": "png", "filename": f"{selected_symbol}_{timeframe_val}_kline"}
                    }
                )

                latest_k = display_df.iloc[-1]
                c1, c2, c3, c4, c5 = st.columns(5)
                price_label = "本週收盤價" if is_weekly else "最新收盤價"
                change_label = "本週漲跌幅" if is_weekly else "最新漲跌幅"
                vol_label = "本週成交量" if is_weekly else "當日成交量"
                ma5_lbl = "5W MA (5週線)" if is_weekly else "5MA 週線"
                ma20_lbl = "20W MA (20週線)" if is_weekly else "20MA 月線"
                ma60_lbl = "60W MA (60週線)" if is_weekly else "60MA 季線"

                with c1:
                    st.metric(price_label, f"{latest_k['Close']:.2f} 元", f"{latest_k['Change_Pct']:+.2f}%")
                with c2:
                    st.metric(vol_label, f"{latest_k['Volume']:,.0f} 張")
                with c3:
                    st.metric(ma5_lbl, f"{latest_k.get('MA5', 0.0):.2f} 元")
                with c4:
                    st.metric(ma20_lbl, f"{latest_k.get('MA20', 0.0):.2f} 元")
                with c5:
                    st.metric(ma60_lbl, f"{latest_k.get('MA60', 0.0):.2f} 元")

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

                    if "D" in details:
                        with st.expander("📌 策略 D：標準 W 底（雙重底） 檢驗細節", expanded=True):
                            info = details["D"]
                            sub_mode_str = "、".join(info.get('sub_modes', ['W底型態']))
                            tgt_gain = round(((info['target_price'] - info['today_close']) / info['today_close'] * 100.0), 2) if info.get('today_close') else 0.0
                            st.markdown(f"""
                            - **觸發子模式：** **{sub_mode_str}**
                            - **左腳最低點 (L1)：** `{info['l1_price']} 元` (日期: `{info['l1_date']}`)，當日量 `{info['l1_vol']:,.0f} 張`
                            - **右腳最低點 (L2)：** `{info['l2_price']} 元` (日期: `{info['l2_date']}`)，當日量 `{info['l2_vol']:,.0f} 張`
                            - **中間頸線峰位 (Neckline)：** `{info['neckline_price']} 元` (日期: `{info['neckline_date']}`)
                            - **空間幅度過濾：** 頸線比左腳高出 **`{info['neckline_gain_pct']:.2f}%`** (大於 6% 門檻：✅ 空間充分，非死魚盤整)
                            - **雙底對稱與間隔：** 兩腳相隔 **`{info['interval_days']} 個交易日`** (>=門檻)，L2/L1 價差 **`{info['l2_diff_pct']:+.2f}%`** (✅ 雙底支撐確立)
                            - **量能結構過濾：** 右腳量縮於左腳或小於 20MA 均量：{'✅ 賣壓枯竭' if info['vol_ok'] else '❌'} (今日量/5MA比：`{info['vol_ratio']} 倍`)
                            - **🎯 等幅測距目標價：** **`{info['target_price']} 元`** (預估潛在漲幅：**`{tgt_gain:+.2f}%`**)
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

        #### 3. XQ 全球贏家 XS 選股腳本實作範例 (策略 D：標準 W 底帶量突破)
        ```pascal
        // ==========================================
        // 策略 D: 標準 W 底 (Double Bottom) 帶量突破頸線 (XQ XS 選股腳本)
        // ==========================================
        input: minVol(1000, "5日均量門檻(張)");
        input: minPrice(15, "收盤價下限(元)");
        input: neckPct(6.0, "頸線反彈最小幅度(%)");
        input: volMult(1.8, "D-2突破爆量倍數");

        settotalbar(60);

        // 1. 基礎流動性過濾
        condition1 = Average(volume, 5) >= minVol and close >= minPrice;

        // 2. 左腳 L1: 過去第 16 天至第 30 天之最低價
        variable: L1_Price(0), L1_Bar(0), L2_Price(0), L2_Bar(0), Neck_Price(0), i(0);
        L1_Price = Lowest(low[16], 15);
        for i = 16 to 30 begin
            if low[i] = L1_Price then L1_Bar = i;
        end;

        // 3. 右腳 L2: 過去第 2 天至第 10 天之最低價
        L2_Price = Lowest(low[2], 9);
        for i = 2 to 10 begin
            if low[i] = L2_Price then L2_Bar = i;
        end;

        // 4. 頸線峰位: L1 之後至前 5 天最高價
        Neck_Price = Highest(high[5], maxlist(L1_Bar - 5, 1));

        // 5. 兩腳間隔 >= 7 天，且 L2/L1 價差在 -1.5% 至 +3.5% 之間 (左低右高支撐確立)
        condition2 = (L1_Bar - L2_Bar >= 7) 
                     and (L2_Price >= L1_Price * 0.985) 
                     and (L2_Price <= L1_Price * 1.035);

        // 6. 空間幅度過濾：頸線高出左腳至少 6%
        condition3 = (Neck_Price - L1_Price) / L1_Price * 100 >= neckPct;

        // 7. 模式 D-2 帶量突破頸線買點 (今日收紅、創高突頸線、出量1.8倍)
        condition4 = (close > Neck_Price) and (close > open) and (volume >= Average(volume, 5) * volMult);

        if condition1 and condition2 and condition3 and condition4 then ret = 1;
        ```
        """)


if __name__ == "__main__":
    main()
