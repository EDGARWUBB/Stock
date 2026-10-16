import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional, List, Dict, Any

def plot_stock_kline(
    df: pd.DataFrame, 
    symbol: str, 
    name: str = "", 
    signals: Optional[List[str]] = None,
    days: int = 60,
    w_bottom_info: Optional[Dict[str, Any]] = None,
    timeframe: str = "daily"  # "daily" (日K) 或 "weekly" (週K)
) -> go.Figure:
    """
    使用 Plotly 繪製台股 K 線圖與成交量圖，具備現代化觸控手勢縮放與平移 (Pan & Zoom)
    支援日 K 線 / 週 K 線雙週期切換
    符合台股紅漲綠跌習慣:
    - 漲: 紅色 (#E74C3C)
    - 跌: 綠色 (#2ECC71)
    包含:
    - K 線 (Candlestick) + 十字遊標 (Crosshair)
    - 均線 (MA5, MA10, MA20, MA60) 自適應日/週週期標籤
    - 成交量柱狀圖 (量能紅綠配合 K 線)
    - 均量線 (Vol_MA5)
    - 觸發訊號標記箭頭
    - W 底型態標註: 頸線 (Neckline)、雙底圓點 (L1/L2)、等幅測距目標價
    - 手勢操作優化:
      * dragmode='pan': 手機單指滑動或滑鼠拖曳直接「平移滑動」
      * 底部時間導航滑桿 (Rangeslider): 支援手勢雙向拉動縮放
      * 滾輪 / 雙指捏合縮放 (Pinch-to-zoom)
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="無可用歷史資料", showarrow=False, font=dict(size=18))
        return fig

    # 保留完整資料以利向左滑動回溯，以 days 設定初始顯示視野
    plot_df = df.copy()
    plot_df.sort_index(ascending=True, inplace=True)

    # 確保日期格式字串 (避開週末休市空隙)
    date_strs = [d.strftime("%Y-%m-%d") for d in plot_df.index]

    # 計算初始顯示範圍 (聚焦於最後 N 根 K 棒，其餘可自由向前滾動)
    if days > 0 and len(date_strs) > days:
        init_range = [len(date_strs) - days - 0.5, len(date_strs) - 0.5]
    else:
        init_range = [0, len(date_strs) - 0.5]

    # 計算成交量紅綠顏色 (符合台股紅漲綠跌: 收盤 >= 開盤 為紅，否則為綠)
    vol_colors = [
        "#E74C3C" if c >= o else "#2ECC71"
        for c, o in zip(plot_df["Close"], plot_df["Open"])
    ]

    is_weekly = (timeframe.lower() == "weekly")
    freq_unit = "週" if is_weekly else "日"
    vol_title = "週成交量 (張)" if is_weekly else "日成交量 (張)"

    # 建立 2 列子圖 (K線圖高 72%，成交量圖高 28%)
    title_text = f"<b>{symbol} {name}</b> - {freq_unit} K 線走勢圖"
    if signals:
        title_text += f" <span style='color:#E74C3C;'>[觸發: {', '.join(signals)}]</span>"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("", vol_title),
        row_heights=[0.72, 0.28]
    )

    # 1. K 線圖 (Candlestick)
    fig.add_trace(
        go.Candlestick(
            x=date_strs,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="K線",
            increasing_line_color="#E74C3C",
            increasing_fillcolor="#E74C3C",
            decreasing_line_color="#2ECC71",
            decreasing_fillcolor="#2ECC71",
            text=[f"日期: {d}<br>開: {o:.2f}<br>高: {h:.2f}<br>低: {l:.2f}<br>收: {c:.2f}" 
                  for d, o, h, l, c in zip(date_strs, plot_df["Open"], plot_df["High"], plot_df["Low"], plot_df["Close"])],
            hoverinfo="text"
        ),
        row=1, col=1
    )

    # 2. 均線: 5MA, 10MA, 20MA, 60MA
    ma5_label = "5W MA (5週均線)" if is_weekly else "5MA (週線)"
    ma10_label = "10W MA (10週均線)" if is_weekly else "10MA (雙週線)"
    ma20_label = "20W MA (20週均線)" if is_weekly else "20MA (月線)"
    ma60_label = "60W MA (60週均線)" if is_weekly else "60MA (季線)"

    if "MA5" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=date_strs,
                y=plot_df["MA5"],
                mode="lines",
                name=ma5_label,
                line=dict(color="#F39C12", width=1.5),
                hoverinfo="name+y"
            ),
            row=1, col=1
        )

    if "MA10" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=date_strs,
                y=plot_df["MA10"],
                mode="lines",
                name=ma10_label,
                line=dict(color="#1ABC9C", width=1.2, dash="dash"),
                hoverinfo="name+y"
            ),
            row=1, col=1
        )

    if "MA20" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=date_strs,
                y=plot_df["MA20"],
                mode="lines",
                name=ma20_label,
                line=dict(color="#3498DB", width=2.0),
                hoverinfo="name+y"
            ),
            row=1, col=1
        )

    if "MA60" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=date_strs,
                y=plot_df["MA60"],
                mode="lines",
                name=ma60_label,
                line=dict(color="#9B59B6", width=2.0),
                hoverinfo="name+y"
            ),
            row=1, col=1
        )

    # 2.5 策略 D: W 底（雙重底）特徵繪製 (僅在日K模式下標註)
    if w_bottom_info and not is_weekly:
        neck_p = w_bottom_info.get("neckline_price")
        l1_p = w_bottom_info.get("l1_price")
        l1_d = str(w_bottom_info.get("l1_date", ""))
        l2_p = w_bottom_info.get("l2_price")
        l2_d = str(w_bottom_info.get("l2_date", ""))
        tgt_p = w_bottom_info.get("target_price")

        # 繪製頸線 (Neckline) 橘色虛線
        if neck_p and len(date_strs) > 0:
            fig.add_shape(
                type="line",
                x0=date_strs[0],
                x1=date_strs[-1],
                y0=neck_p,
                y1=neck_p,
                line=dict(color="#E67E22", width=2, dash="dash"),
                row=1, col=1
            )
            fig.add_annotation(
                x=date_strs[-1],
                y=neck_p,
                text=f"<b>頸線 (Neckline): {neck_p:.2f}元</b>",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(color="#D35400", size=11),
                bgcolor="rgba(254, 249, 231, 0.9)",
                bordercolor="#E67E22",
                borderwidth=1,
                row=1, col=1
            )

        # 繪製 L1（左腳）與 L2（右腳）圓點標註
        scatter_x = []
        scatter_y = []
        scatter_text = []
        if l1_d in date_strs and l1_p:
            scatter_x.append(l1_d)
            scatter_y.append(l1_p)
            scatter_text.append(f"<b>L1 左腳</b><br>{l1_p:.2f}元")
        if l2_d in date_strs and l2_p:
            scatter_x.append(l2_d)
            scatter_y.append(l2_p)
            scatter_text.append(f"<b>L2 右腳</b><br>{l2_p:.2f}元")

        if scatter_x:
            fig.add_trace(
                go.Scatter(
                    x=scatter_x,
                    y=scatter_y,
                    mode="markers+text",
                    name="W底雙腳 (L1 / L2)",
                    marker=dict(symbol="circle", size=13, color="#8E44AD", line=dict(color="white", width=2)),
                    text=scatter_text,
                    textposition=["bottom center"] * len(scatter_x),
                    textfont=dict(color="#8E44AD", size=11),
                    hoverinfo="text"
                ),
                row=1, col=1
            )

        # 繪製 W 底等幅測距目標價 綠色點線
        if tgt_p and len(date_strs) > 0:
            fig.add_shape(
                type="line",
                x0=date_strs[0],
                x1=date_strs[-1],
                y0=tgt_p,
                y1=tgt_p,
                line=dict(color="#27AE60", width=2, dash="dot"),
                row=1, col=1
            )
            fig.add_annotation(
                x=date_strs[-1],
                y=tgt_p,
                text=f"<b>🎯 W底等幅目標價: {tgt_p:.2f}元</b>",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(color="#196F3D", size=11),
                bgcolor="rgba(234, 250, 234, 0.9)",
                bordercolor="#27AE60",
                borderwidth=1,
                row=1, col=1
            )

    # 3. 標記觸發訊號位置 (在最新一個 K 棒高點上方標註箭頭)
    if signals and len(plot_df) > 0 and not is_weekly:
        latest_date = date_strs[-1]
        latest_high = float(plot_df["High"].iloc[-1])
        signal_annotation = "<br>".join([f"🔥 {s}" for s in signals])
        fig.add_annotation(
            x=latest_date,
            y=latest_high,
            text=f"<b>訊號觸發!</b><br>{signal_annotation}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="#E74C3C",
            ax=0,
            ay=-55,
            bordercolor="#E74C3C",
            borderwidth=1.5,
            borderpad=4,
            bgcolor="rgba(255, 235, 235, 0.95)",
            font=dict(size=11, color="#900C3F"),
            row=1, col=1
        )

    # 4. 成交量柱狀圖 (第 2 列)
    fig.add_trace(
        go.Bar(
            x=date_strs,
            y=plot_df["Volume"],
            name="成交量 (張)",
            marker_color=vol_colors,
            text=[f"量: {v:.0f} 張" for v in plot_df["Volume"]],
            hoverinfo="text+x"
        ),
        row=2, col=1
    )

    # 5. 5 均量線 (Vol_MA5)
    vol_ma5_label = "5週均量" if is_weekly else "5日均量"
    if "Vol_MA5" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=date_strs,
                y=plot_df["Vol_MA5"],
                mode="lines",
                name=vol_ma5_label,
                line=dict(color="#F39C12", width=1.5, dash="dot"),
                hoverinfo="name+y"
            ),
            row=2, col=1
        )

    # 圖表整體佈局美化與觸控優化
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=16),
            x=0.02,
            y=0.98
        ),
        dragmode="pan",  # 核心重點：讓單指觸控與滑鼠拖曳為平移滑動，避免變成擾人的方框選取
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=45, r=45, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        ),
        height=660
    )

    # 1 列 (K線): 去除假日斷層 + 十字遊標輔助線 + 與下方同步縮放
    fig.update_xaxes(
        type="category",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikedash="dot",
        spikecolor="#888888",
        rangeslider=dict(visible=False),
        range=init_range,
        row=1, col=1
    )

    # 2 列 (成交量): 包含時間導航滑桿 (Rangeslider)，方便手指雙向拉動縮放
    fig.update_xaxes(
        type="category",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikedash="dot",
        spikecolor="#888888",
        rangeslider=dict(visible=True, thickness=0.08),
        range=init_range,
        row=2, col=1
    )

    fig.update_yaxes(
        title_text="價格 (元)",
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikedash="dot",
        spikecolor="#888888",
        row=1, col=1,
        gridcolor="#EEEEEE"
    )
    fig.update_yaxes(
        title_text="張數",
        row=2, col=1,
        gridcolor="#EEEEEE"
    )

    return fig
