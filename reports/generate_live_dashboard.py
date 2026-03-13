"""
generate_live_dashboard.py
==========================
生成 TradingView 風格 Live Dashboard（純深色主題）。
新功能:
  - 點擊持股顯示個股 K 線圖（Yahoo Finance 資料，分頁切換）
  - 交易紀錄改為可折疊表格
  - scrollZoom 禁用，固定視窗手動縮放
  - 圖表區填滿視窗高度
"""

import json
import re
import time
import pandas as pd
import plotly.graph_objects as go
import os
import requests
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    print("[WARN] yfinance 未安裝，個股圖表功能停用")

# ─── 讀取資料 ─────────────────────────────────────────────────────────────────
with open("portfolio.json", "r") as f:
    pf = json.load(f)

latest_date       = pf.get("latest_date", "N/A")
latest_prices     = pf.get("latest_prices", {})
history           = pf.get("history", [])
dates             = [row["date"] for row in history]
navs              = [row["nav"]  for row in history]
trade_log_history = pf.get("trade_log_history", [])

# ─── 統計計算 ──────────────────────────────────────────────────────────────────
current_nav = navs[-1] if navs else pf.get("cash", 1_000_000.0)
prev_nav    = navs[-2] if len(navs) >= 2 else 1_000_000.0
init_nav    = 1_000_000.0
daily_ret   = (current_nav / prev_nav) - 1
total_ret   = (current_nav / init_nav) - 1
cash        = pf.get("cash", 0)
positions   = pf.get("positions", {})
cost_basis  = pf.get("cost_basis", {})

# ─── 取得股票中文名稱 ─────────────────────────────────────────────────────────
def fetch_stock_names(stock_ids):
    name_map = {}
    try:
        resp = requests.get(
            "https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInfo",
            timeout=10
        )
        df = pd.DataFrame(resp.json()["data"])
        df = df[df["stock_id"].isin([str(s) for s in stock_ids])].drop_duplicates("stock_id")
        name_map = dict(zip(df["stock_id"], df["stock_name"]))
    except Exception as e:
        print(f"[WARN] 無法取得股票名稱: {e}")
    return name_map

name_map = fetch_stock_names(list(positions.keys()))

# ─── 預先抓取個股 OHLCV (yfinance，1年日線) ──────────────────────────────────
def fetch_stock_ohlcv(stock_ids):
    """回傳 {stock_id: {dates:[], opens:[], highs:[], lows:[], closes:[], volumes:[]}} 或 {}"""
    if not HAS_YF:
        return {}
    result = {}
    symbols = [sid + ".TW" for sid in stock_ids] + [sid + ".TWO" for sid in stock_ids]
    try:
        # 批次下載，減少 API 呼叫次數
        raw = yf.download(
            symbols, period="1y", interval="1d",
            auto_adjust=True, progress=False, threads=True
        )
        for sid in stock_ids:
            # 優先找 .TW，若無資料再找 .TWO
            for suffix in [".TW", ".TWO"]:
                sym = sid + suffix
                try:
                    if len(symbols) == 1:
                        df = raw
                    else:
                        df = raw.xs(sym, axis=1, level=1) if sym in raw.columns.get_level_values(1) else None
                    if df is not None and not df.empty:
                        df = df.dropna(subset=["Close"])
                        if not df.empty:
                            result[sid] = {
                                "dates":   df.index.strftime("%Y-%m-%d").tolist(),
                                "opens":   [round(v, 2) for v in df["Open"].tolist()],
                                "highs":   [round(v, 2) for v in df["High"].tolist()],
                                "lows":    [round(v, 2) for v in df["Low"].tolist()],
                                "closes":  [round(v, 2) for v in df["Close"].tolist()],
                                "volumes": [int(v) for v in df["Volume"].tolist()],
                            }
                            break # Found data for this stock, move to next sid
                except Exception as e:
                    print(f"[WARN] {sym}: {e}")
    except Exception as e:
        print(f"[WARN] yfinance batch download failed: {e}")
    print(f"[INFO] 抓取個股 OHLCV: {len(result)}/{len(stock_ids)} 支成功")
    return result

stock_ids = list(positions.keys())
stock_ohlcv = fetch_stock_ohlcv(stock_ids)
stock_ohlcv_json = json.dumps(stock_ohlcv, ensure_ascii=False)

# ─── 持股資料整理 ─────────────────────────────────────────────────────────────
stock_data   = []
total_equity = 0
for stock, shares in positions.items():
    price = latest_prices.get(stock, 0)
    val   = shares * price
    total_equity += val
    
    # 計算 PnL
    cost_price = cost_basis.get(stock, price)
    pnl = (price - cost_price) * shares
    pnl_pct = (price / cost_price - 1) * 100 if cost_price > 0 else 0

    stock_data.append({
        "stock":  stock,
        "name":   name_map.get(stock, stock),
        "shares": shares,
        "price":  price,
        "value":  val,
        "pnl":    pnl,
        "pnl_pct": pnl_pct
    })

stock_data.sort(key=lambda x: x["value"], reverse=True)
for item in stock_data:
    item["weight"] = item["value"] / current_nav if current_nav > 0 else 0

# ─── 顏色常數 ─────────────────────────────────────────────────────────────────
TV_BG    = "#131722"
TV_PANEL = "#1E222D"
TV_GRID  = "#2B3139"
TV_TEXT  = "#D1D4DC"
TV_MUTED = "#8D94A6"
TV_GREEN = "#22AB94"
TV_RED   = "#F05350"
TV_BLUE  = "#2962FF"

CHART_COLORS = [
    "#2962FF", "#1565C0", "#42A5F5", "#00BCD4", "#26C6DA",
    "#00ACC1", "#29B6F6", "#0288D1", "#4CAF50", "#00897B",
    "#7E57C2", "#9575CD", "#FFA726", "#FF7043", "#546E7A",
    "#78909C", "#5C6BC0", "#3949AB", "#1E88E5", "#039BE5",
]

# ─── NAV 折線圖（scrollZoom 禁用）────────────────────────────────────────────
nav_min = min(navs) * 0.99 if navs else 0
nav_max = max(navs) * 1.01 if navs else 1000000

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=dates, y=navs, mode="lines",
    line=dict(color=TV_BLUE, width=2),
    fill="tozeroy", fillcolor="rgba(41, 98, 255, 0.15)",
    hoverinfo="x+y"
))
fig_line.update_layout(
    margin=dict(l=50, r=20, t=20, b=40),
    plot_bgcolor=TV_PANEL, paper_bgcolor=TV_PANEL,
    font=dict(color=TV_TEXT),
    xaxis=dict(showgrid=True, gridcolor=TV_GRID, tickcolor=TV_MUTED, fixedrange=False),
    yaxis=dict(showgrid=True, gridcolor=TV_GRID, tickformat=",.0f",
               side="right", tickcolor=TV_MUTED, range=[nav_min, nav_max], fixedrange=False),
    hovermode="x unified",
    dragmode="pan",
)
html_line = fig_line.to_html(
    include_plotlyjs="cdn", full_html=False,
    config={"displayModeBar": True, "scrollZoom": False,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"]}
)

# ─── 圓餅圖 ──────────────────────────────────────────────────────────────────
labels = ["Cash"] + [s["stock"] for s in stock_data]
values = [cash]   + [s["value"] for s in stock_data]
pie_colors = [TV_MUTED] + [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(stock_data))]

fig_pie = go.Figure(data=[go.Pie(
    labels=labels, values=values, hole=.65,
    marker=dict(colors=pie_colors, line=dict(color="#131722", width=1.5)),
    textinfo="none", hoverinfo="label+percent"
)])
fig_pie.update_layout(
    showlegend=False,
    margin=dict(l=8, r=8, t=8, b=8),
    plot_bgcolor=TV_PANEL, paper_bgcolor=TV_PANEL,
    font=dict(color=TV_TEXT),
    height=185,
)
html_pie = fig_pie.to_html(
    include_plotlyjs=False, full_html=False,
    config={"displayModeBar": False}
)

# ─── Holdings Table HTML（加上 onclick）──────────────────────────────────────
holdings_html = ""
for item in stock_data:
    color = CHART_COLORS[stock_data.index(item) % len(CHART_COLORS)]
    safe_name = item['name'].replace("'", "\\'").replace('"', '\\"')
    pnl_color = TV_GREEN if item['pnl'] >= 0 else TV_RED
    pnl_sign = "+" if item['pnl'] >= 0 else ""
    holdings_html += f"""
    <div class="table-row holding-row" onclick="loadStockChart('{item['stock']}', '{safe_name}')" data-stock="{item['stock']}">
        <div class="col-ticker">
            <span class="stock-id">{item['stock']}</span>
            <span class="stock-name">{item['name']}</span>
        </div>
        <div class="col-right">{item['price']:.2f}</div>
        <div class="col-right">{item['shares']}</div>
        <div class="col-right">${item['value']:,.0f}</div>
        <div class="col-right" style="color:{color}; font-weight:600;">{item['weight'] * 100:.1f}%</div>
        <div class="col-right col-pnl" style="color:{pnl_color}; font-weight:600; display:flex; flex-direction:column; align-items:flex-end;">
            <span>{pnl_sign}{item['pnl']:,.0f}</span>
            <span style="font-size:10px; opacity:0.8;">{pnl_sign}{item['pnl_pct']:.1f}%</span>
        </div>
    </div>"""

# ─── 每日 NAV 歷史 ────────────────────────────────────────────────────────────
history_table_rows = ""
display_history = history[-30:]
for i, row in enumerate(reversed(display_history)):
    date_str   = row["date"]
    nav_val    = row["nav"]
    global_idx = history.index(row)
    prev_val   = history[global_idx - 1]["nav"] if global_idx > 0 else init_nav
    day_ret    = (nav_val / prev_val - 1) * 100
    color      = TV_GREEN if day_ret >= 0 else TV_RED
    sign       = "+" if day_ret >= 0 else ""
    history_table_rows += f"""
    <tr>
        <td>{date_str}</td>
        <td>${nav_val:,.0f}</td>
        <td style="color:{color};">{sign}{day_ret:.2f}%</td>
    </tr>"""

# ─── 交易損益明細：解析成表格 ────────────────────────────────────────────────
SELL_RE = re.compile(
    r'賣出\s+(\S+)\s+(.+?)\s+(\d+)股成本@([0-9.]+)\s+賣@([0-9.]+)\s+實現([+-][0-9,]+)')
BUY_RE = re.compile(
    r'買進\s+(\S+)\s+(.+?)\s+(\d+)股\s+@([0-9.]+)\s+成本\s+([0-9,]+)')

def parse_logs(logs):
    trades = []
    for log in logs:
        m = SELL_RE.match(log)
        if m:
            trades.append({
                'dir': 'sell',
                'stock_id': m.group(1),
                'name': m.group(2),
                'shares': int(m.group(3)),
                'cost_price': float(m.group(4)),
                'trade_price': float(m.group(5)),
                'pnl': int(m.group(6).replace(',', '')),
            })
            continue
        m = BUY_RE.match(log)
        if m:
            trades.append({
                'dir': 'buy',
                'stock_id': m.group(1),
                'name': m.group(2),
                'shares': int(m.group(3)),
                'trade_price': float(m.group(4)),
                'pnl': None,
            })
    return trades

def render_trade_table(trades, raw_logs):
    if not trades:
        return "".join(
            f'<tr><td colspan="7" style="color:{TV_MUTED}; padding:8px 12px;">{l}</td></tr>'
            for l in raw_logs
        )
    rows = ""
    for t in trades:
        if t['dir'] == 'sell':
            dir_cell = '<span class="dir-sell">賣出</span>'
            cost_cell = f"{t['cost_price']:.2f}"
            pnl = t['pnl']
            pc = TV_GREEN if pnl >= 0 else TV_RED
            ps = '+' if pnl >= 0 else ''
            pnl_cell = f'<span style="color:{pc};">{ps}{pnl:,}</span>'
        else:
            dir_cell = '<span class="dir-buy">買進</span>'
            cost_cell = "—"
            pnl_cell = "—"
        rows += f"""
        <tr>
            <td>{dir_cell}</td>
            <td class="td-code">{t['stock_id']}</td>
            <td>{t['name']}</td>
            <td class="td-num">{t['shares']:,}</td>
            <td class="td-num">{cost_cell}</td>
            <td class="td-num">{t['trade_price']:.2f}</td>
            <td class="td-num">{pnl_cell}</td>
        </tr>"""
    return rows

trade_detail_html = ""
if trade_log_history:
    for idx, entry in enumerate(reversed(trade_log_history[-10:])):
        d         = entry.get("date", "")
        pnl       = entry.get("realized_pnl", entry.get("pnl", None))
        logs      = entry.get("logs", [])
        pnl_color = TV_GREEN if pnl and pnl >= 0 else TV_RED
        pnl_sign  = "+" if pnl and pnl >= 0 else ""
        pnl_str   = (f'<span style="color:{pnl_color};">{pnl_sign}{pnl:,.0f}</span>') if pnl is not None else ""
        trades    = parse_logs(logs)
        rows      = render_trade_table(trades, logs)
        open_attr = "open" if idx == 0 else ""
        trade_detail_html += f"""
        <details {open_attr} class="trade-entry-details">
            <summary class="trade-summary">
                <span class="trade-date-label">{d}</span>
                <span class="trade-pnl-label">實現損益:&nbsp;{pnl_str}</span>
                <span class="summary-arrow"></span>
            </summary>
            <div class="trade-table-wrap">
                <table class="trade-table">
                    <thead><tr>
                        <th>方向</th><th>代碼</th><th>名稱</th>
                        <th class="th-num">股數</th>
                        <th class="th-num">成本價</th>
                        <th class="th-num">成交價</th>
                        <th class="th-num">實現損益</th>
                    </tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </details>"""
else:
    trade_detail_html = f'<p style="color:{TV_MUTED}; padding:16px; font-size:12px;">尚無交易紀錄</p>'

# ─── 格式化指標 ────────────────────────────────────────────────────────────────
daily_color = TV_GREEN if daily_ret >= 0 else TV_RED
total_color = TV_GREEN if total_ret >= 0 else TV_RED
daily_sign  = "+" if daily_ret >= 0 else ""
total_sign  = "+" if total_ret >= 0 else ""

# ─── CSS ──────────────────────────────────────────────────────────────────────
DASHBOARD_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
body { background: #131722; color: #D1D4DC; padding: 16px;
       font-size: 14px; min-height: 100vh; }

.grid-main { display: grid; grid-template-columns: 2fr 1fr; gap: 16px;
             max-width: 1600px; margin: 0 auto 16px; }
.header-bar { grid-column: 1 / -1; }
.grid-bottom { display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
               max-width: 1600px; margin: 0 auto; }

@media (max-width: 1024px) {
    .grid-main { grid-template-columns: 1fr; }
    .grid-bottom { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
    body { padding: 8px; font-size: 13px; }
    .metric-val { font-size: 18px !important; }
    .header-right { flex-direction: column; gap: 10px !important; }
    .col-right:nth-child(3) { display: none; }
    .chart-panel, .right-sidebar { height: auto; min-height: 400px; }
}

.panel { background: #1E222D; border-radius: 6px; border: 1px solid #2A2E39;
         display: flex; flex-direction: column; overflow: hidden; }
.panel-header { padding: 10px 16px; font-size: 11px; font-weight: 600;
                color: #8D94A6; border-bottom: 1px solid #2A2E39;
                text-transform: uppercase; letter-spacing: 0.5px; flex-shrink: 0; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #363C4E; border-radius: 3px; }

/* Top Bar */
.top-bar-content { padding: 14px 24px; display: flex;
                   justify-content: space-between; align-items: center; }
.metric-group { display: flex; flex-direction: column; }
.metric-title { color: #8D94A6; font-size: 11px; margin-bottom: 3px;
                text-transform: uppercase; letter-spacing: 0.5px; }
.metric-val { font-size: 22px; font-weight: 700; }
.metric-sub { font-size: 13px; font-weight: 600; margin-left: 10px; }
.header-right { display: flex; gap: 40px; text-align: right; }

/* Chart Panel */
.chart-panel { height: calc(100vh - 196px); min-height: 500px;
               display: flex; flex-direction: column; }

/* Tab Bar */
.tab-bar { display: flex; border-bottom: 1px solid #2A2E39;
           flex-shrink: 0; background: #161B27; }
.tab-btn { padding: 9px 18px; background: transparent; border: none;
           border-bottom: 2px solid transparent; color: #8D94A6;
           font-size: 12px; font-weight: 600; cursor: pointer;
           font-family: 'Inter', sans-serif; transition: color 0.15s, border-color 0.15s;
           white-space: nowrap; }
.tab-btn:hover { color: #D1D4DC; }
.tab-btn.active { color: #D1D4DC; border-bottom-color: #2962FF; }

.tab-content { flex: 1; min-height: 0; display: none; flex-direction: column; }
.tab-content.active { display: flex; }
#tab-portfolio > div { flex: 1; min-height: 0; }

/* Stock Chart Tab */
.chart-toolbar { display: flex; align-items: center; gap: 12px; padding: 8px 14px;
                 border-bottom: 1px solid #2A2E39; flex-shrink: 0; flex-wrap: wrap; }
.range-btns, .indicator-btns { display: flex; gap: 4px; }
.tool-divider { width: 1px; height: 20px; background: #2A2E39; align-self: center; }
.tool-btn { padding: 4px 10px; background: transparent; border: 1px solid #2A2E39;
            border-radius: 3px; color: #8D94A6; font-size: 11px; font-weight: 600;
            cursor: pointer; font-family: 'Inter', sans-serif; transition: all 0.15s;
            letter-spacing: 0.3px; }
.tool-btn:hover { border-color: #4C5468; color: #D1D4DC; }
.tool-btn.active { background: #2962FF; border-color: #2962FF; color: #fff; }

.stock-info-bar { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
                  padding: 8px 14px; border-bottom: 1px solid #2A2E39; flex-shrink: 0; }
.stock-info-name { font-size: 15px; font-weight: 700; color: #D1D4DC; }
.stock-info-sub { font-size: 12px; color: #8D94A6; }
.stock-info-price { font-size: 20px; font-weight: 700; }
.stock-info-change { font-size: 13px; font-weight: 600; }

#stock-chart-container { flex: 1; min-height: 0; position: relative; }
#chart-placeholder { display: flex; align-items: center; justify-content: center;
                     height: 100%; color: #8D94A6; font-size: 13px;
                     flex-direction: column; gap: 10px; }
.placeholder-hint { font-size: 11px; opacity: 0.6; }
#chart-loading { position: absolute; inset: 0; display: none; align-items: center;
                 justify-content: center; background: rgba(30,34,45,0.85);
                 color: #8D94A6; font-size: 13px; z-index: 10; }

/* Right Sidebar */
.right-sidebar { display: flex; flex-direction: column;
                 height: calc(100vh - 196px); min-height: 500px; }
.donut-wrapper { padding: 12px 16px 4px; border-bottom: 1px solid #2A2E39; flex-shrink: 0; }
.donut-container { position: relative; width: 185px; height: 185px; margin: 0 auto 4px; }
.donut-label { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
               text-align: center; pointer-events: none; }
.donut-val { font-size: 22px; font-weight: 700; color: #D1D4DC; }
.donut-sub { font-size: 10px; color: #8D94A6; text-transform: uppercase;
             margin-top: 2px; letter-spacing: 0.5px; }

/* Holdings */
.tbl-header { display: flex; padding: 7px 14px; color: #8D94A6; font-size: 10px;
              border-bottom: 1px solid #2A2E39; text-transform: uppercase;
              letter-spacing: 0.3px; flex-shrink: 0; }
.table-row { display: flex; padding: 9px 14px; border-bottom: 1px solid #2A2E39;
             font-size: 12px; align-items: center; transition: background 0.15s; }
.holding-row { cursor: pointer; }
.holding-row:hover { background: rgba(41,98,255,0.08); }
.holding-row.active-holding { background: rgba(41,98,255,0.12);
    border-left: 2px solid #2962FF; padding-left: 12px; }
.col-ticker { flex: 1.5; font-weight: 600; display: flex; flex-direction: column; gap: 1px; }
.stock-id { font-size: 13px; }
.stock-name { font-size: 10px; color: #8D94A6; font-weight: 400; }
.col-right { flex: 1; text-align: right; }
.col-pnl { flex: 1.2; text-align: right; }
.holdings-scroll { flex: 1; overflow-y: auto; }

/* Bottom Tables */
.history-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.history-table th { padding: 7px 14px; text-align: left; background: #161d2b;
                    color: #8D94A6; font-size: 10px; text-transform: uppercase;
                    letter-spacing: 0.3px; border-bottom: 1px solid #2A2E39;
                    position: sticky; top: 0; z-index: 1; }
.history-table td { padding: 8px 14px; border-bottom: 1px solid #2A2E39; }
.history-table tr:last-child td { border-bottom: none; }
.history-table tr:hover td { background: rgba(255,255,255,0.02); }
.scroll-body { overflow-y: auto; max-height: 300px; }

/* Trade Log */
.trade-entry-details { border-bottom: 1px solid #2A2E39; }
.trade-entry-details:last-child { border-bottom: none; }
.trade-summary { display: flex; align-items: center; gap: 10px; padding: 10px 14px;
                 cursor: pointer; list-style: none; font-size: 12px;
                 user-select: none; transition: background 0.15s; }
.trade-summary::-webkit-details-marker { display: none; }
.trade-summary:hover { background: rgba(255,255,255,0.02); }
.trade-date-label { font-weight: 600; color: #D1D4DC; }
.trade-pnl-label { color: #8D94A6; font-size: 11px; }
.summary-arrow { margin-left: auto; color: #8D94A6; font-size: 10px; transition: transform 0.2s; }
.summary-arrow::after { content: '\25B6'; }
details[open] .summary-arrow::after { content: '\25BC'; }
.trade-table-wrap { overflow-x: auto; }
.trade-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.trade-table th { padding: 6px 12px; text-align: left; background: #161B27;
                  color: #8D94A6; font-size: 10px; text-transform: uppercase;
                  letter-spacing: 0.3px; border-bottom: 1px solid #2A2E39;
                  white-space: nowrap; }
.th-num { text-align: right !important; }
.trade-table td { padding: 7px 12px; border-bottom: 1px solid rgba(42,46,57,0.6); color: #D1D4DC; }
.trade-table tr:last-child td { border-bottom: none; }
.trade-table tr:hover td { background: rgba(255,255,255,0.02); }
.td-num { text-align: right; }
.td-code { font-weight: 600; }
.dir-sell { color: #F05350; font-weight: 600; }
.dir-buy  { color: #22AB94; font-weight: 600; }
"""

# ─── JavaScript ───────────────────────────────────────────────────────────────
DASHBOARD_JS = """
// ── State ────────────────────────────────────────────────────────────────────
const chartState = {
    stockId: null, stockName: null, range: '3mo', rawData: null,
    indicators: { ma5: true, ma20: true, ma60: false, bb: false, rsi: false, macd: false }
};

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('tab-btn-' + name).classList.add('active');
    document.getElementById('tab-' + name).classList.add('active');
    if (name === 'portfolio') {
        document.querySelectorAll('.holding-row').forEach(r => r.classList.remove('active-holding'));
        chartState.stockId = null;
    }
    if (name === 'stock' && chartState.rawData) {
        setTimeout(() => renderStockChart(), 50);
    }
}

// ── Load stock chart ──────────────────────────────────────────────────────────
function loadStockChart(stockId, stockName) {
    chartState.stockId = stockId;
    chartState.stockName = stockName;

    document.querySelectorAll('.holding-row').forEach(r => r.classList.remove('active-holding'));
    const row = document.querySelector('[data-stock="' + stockId + '"]');
    if (row) row.classList.add('active-holding');

    document.getElementById('stock-tab-label').textContent = stockId + ' ' + stockName;
    document.getElementById('tab-btn-stock').style.display = '';
    switchTab('stock');

    const data = getStockData(stockId, chartState.range);
    if (!data) {
        document.getElementById('chart-loading').style.display = 'none';
        document.getElementById('chart-placeholder').style.display = 'flex';
        document.getElementById('chart-placeholder').innerHTML =
            '<div style="font-size:28px;opacity:0.3;">!</div>' +
            '<div>\u672a\u627e\u5230 ' + stockId + ' \u7684\u8cc7\u6599</div>';
        return;
    }
    chartState.rawData = data;
    document.getElementById('chart-placeholder').style.display = 'none';
    document.getElementById('chart-loading').style.display = 'none';
    updateStockInfoBar(data);
    renderStockChart();
    document.getElementById('chart-toolbar').style.display = 'flex';
    document.getElementById('stock-info-bar').style.display = 'flex';
}

// ── Embedded OHLCV data (injected by Python) ─────────────────────────────────
const STOCK_DATA = STOCK_OHLCV_PLACEHOLDER;

// Convert embedded OHLCV to Yahoo-like format with optional range slicing
function getStockData(stockId, range) {
    const raw = STOCK_DATA[stockId];
    if (!raw || !raw.dates || raw.dates.length === 0) return null;

    // Determine cutoff date based on range
    const rangeDays = { '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365 };
    const days = rangeDays[range] || 365;
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - days);
    const cutoffStr = cutoff.toISOString().slice(0, 10);

    const idx = raw.dates.findIndex(d => d >= cutoffStr);
    const start = Math.max(0, idx);

    return {
        dates:   raw.dates.slice(start),
        opens:   raw.opens.slice(start),
        highs:   raw.highs.slice(start),
        lows:    raw.lows.slice(start),
        closes:  raw.closes.slice(start),
        volumes: raw.volumes.slice(start),
    };
}

// ── Update info bar ───────────────────────────────────────────────────────────
function updateStockInfoBar(data) {
    const closes = data.closes || [];
    const last = closes.slice(-1)[0];
    const prev = closes.slice(-2)[0];
    const chg = (last && prev) ? last - prev : 0;
    const pct = (last && prev) ? (chg / prev * 100) : 0;
    const color = chg >= 0 ? '#22AB94' : '#F05350';
    const sign  = chg >= 0 ? '+' : '';
    document.getElementById('info-name').textContent = chartState.stockName;
    document.getElementById('info-sub').textContent = chartState.stockId + '.TW';
    document.getElementById('info-price').textContent = last ? last.toFixed(2) : '\u2014';
    document.getElementById('info-price').style.color = color;
    document.getElementById('info-change').textContent = last
        ? sign + chg.toFixed(2) + '  (' + sign + pct.toFixed(2) + '%)' : '';
    document.getElementById('info-change').style.color = color;
}

// ── Indicator math ────────────────────────────────────────────────────────────
function calcMA(arr, n) {
    return arr.map((_, i) => i < n-1 ? null : arr.slice(i-n+1,i+1).reduce((a,b)=>a+(b||0),0)/n);
}
function calcEMA(arr, n) {
    const k = 2/(n+1); const out = [];
    let e = null;
    for (const v of arr) {
        if (v == null) { out.push(null); continue; }
        e = (e === null) ? v : v*k + e*(1-k);
        out.push(e);
    }
    return out;
}
function calcBB(closes, n=20, mult=2) {
    const mid = calcMA(closes, n);
    const upper = [], lower = [];
    for (let i=0; i<closes.length; i++) {
        if (mid[i] === null) { upper.push(null); lower.push(null); continue; }
        const sl = closes.slice(Math.max(0,i-n+1), i+1).filter(v=>v!=null);
        const mean = sl.reduce((a,b)=>a+b,0)/sl.length;
        const std = Math.sqrt(sl.reduce((a,b)=>a+(b-mean)**2,0)/sl.length);
        upper.push(mid[i] + mult*std);
        lower.push(mid[i] - mult*std);
    }
    return { mid, upper, lower };
}
function calcRSI(closes, n=14) {
    const rsi = new Array(closes.length).fill(null);
    let avgG=0, avgL=0;
    for (let i=1; i<closes.length; i++) {
        const d = (closes[i]||0) - (closes[i-1]||0);
        if (i <= n) { avgG += Math.max(d,0)/n; avgL += Math.max(-d,0)/n; }
        else {
            avgG = (avgG*(n-1)+Math.max(d,0))/n;
            avgL = (avgL*(n-1)+Math.max(-d,0))/n;
            rsi[i] = avgL===0 ? 100 : 100 - 100/(1+avgG/avgL);
        }
    }
    return rsi;
}
function calcMACD(closes) {
    const ema12 = calcEMA(closes, 12);
    const ema26 = calcEMA(closes, 26);
    const dif = ema12.map((v,i) => (v!=null && ema26[i]!=null) ? v-ema26[i] : null);
    const dea = calcEMA(dif, 9);
    const hist = dif.map((v,i) => (v!=null && dea[i]!=null) ? v-dea[i] : null);
    return { dif, dea, hist };
}

// ── Render stock chart ────────────────────────────────────────────────────────
function renderStockChart() {
    document.getElementById('chart-loading').style.display = 'none';
    const data    = chartState.rawData;
    const ts      = data.dates   || [];
    const opens   = data.opens   || [];
    const highs   = data.highs   || [];
    const lows    = data.lows    || [];
    const closes  = data.closes  || [];
    const volumes = data.volumes || [];
    const ind = chartState.indicators;

    // Determine subplot layout
    const showRSI  = ind.rsi;
    const showMACD = ind.macd;
    const panels = 1 + (showRSI?1:0) + (showMACD?1:0); // main+volume is always 1 combined panel
    const domainGap = 0.03;
    let domains;
    if (panels === 1)      domains = { main:[0.22,1], vol:[0,0.20] };
    else if (panels === 2) domains = { main:[0.36,1], vol:[0.24,0.34], rsi:showRSI?[0,0.22]:null, macd:showMACD?[0,0.22]:null };
    else                   domains = { main:[0.48,1], vol:[0.36,0.46], rsi:[0.12,0.34], macd:[0,0.10] };

    const traces = [];

    // Candlestick
    traces.push({ type:'candlestick', x:ts, open:opens, high:highs, low:lows, close:closes,
        increasing:{line:{color:'#22AB94'},fillcolor:'#22AB94'},
        decreasing:{line:{color:'#F05350'},fillcolor:'#F05350'},
        name:'K線', xaxis:'x', yaxis:'y', showlegend:false,
        hoverinfo:'x+y' });

    // MAs
    const maColors = { ma5:'#F0B429', ma20:'#2962FF', ma60:'#E040FB' };
    for (const [key, n] of [['ma5',5],['ma20',20],['ma60',60]]) {
        if (!ind[key]) continue;
        const vals = calcMA(closes, n);
        traces.push({ type:'scatter', x:ts, y:vals, mode:'lines', name:key.toUpperCase(),
            line:{color:maColors[key], width:1.2}, xaxis:'x', yaxis:'y', showlegend:false });
    }

    // Bollinger Bands
    if (ind.bb) {
        const bb = calcBB(closes);
        const fillX = [...ts, ...ts.slice().reverse()];
        const fillY = [...bb.upper, ...bb.lower.slice().reverse()];
        traces.push({ type:'scatter', x:fillX, y:fillY, fill:'toself',
            fillcolor:'rgba(41,98,255,0.06)', line:{color:'transparent'},
            showlegend:false, hoverinfo:'skip', xaxis:'x', yaxis:'y' });
        traces.push({ type:'scatter', x:ts, y:bb.upper, mode:'lines', name:'BB Upper',
            line:{color:'rgba(41,98,255,0.5)', width:1, dash:'dot'},
            xaxis:'x', yaxis:'y', showlegend:false });
        traces.push({ type:'scatter', x:ts, y:bb.lower, mode:'lines', name:'BB Lower',
            line:{color:'rgba(41,98,255,0.5)', width:1, dash:'dot'},
            xaxis:'x', yaxis:'y', showlegend:false });
    }

    // Volume
    const volColors = volumes.map((_,i) => (closes[i]||0) >= (opens[i]||0) ? '#1B5E4B' : '#5C1E1E');
    traces.push({ type:'bar', x:ts, y:volumes, name:'Volume', marker:{color:volColors},
        xaxis:'x', yaxis:'y2', showlegend:false });

    // RSI
    if (showRSI) {
        const rsi = calcRSI(closes);
        traces.push({ type:'scatter', x:ts, y:rsi, mode:'lines', name:'RSI',
            line:{color:'#7E57C2', width:1.5}, xaxis:'x', yaxis:'y3', showlegend:false });
        traces.push({ type:'scatter', x:[ts[0],ts[ts.length-1]], y:[70,70], mode:'lines',
            line:{color:'rgba(240,83,80,0.4)', width:1, dash:'dot'},
            xaxis:'x', yaxis:'y3', showlegend:false, hoverinfo:'skip' });
        traces.push({ type:'scatter', x:[ts[0],ts[ts.length-1]], y:[30,30], mode:'lines',
            line:{color:'rgba(34,171,148,0.4)', width:1, dash:'dot'},
            xaxis:'x', yaxis:'y3', showlegend:false, hoverinfo:'skip' });
    }

    // MACD
    if (showMACD) {
        const macd = calcMACD(closes);
        const macdYaxis = showRSI ? 'y4' : 'y3';
        const macdXaxis = 'x';
        const histColors = macd.hist.map(v => (v||0) >= 0 ? 'rgba(34,171,148,0.7)' : 'rgba(240,83,80,0.7)');
        traces.push({ type:'bar', x:ts, y:macd.hist, name:'MACD Hist',
            marker:{color:histColors}, xaxis:macdXaxis, yaxis:macdYaxis, showlegend:false });
        traces.push({ type:'scatter', x:ts, y:macd.dif, mode:'lines', name:'DIF',
            line:{color:'#2962FF', width:1.2}, xaxis:macdXaxis, yaxis:macdYaxis, showlegend:false });
        traces.push({ type:'scatter', x:ts, y:macd.dea, mode:'lines', name:'DEA',
            line:{color:'#F0B429', width:1.2}, xaxis:macdXaxis, yaxis:macdYaxis, showlegend:false });
    }

    const sharedAxis = { showgrid:true, gridcolor:'#2B3139', tickcolor:'#8D94A6',
                          zeroline:false, fixedrange:false };
    const layout = {
        plot_bgcolor:'#1E222D', paper_bgcolor:'#1E222D',
        font:{color:'#D1D4DC'}, showlegend:false,
        margin:{l:12, r:60, t:8, b:40},
        dragmode:'pan',
        hovermode:'x unified',
        xaxis: { ...sharedAxis, domain:[0,1], anchor:'y2',
                 rangeslider:{visible:false}, type:'category',
                 tickangle:-45, tickfont:{size:10} },
        yaxis:  { ...sharedAxis, domain:domains.main, side:'right', tickformat:'.1f' },
        yaxis2: { ...sharedAxis, domain:domains.vol,  side:'right', showticklabels:false },
    };
    if (showRSI) {
        layout.yaxis3 = { ...sharedAxis, domain:domains.rsi, side:'right',
                          range:[0,100], tickvals:[30,70], ticktext:['30','70'], tickfont:{size:9} };
    }
    if (showMACD) {
        const macdYkey = showRSI ? 'yaxis4' : 'yaxis3';
        const macdDomain = showRSI ? domains.macd : (showRSI ? domains.rsi : domains.rsi || [0,0.22]);
        layout[macdYkey] = { ...sharedAxis, domain: showRSI ? domains.macd : [0,0.22],
                              side:'right', tickfont:{size:9} };
    }

    const container = document.getElementById('stock-chart-container');
    Plotly.newPlot(container, traces, layout,
        { displayModeBar:true, scrollZoom:false, responsive:true,
          modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d'] });
}

// ── Range / Indicator toggles ─────────────────────────────────────────────────
function setRange(r) {
    chartState.range = r;
    document.querySelectorAll('.range-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.range === r);
    });
    if (chartState.stockId) {
        chartState.rawData = getStockData(chartState.stockId, r);
        if (chartState.rawData) {
            updateStockInfoBar(chartState.rawData);
            renderStockChart();
        }
    }
}

function toggleIndicator(key) {
    chartState.indicators[key] = !chartState.indicators[key];
    const btn = document.getElementById('ind-btn-' + key);
    if (btn) btn.classList.toggle('active', chartState.indicators[key]);
    if (chartState.rawData) renderStockChart();
}

window.addEventListener('resize', () => {
    if (chartState.rawData) {
        const c = document.getElementById('stock-chart-container');
        if (c._fullLayout) Plotly.relayout(c, {});
    }
});
"""

# ─── 組合完整 HTML ─────────────────────────────────────────────────────────────
html_content = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
    <title>台股ML虛擬基金 Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>{DASHBOARD_CSS}</style>
</head>
<body>

<div class="grid-main">
    <!-- Top Bar -->
    <div class="header-bar panel">
        <div class="top-bar-content">
            <div class="metric-group">
                <span class="metric-title">模擬交易組合</span>
                <div>
                    <span class="metric-val">${current_nav:,.0f}</span>
                    <span class="metric-sub" style="color:{daily_color};">{daily_sign}{daily_ret*100:.2f}% (1D)</span>
                    <span class="metric-sub" style="color:{total_color}; margin-left:12px;">{total_sign}{total_ret*100:.2f}% (All)</span>
                </div>
            </div>
            <div class="header-right">
                <div class="metric-group">
                    <span class="metric-title">Equity</span>
                    <span style="font-size:15px;font-weight:600;">${total_equity:,.0f}</span>
                </div>
                <div class="metric-group">
                    <span class="metric-title">Cash</span>
                    <span style="font-size:15px;font-weight:600;">${cash:,.0f}</span>
                </div>
                <div class="metric-group" style="text-align:right;">
                    <span class="metric-title">Market Date</span>
                    <span style="font-size:15px;font-weight:600;">{latest_date}</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Chart Panel with Tabs -->
    <div class="panel chart-panel">
        <div class="tab-bar">
            <button class="tab-btn active" id="tab-btn-portfolio" onclick="switchTab('portfolio')">Portfolio Curve</button>
            <button class="tab-btn" id="tab-btn-stock" onclick="switchTab('stock')" style="display:none;">
                <span id="stock-tab-label">— 走勢</span>
            </button>
        </div>

        <!-- Tab: Portfolio NAV -->
        <div class="tab-content active" id="tab-portfolio">
            <div style="padding:0; min-height:0;">
                {html_line}
            </div>
        </div>

        <!-- Tab: Stock Chart -->
        <div class="tab-content" id="tab-stock">
            <!-- Toolbar -->
            <div class="chart-toolbar" id="chart-toolbar" style="display:none;">
                <div class="range-btns">
                    <button class="tool-btn range-btn" data-range="1mo" onclick="setRange('1mo')">1M</button>
                    <button class="tool-btn range-btn active" data-range="3mo" onclick="setRange('3mo')">3M</button>
                    <button class="tool-btn range-btn" data-range="6mo" onclick="setRange('6mo')">6M</button>
                    <button class="tool-btn range-btn" data-range="1y" onclick="setRange('1y')">1Y</button>
                </div>
                <div class="tool-divider"></div>
                <div class="indicator-btns">
                    <button class="tool-btn active" id="ind-btn-ma5"  onclick="toggleIndicator('ma5')">MA5</button>
                    <button class="tool-btn active" id="ind-btn-ma20" onclick="toggleIndicator('ma20')">MA20</button>
                    <button class="tool-btn" id="ind-btn-ma60" onclick="toggleIndicator('ma60')">MA60</button>
                    <button class="tool-btn" id="ind-btn-bb"   onclick="toggleIndicator('bb')">BB</button>
                    <button class="tool-btn" id="ind-btn-rsi"  onclick="toggleIndicator('rsi')">RSI</button>
                    <button class="tool-btn" id="ind-btn-macd" onclick="toggleIndicator('macd')">MACD</button>
                </div>
            </div>

            <!-- Stock Info Bar -->
            <div class="stock-info-bar" id="stock-info-bar" style="display:none;">
                <span class="stock-info-name" id="info-name"></span>
                <span class="stock-info-sub" id="info-sub"></span>
                <span class="stock-info-price" id="info-price"></span>
                <span class="stock-info-change" id="info-change"></span>
            </div>

            <!-- Chart + Loading -->
            <div id="stock-chart-container">
                <div id="chart-loading">載入中...</div>
                <div id="chart-placeholder">
                    <div style="font-size:32px; opacity:0.25;">K</div>
                    <div>點擊右側持股查看 K 線走勢</div>
                    <div class="placeholder-hint">支援 MA / BB / RSI / MACD</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Right Sidebar -->
    <div class="panel right-sidebar">
        <div class="panel-header">Portfolio Allocation</div>

        <div class="donut-wrapper">
            <div class="donut-container">
                {html_pie}
                <div class="donut-label">
                    <div class="donut-val">{len(stock_data)}</div>
                    <div class="donut-sub">Positions</div>
                </div>
            </div>
        </div>

        <div class="panel-header" style="border-top:none;">Current Holdings</div>
        <div class="tbl-header">
            <div class="col-ticker">Symbol / 名稱</div>
            <div class="col-right">Price</div>
            <div class="col-right">Shares</div>
            <div class="col-right">Value</div>
            <div class="col-right">Weight</div>
            <div class="col-right col-pnl">PnL</div>
        </div>
        <div class="holdings-scroll">
            {holdings_html}
        </div>
    </div>
</div>

<!-- Bottom: NAV History + Trade Log -->
<div class="grid-bottom">
    <div class="panel">
        <div class="panel-header">每日淨值歷史（最近30天）</div>
        <div class="scroll-body">
            <table class="history-table">
                <thead><tr><th>日期</th><th>淨值 (NAV)</th><th>當日報酬</th></tr></thead>
                <tbody>{history_table_rows}</tbody>
            </table>
        </div>
    </div>

    <div class="panel">
        <div class="panel-header">交易紀錄 &amp; 損益明細</div>
        <div class="scroll-body">
            {trade_detail_html}
        </div>
    </div>
</div>

<script>{DASHBOARD_JS.replace("STOCK_OHLCV_PLACEHOLDER", stock_ohlcv_json)}</script>
</body>
</html>
"""

os.makedirs("frontend", exist_ok=True)
with open("frontend/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Dashboard 生成完成 -> frontend/index.html")
print(f"   持股: {len(stock_data)} 支，歷史: {len(history)} 天")
