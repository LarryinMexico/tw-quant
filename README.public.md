# 台股 ML 量化選股系統 · tw-quant

> 🤖 **LightGBM Walk-Forward** × 📈 **動能 + 籌碼 13 因子** × ⚡ **GitHub Actions 全自動每日更新**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/market-台灣股市-red)](#)

基於 Python 與 Machine Learning 的台股量化選股系統。  
採用 **FinMind** 作為免費資料來源、**vectorbt** 進行回測、**LightGBM Walk-Forward** 進行選股，並透過 **GitHub Actions** 實現盤後全自動化。

---

## 📐 系統架構

```
 FinMind API ──► data_loaders/ ──► strategy.py ──► weights.pkl
      │                                 │               │
      │            Walk-Forward LightGBM│               ▼
      │            13 因子 × 1800 支股票│       live_trade.py
      │                                 │          │        │
 yfinance (每日) ─────────────────────────   portfolio.json  LINE 通知
                                                     │
                                             GitHub Pages Dashboard

 GitHub Actions 每日 15:30 (台灣時間) 自動執行上述全流程
```

---

## ⚡ Quick Start（5 分鐘跑起來）

### 1. 環境準備

```bash
git clone https://github.com/LarryinMexico/tw-quant.git
cd tw-quant

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 設定 API Token

```bash
cp .env.example .env
# 用文字編輯器打開 .env，填入你的 FinMind API Token
```

> FinMind 免費帳號：[https://finmindtrade.com/](https://finmindtrade.com/)  
> 免費方案每小時 600 次 API 請求，下載歷史資料約需 4~6 小時。

### 3. 下載歷史資料

```bash
python data_loaders/01_fetch_finmind_data.py
# 約 4~6 小時（FinMind 免費 API 有速率限制，程式內建自動睡眠）
```

### 4. 訓練模型 & 回測

```bash
python strategy.py
# 約 10~20 分鐘，完成後自動生成 predictions.pkl 與 weights.pkl
```

### 5. 生成回測報告

```bash
python reports/generate_report.py
# 開啟 reports/backtest_report.html 查看 14 張互動圖表
```

---

## 🎛️ 核心參數調整指南

所有核心參數集中在 `strategy.py` 頂部的 `# Config` 區塊，修改後重新執行 `python strategy.py` 即可。

### strategy.py — 模型與策略參數

| 參數 | 預設值 | 說明 | 調整建議 |
|------|--------|------|----------|
| `TRAIN_MONTHS` | `48` | Walk-Forward 訓練視窗（月） | 市場變化快 → 縮短至 `36`；穩定市場 → 延長至 `60` |
| `STEP_MONTHS` | `3` | 每隔幾個月重新訓練一次 | 改 `1` = 每月 retrain（更即時但更慢），`6` = 半年 retrain |
| `TOP_K` | `40` | 同時持有幾支股票 | 集中度高 → `20`（高風險高報酬）；更分散 → `60` |
| `WEIGHT_TEMP` | `5.0` | Softmax 溫度（資金分配集中度） | 接近等權重。`1.0` = 高度集中前幾名；`10.0` = 更均勻 |
| `LIQUIDITY_MIN` | `30_000_000` | 30 日均量門檻（元）| 降至 `10_000_000` 可納入更多小型股；提高可只選大型股 |
| `TOP_FACTORS_K` | `8` | 每次 fold 自動選用幾個因子 | 保守 → `6`；激進（用全部）→ `13` |
| `INIT_CASH` | `1_000_000` | 回測初始資金（元）| 依個人資金規模調整（僅影響回測顯示，不影響策略邏輯） |
| `TEST_START` | `"2019-01-01"` | 回測起始日 | 依你下載的歷史資料範圍調整 |

#### 換倉頻率調整（月度 vs. 週度）

預設為**月度換倉**（每月底換股）。如果你想提高換倉頻率：

```python
# strategy.py - 調整 resample 週期
# 月度（預設）：.resample("ME").last()
# 週度：       .resample("W").last()
# 雙週度：     .resample("2W").last()

def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("ME").last()   # ← 改成 "W" 即為週度換倉
```

> ⚠️ 注意：換倉頻率越高 → 手續費摩擦成本越高 → CAGR 可能下降。  
> 建議同時調高 `FEE` 的估算以反映更高頻交易的真實成本。

#### 大盤濾網調整（Regime Filter）

```python
# strategy.py
# 目前：2 of 3 訊號同意才進場（score >= 0.4）
# 保守（只有多頭才進）：改成 >= 0.7
# 激進（幾乎全時間在場）：改成 >= 0.1
is_bull_daily = regime_score >= 0.4   # ← 在此行調整門檻
```

### live_trade.py — 虛擬交易參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `MAX_INVEST_RATIO` | `0.90` | 最多投入 90% 資金（保留 10% 現金緩衝），可調至 `0.95` |
| `MAX_SINGLE_WEIGHT` | `0.08` | 單支股票最多 8% 倉位，降至 `0.05` 可更分散 |
| `FEE_RATE` | `0.001425` | 手續費率（0.1425%），若有打折可調低 |
| `SLIPPAGE` | `0.001` | 滑價估算（0.1%），流動性差的小型股可調至 `0.002` |
| `GITHUB_PAGES_URL` | 你的網址 | LINE 通知中的 Dashboard 連結，改成你自己的 GitHub Pages URL |

### 初始虛擬資金設定

首次使用時，修改 `portfolio.json`（從 `portfolio.example.json` 複製）：

```json
{
  "cash": 1000000,
  "comment": "初始現金 100 萬，依個人資金規模調整"
}
```

---

## 🎯 策略邏輯

### 1. 大盤安全濾網（0050 多空燈）

每日監控大盤趨勢，3 個訊號加權判斷：
- **60 日均線**（季線）：0050 > MA60 → 長期多頭 × 0.4
- **20 日均線**：0050 > MA20 → 中期多頭 × 0.3  
- **無大跌**：0050 月跌幅 < -5% → 無崩盤 × 0.3

≥ 2 訊號同意才進場；任何月份若觸發空頭，強制 100% 出清變現。

### 2. 13 個選股因子

| 類別 | 因子 | 說明 | AI 重要性排名 |
|------|------|------|:-----------:|
| 籌碼 | `trust_net_10d` | 投信近 10 日淨買超比例 | 🥇 1 |
| 技術 | `rsi_14` | 14 日 RSI（動能延續性） | 🥈 2 |
| 波動 | `atr_rel` | 相對 ATR（波動放大 = 起漲前兆） | 🥉 3 |
| 動能 | `price_52w` | 股價與 52 週高點距離 | 4 |
| 動能 | `mom_1m` | 近 1 月報酬率 | 5 |
| 基本 | `rev_yoy` | 月營收年增率 | 6 |
| 基本 | `rev_mom` | 月營收月增率 | 7 |
| 動能 | `mom_1m_ra` | 風險調整後動能 | 8 |
| 動能 | `mom_3m` | 近 3 月報酬率 | 9 |
| 動能 | `mom_6m` | 近 6 月報酬率 | 10 |
| 技術 | `vol_ratio` | 成交量增溫速率 | 11 |
| 基本 | `rev_accel` | 營收成長加速度 | 12 |
| 籌碼 | `foreign_net_20d` | 外資近 20 日淨買超比例 | 13 |

> 每次 Walk-Forward retrain 時，LightGBM 在訓練集內透過 **Fold-Internal ICIR 分析**自動選出最穩定的 Top-8 因子進行預測（避免 Lookahead Bias）。

### 3. 資金分配與換倉慣性

- **持股數量**：前 `TOP_K=40` 名（可調整）
- **等權重**：`WEIGHT_TEMP=5.0` 讓資金趨近平均分配，單支最多 8%
- **留校察看（Inertia）**：原持股跌至第 80 名以內就繼續留著 → 每年省下約 10% 換倉摩擦成本

---

## 📊 回測結果

> 截至 2026-03，回測期間 2019~2026（含 COVID 崩盤 + 2022 升息空頭）

| 指標 | 本策略 | 0050 Benchmark |
|------|--------|----------------|
| CAGR | **+9.25%** | +26.59% |
| Total Return | +88.58% | +441.5% |
| Sharpe Ratio | 0.54 | 1.23 |
| Max Drawdown | **-28.30%** | -33.83% |

> ⚠️ **說明**：本策略以「降低最大回撤」為核心目標。CAGR 落後 0050 但 Max DD 略優於 0050。費用模型已充分還原（手續費 0.1425% + 稅 0.3% + 滑價 0.1%）。過去績效不代表未來結果。

---

## 🚀 雲端全自動化（GitHub Actions）

### 設定步驟

1. **Fork 或 clone 此 repo 到你的 GitHub 帳號**

2. **在 GitHub Repo 設定 Secrets**（Settings → Secrets and variables → Actions）：

   | Secret 名稱 | 說明 |
   |-------------|------|
   | `FINMIND_API_TOKEN` | FinMind API Token（必填）|
   | `LINE_CHANNEL_ACCESS_TOKEN` | LINE Messaging API Token（選填）|
   | `LINE_USER_ID` | 你的 LINE User ID（選填）|

3. **啟用 GitHub Pages**（Settings → Pages → Source: GitHub Actions）

4. **初始化你的虛擬存摺**：
   ```bash
   cp portfolio.example.json portfolio.json
   # 修改 portfolio.json 中的 cash 為你的初始資金（預設 100 萬）
   git add portfolio.json && git commit -m "init: my portfolio" && git push
   ```

5. **上傳模型檔**（將本地回測好的 pkl 推上去）：
   ```bash
   git add predictions.pkl weights.pkl eq.pkl bm_eq.pkl
   git commit -m "feat: upload trained model"
   git push
   ```

之後系統每天台灣時間 **15:30** 後自動執行，更新 Dashboard 並（選填）發送 LINE 通知。

### 手動觸發

在 GitHub → Actions → `Daily Quant Update` → `Run workflow` 可隨時手動觸發。

---

## 🏗️ 專案結構

```
tw-quant/
├── .github/
│   └── workflows/
│       └── daily-run.yml          # GitHub Actions 盤後自動化
├── data_loaders/
│   ├── 01_fetch_finmind_data.py   # 下載價量、月營收、三大法人資料
│   ├── 02_fetch_fundamental_data.py  # 下載基本面資料
│   └── 03_fix_financial.py        # 資料修正工具
├── reports/
│   └── generate_report.py         # 生成 14 張 Plotly 回測報告
├── research/                      # 研究 Notebook（共 10 個）
│   ├── 01_Data_Pipeline.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── ...
├── tests/
│   ├── test_live_trade.py         # 交易邏輯單元測試
│   └── test_strategy_utils.py     # 策略工具函數測試
├── .env.example                   # 環境變數範本（複製為 .env 後填入）
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── strategy.py                    # 核心：ML 選股模型 + 回測
├── live_trade.py                  # 盤後虛擬交易 + LINE 通知
└── portfolio.example.json         # 虛擬存摺範本
```

---

## 🧪 Unit Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

預期：25 passed（不需要 API Token，全部使用 mock data）

| 測試檔案 | 涵蓋項目 |
|---------|---------|
| `test_live_trade.py` | 手續費、交易稅、滑價、Weight Cap、NAV 計算 |
| `test_strategy_utils.py` | Softmax 權重、Z-score、Winsorize、Inertia 收斂 |

---

## 🔬 Research Notebooks（探索用）

`research/` 目錄下 10 個 Jupyter Notebook，依序執行可完整理解每個設計決策的實驗過程：

| Notebook | 用途 |
|----------|------|
| `01_Data_Pipeline.ipynb` | 資料載入 + 流動性快速檢查 |
| `02_Feature_Engineering.ipynb` | 13 因子計算 + IC 分析 |
| `03_Model_Training.ipynb` | Walk-Forward + LightGBM |
| `04_Backtester.ipynb` | vectorbt 回測 + 月度熱力圖 |
| `05_Factor_Stability.ipynb` | 因子穩定度 ICIR 熱力圖 |
| `06_Portfolio_Size_Sweep.ipynb` | 持股數量掃描（TOP_K 最佳化）|
| `07_Temperature_Sweep.ipynb` | Softmax 溫度掃描（等權 vs. 集中）|
| `08_Turnover_Cost_Analysis.ipynb` | 換倉率 vs. 摩擦成本 |
| `09_Trailing_Stop_Loss_Analysis.ipynb` | 移動停損測試（實證：台股不適用）|
| `10_Fundamental_Alpha_Ensemble.ipynb` | 基本面濾網測試（PE/PB 會錯殺飆股）|

---

## ⚙️ 年度維修 SOP（Alpha Decay 防止策略失效）

建議每 6~12 個月重新訓練一次以對抗因子衰退：

```bash
source .venv/bin/activate

# 1. 更新資料（約 4~6 小時）
python data_loaders/01_fetch_finmind_data.py

# 2. 重新訓練（約 10~20 分鐘）
python strategy.py

# 3. 推送新模型上線
git add *.pkl
git commit -m "chore: annual model retrain $(date +%Y-%m)"
git push
```

---

## ⚠️ 免責聲明 (Disclaimer)

本專案為**個人學術研究與技術展示用途**，所有回測結果、選股訊號、績效數字均基於歷史資料模擬，**不構成任何形式的投資建議、要約或推薦**。

- **過去的回測績效不代表未來的實際交易盈虧。**
- 量化策略在市場結構改變、流動性不足、或因子失效時，可能產生重大虧損。
- 使用本系統進行任何形式的實際交易前，請充分了解相關金融風險，並自行承擔所有交易損益責任。
- 本專案作者不對任何使用本程式碼所造成的財務損失負責。

> **This project is for educational and research purposes only. It does not constitute investment advice. Past backtested performance does not guarantee future results. Trade at your own risk.**

## 📄 License

本專案採用 [MIT License](LICENSE) 授權。你可以自由使用、修改與分發，但請保留原始授權聲明。
