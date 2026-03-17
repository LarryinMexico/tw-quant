# Taiwan Stock ML Quant Trading System (tw-quant)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-orange.svg)](https://lightgbm.readthedocs.io/)
[![Backtesting](https://img.shields.io/badge/vectorbt-0.26%2B-blueviolet.svg)](https://vectorbt.dev/)
[![Automated Trading](https://img.shields.io/badge/GitHub_Actions-Automated-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](LICENSE)

本專案為針對台灣股票市場開發的端到端機器學習量化交易系統。系統整合了資料傳輸、特徵工程、Walk-Forward 滾動交叉驗證、基於 LightGBM 的動態因子選擇模型，以及全自動化的盤後紙上交易 (Paper Trading) 與報表渲染模組。


## 核心技術特性 (Key Features)

- **Walk-Forward 嚴謹驗證機制**：採用滾動式時間視窗訓練 (48 個月訓練、1 個月 Purge 避免前瞻偏差、3 個月樣本外測試)，模擬最真實的 Out-of-Sample 預測情境，並避免資料外洩 (Data Leakage)。
- **動態因子選擇演算法**：內建 Fold-Internal 資訊係數 (IC) 與資訊係數比率 (ICIR) 檢驗，在每次滾動訓練的對應子集內自主篩選穩健因子，防止 Lookahead Bias 並適應多變的市場微結構 (Market Microstructure)。
- **系統性風險控管 (Regime Filter)**：整合 0050 ETF 的多重技術訊號 (MA60, MA20 與無大幅衰退)，於市場系統性回撤 (Drawdown) 時強制清倉，達成優異的資產保護能力。
- **交易摩擦成本最佳化 (Turnover Optimization)**：結合 Softmax 溫度參數權重與持股留校察看 (Keep_Top_K) 慣性演算法，有效抑制頻繁換股，粗估降低月度摩擦成本達 10%。
- **全自動雲端作業流 (Serverless Infrastructure)**：依賴 GitHub Actions 定期觸發，建構涵蓋盤後資料匯入、投組更新、儀表板部署及即時通訊的完善 CI/CD 工作流。

## 回測績效與指標 (Performance Metrics)

回測期間涵蓋 2019 年至 2026 年 (含 COVID-19 系統性崩盤)。
系統內建極具保守性的摩擦成本估算：單邊手續費 0.1425%、交易稅 0.3% 及市場滑差 0.1%。對照組使用經除權息及分割還原的台股基準。

| 指標 (Metric)               | tw-quant ML 策略 | 基準指數 (0050 ETF) |
| :-------------------------- | :----------------- | :------------------ |
| **CAGR (年化報酬率)**       | **9.25%**          | 26.59%              |
| **Max Drawdown (最大回撤)** | **-28.30%**        | -33.83%             |
| **Sharpe Ratio (夏普值)**   | 0.54               | 1.23                |
| **Total Return (總報酬率)** | 88.58%             | 441.50%             |

> **設計目標解析**：量化模型首重資產存續能力。本策略在面臨每月底高頻換倉產生的大額交易稅費下，仍顯著降低了 Max Drawdown 的侵蝕幅度。在規避劇烈回撤的首要目標上，展示了系統性架構防禦的可靠性，亦可用作探索新 Alpha 因子的穩定基準版。

## 系統部署與執行 (Installation & Quick Start)

### 1. 本地環境建置

需求環境為 Python 3.10+。

```bash
git clone https://github.com/LarryinMexico/tw-quant.git
cd tw-quant

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 環境變數設定

複製環境變數範本件並更新您的 API 憑證：

```bash
cp .env.example .env
```
確保 `FINMIND_API_TOKEN` 正確輸入以利透過合法端點擷取資料。

### 3. 本地管線執行 (Pipeline Execution)

依序啟動管線的各元件，包含拉取外部歷史資料、訓練 LightGBM 及生成分析報告：

```bash
# Data Ingestion (依據速率限制，約需 4~6 小時)
python data_loaders/01_fetch_finmind_data.py

# Model Inference & Backtest (約 15 分鐘)
python strategy.py

# HTML Dashboard Rendering
python reports/generate_report.py
```
執行完畢後開啟 `reports/backtest_report.html`，即能查閱 14 項圖表構成的互動式儀表板。

## 核心選股因子集 (Core Factors Directory)

策略模組運算 13 項跨領域的量化特徵，結合截面標準化 (Cross-Sectional Z-Score) 後輸入模型。以下為依據 ICIR 平均表現排序的前 8 大穩健因子：

| 特徵類別 | 變數代碼 | 經濟/演算法涵義定義 | 實證排序 |
| :--- | :--- | :--- | :---: |
| Institution | `trust_net_10d` | 本土投信近 10 日淨買超對特定成交總額佔比 | 1 |
| Technical | `rsi_14` | 股價 14 日相對強弱平滑指標 (RSI) | 2 |
| Volatility | `atr_rel` | 平均真實波幅相對值 (Relative ATR) - 表徵波動擴張 | 3 |
| Momentum | `price_52w` | 現價距 52 週波段高點之相對強度 | 4 |
| Momentum | `mom_1m` | 1 個月絕對價格動量 | 5 |
| Fundamental | `rev_yoy` | 月度營收之跨年同期成長率 (YoY) | 6 |
| Fundamental | `rev_mom` | 月度營收之跨月成長率 (MoM) | 7 |
| Momentum | `mom_1m_ra` | 風險調整後之 1 個月時間序列動能 (平滑雜訊) | 8 |

## 原始碼架構 (Project Structure)

```text
tw-quant/
├── .github/workflows/
│   └── daily-run.yml          # GitHub Actions CI/CD 組態設定
├── data_loaders/
│   ├── 01_fetch_finmind_data.py
│   └── 02_fetch_fundamental_data.py
├── reports/
│   └── generate_report.py     # Plotly 報表與分析模組
├── research/                  # 量化工程實驗與因子驗證 Jupyter Notebooks
│   ├── 02_Feature_Engineering.ipynb
│   ├── 05_Factor_Stability.ipynb
│   └── 08_Turnover_Cost_Analysis.ipynb
├── tests/                     # 涵蓋交易規則及權限單元測試 (Pytest)
├── strategy.py                # 整合 LightGBM 訓練、推理與 Vectorbt 框架之入口
├── live_trade.py              # 虛擬資產組合管理與每日交易引擎
└── requirements.txt
```

## 雲端自建 (Continuous Integration)

本專案附帶完備的 `daily-run.yml` 部署流程檔，供建立自有定時派送任務。

1. 打開此儲存庫並 Fork 至個人名單下。
2. 配置 Secrets：至 **Settings > Secrets and variables > Actions** 加入 `FINMIND_API_TOKEN`。
3. 配置推播通知 (選填)：設定 `LINE_CHANNEL_ACCESS_TOKEN` 及 `LINE_USER_ID`。
4. 解除 `.github/workflows/daily-run.yml` 中的 `schedule` 註解標記，即完成自動化。

## Disclaimer (免責聲明)

本專案及釋出之原始碼僅供學術探討、工程技術交流展示之用。系統產出之回測分析、機器學習篩選標的等結果均立基於歷史數據。**此專案完全不構成任何投資建議、財務指導或獲利保證。**

- 量化模型仰賴歷史規律，不保證等效之未來營利水準。
- 因市場微結構遷移 (Regime Shift)、流動性黑洞或因子衰減 (Alpha Decay) 的發生，演算法極可能遭遇劇烈損失。
- 將本專案應用於實盤或金融實作前，請自行釐清衍生之資本風險，作者與協作者對衍生之具體盈虧概不負責。

## License

This project is licensed under the [MIT License](LICENSE). 
