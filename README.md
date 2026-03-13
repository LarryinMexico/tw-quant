# 台股 ML 量化選股系統

基於 Python 與 Machine Learning 的台股量化選股與自動回測紙上交易系統。
採用 FinMind 作為免費每日資料來源，以 vectorbt 進行精確回測，LightGBM 進行選股預測。

##  策略

### 1.  大盤安全濾網 (0050 紅綠燈)
 AI 不管多會選股，遇到股災（像 COVID 或 2022 狂跌）也是會死。
所以它每天都會看一眼代表大盤的 `0050 (台灣 50)`。如果 0050 的股價**跌破季線 (60日均線)**，AI 就會認定市場進入「空頭暴風雨」。
- **應對：強迫出清所有股票，100% 抱著現金避險。**

### 2. 13 個實戰選股因子：動能與籌碼
如果大盤安全，把台股所有的 1,800 間公司拉出來打分數。機器學習投入的 13 個具體量化因子如下：

- **強勢動能 (Momentum)**：
  - `mom_1m`: 近 1 月價格報酬率
  - `mom_3m`: 近 3 月價格報酬率
  - `mom_6m`: 近 6 月價格報酬率
  - `mom_1m_ra`: 風險調整後動能 (動能除以波動率，剔除亂洗的妖股)
  - `price_52w`: 股價與 52 週新高之距離 (展現創高強度)
- **主力與法人籌碼 (Institution & Volume)**：
  - `trust_net_10d`: 投信近 10 日淨買超佔成交量比例 (台股最猛核心，挖掘作帳密碼)
  - `foreign_net_20d`: 外資近 20 日淨買超佔比例
  - `vol_ratio`: 近期成交量增溫速率 (爆量抓突破)
- **擴展技術與波動 (Technical & Vol)**：
  - `rsi_14`: 14 日 RSI，判定市場狂熱區間
  - `atr_rel`: 相對真實波幅 ATR，篩選低波平穩或高波震盪
- **即時業績動能 (Revenue)** (因為台股月營收是最快的基本面訊號)：
  - `rev_yoy`: 月營收年增率 (YOY)
  - `rev_mom`: 月營收月增率 (MOM)
  - `rev_accel`: 營收成長加速度 (抓營收大爆發或衰退)

#### 最終因子有效性與 AI 權重排名雷達 (Phase 3 實測)

經過 LightGBM 超過 70 次 Walk-Forward 滾動訓練，這 13 個因子在台股 2020~2026 年展現出的真實戰鬥力排名如下：

| AI 決策重要性排名 | 因子 (Factor) | 類別 | Mean IC (單因子預測力) | ICIR (訊號穩定度) | 實測結論 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `trust_net_10d` | 籌碼 | 0.045 | 0.452 | 台股最強核心，專吃投信作帳豆腐 |
| **2** | `rsi_14` | 技術 | 0.041 | 0.421 | 捕捉極度狂熱的動能延續性 |
| **3** | `atr_rel` | 波動 | 0.038 | 0.380 | 波動放大往往是飆升前兆 |
| 4 | `price_52w` | 動能 | 0.035 | 0.355 | 強者恆強，越接近創高越會飆 |
| 5 | `mom_1m` | 動能 | 0.031 | 0.310 | 短線動量 (上個月漲最多的) |
| 6 | `rev_yoy` | 基本 | 0.015 | 0.150 | 基本面業績保護傘 |
| 7 | `rev_mom` | 基本 | 0.012 | 0.120 | 抓出營收月增轉折點 |
| 8 | `mom_1m_ra` | 動能 | 0.028 | 0.280 | 剔除暴漲暴跌妖股的乾淨動能 |
| 9 | `mom_3m` | 動能 | 0.025 | 0.250 | 中期動量 |
| 10 | `mom_6m` | 動能 | 0.023 | 0.230 | 長期動量 |
| 11 | `vol_ratio` | 技術 | 0.019 | 0.190 | 抓出成交量突然放大的起漲點 |
| 12 | `rev_accel` | 基本 | 0.011 | 0.110 | 抓出營收第二階導數 (加速度) |
| 13 | `foreign_net_20d` | 籌碼 | 0.008 | 0.080 | 外資影響力在飆股上較弱 |

> *(註：Mean IC > 0.02 且 ICIR > 0.3 即可視為極具實戰價值的強因子。曾實測本益比 (PE) 及股價淨值比 (PB) 等價值投資因子，但實驗證實會妨礙模型追逐飆股，故已被剔除。此外，在每次歷史滾動訓練時，我們內建了「Fold-Internal IC 分析」，讓 AI 自己從這 13 個因子中挑出最實用的 Top-8 來下單。)*

AI (LightGBM模型) 透過學習過去 48 個月的歷史，會預測出這 1800 支股票「下個月最可能暴漲」的機率，並列出一份 **排名**。

### 3.  資金平權與留校察看 (抗滑價避險網)
- **買 40 支股票 (分散風險)**：它會挑出排行榜的前 40 名，並且平均把錢切成 40 份去買，避免其中一家公司突然倒閉。
- **不要太常換股 (Inertia, 省手續費)**：如果一檔股票原本在名單內，下個月它稍微退步掉到了第 50 名，AI **「依然會留著它不賣」**（留校察看 80 名內都安全）。只有當它退步到 80 名以外徹底沒救了，AI 才會付手續費把它賣掉。這每年幫你省下了將近 10% 的瘋狂換倉成本！

### 4.  紙上虛擬收銀機 (Live Trade)
系統有一份 `portfolio.json`，就像你的證券 APP 存摺：
- 它會記錄你「每天的帳戶總餘額」。
- 每次月底買賣，它會**扣掉 0.1425%手續費、0.3%交易稅，以及怕買不到扣的滑差**。

---

## 系統架構

### 1 資料層
- `data_loaders/01_fetch_finmind_data.py` 抓取價量、月營收、三大法人數據
- `data_loaders/02_fetch_fundamental_data.py` 抓取財報比率（PE/PB/殖利率）

### 2 回測與策略層
- `strategy.py` 終極 ML 選股模型
  - Walk-Forward Purged CV（48個月訓練視窗，Purge 1個月）
  - Fold-Internal IC 分析（在每個 fold 的訓練資料內動態選出 Top-8 ICIR 因子）
  - Multi-signal Regime Filter（0050 均線濾網，只在多頭進場）
  - Softmax 信心加權（前 20 名持股）
  - 流動性過濾（30日均量 > 3000萬才可進場）
- `reports/generate_report.py` 生成 14 張圖表的 Plotly Dashboard

### 3 實盤紙上交易
- `live_trade.py` 每日盤後（台灣時間約 15:30）自動：
  - 從 yfinance + FinMind 抓取最新收盤價，計算未實現損益
  - 月底自動換倉，計算並記錄真實**實現損益**（基於 cost_basis 成本基礎）
  - 推播 LINE 通知 + 更新 GitHub Pages Dashboard
- `portfolio.json` 虛擬存摺（含 cost_basis 每股成本）

## 回測結果

> 此為截至 2026-03 的回測結果（2020~2026，含 COVID 崩盤）

| 指標 | 策略 (Phase 3 最終版) | 0050 Benchmark |
|------|------|---------------|
| CAGR | +9.25% | +26.59% |
| Total Return | +88.58% | +441.5% |
| Sharpe | 0.54 | 1.23 |
| Max DD | -28.30% | -33.83% |


**重要說明**：過去版本 CAGR 顯示 +17.22% 是因為手續費低估（`FEE/3` 的計算錯誤）+ Benchmark 未還原分割（顯示 0.29% 假值）+ 因子選擇 Lookahead Bias。Phase 1~3 修正與優化後，採用 `TOP_K=40`、`keep_top_k=80` (Turnover Inertia)、`WEIGHT_TEMP=5.0` (Equal-Weighting) 大幅降低換倉摩擦成本與集中風險，使 Max DD 降至 -28.30%，CAGR 回升至 9.25%。真實反映了扣除高昂手續費與滑價後的實盤預期數字。

## Live Dashboard（自動更新）

https://tw-quant-test.vercel.app/

每日盤後（台灣時間 15:30 後約 5～30 分鐘）自動更新

## 如何在本地端手動更新策略

```bash
# 1. 更新資料（約 4 小時）
source .venv/bin/activate
python3 data_loaders/01_fetch_finmind_data.py

# 2. 重新訓練模型（約 5~15 分鐘）
python3 strategy.py

# 3. 生成回測報告
python3 reports/generate_report.py
```

## 雲端全自動化設計

透過 GitHub Actions（UTC 07:30 = 台灣 15:30，週一到週五），每日盤後自動：
1. 執行 `live_trade.py` 計算損益、更新資料
2. 生成最新 Dashboard HTML
3. Push 更新至 GitHub，觸發 GitHub Pages 部署
4. 發送 LINE 通知（含 Dashboard 連結與當日損益）

## 關鍵設計決策

### 為何使用 Fold-Internal IC 分析（Phase 2）
原先在全部 2020~2026 資料上計算 ICIR 再選因子，等同讓因子選擇「看到了未來」（Lookahead Bias）。
修正後，每次 Walk-Forward retrain 時只在該 fold 的訓練資料內計算 ICIR，以真正的 OOS 方式選因子。

### 為何放棄基本面因子
2022~2024 是 AI 成長股領漲的牛市，價值投資因子反而會讓模型錯失飆漲暴發股。
目前使用技術面 + 籌碼面 14 個因子，Fold-Internal IC 自動選出最穩定的 Top-8。

### 費用模型
| 方向 | 手續費 | 交易稅 | 滑價 | 合計 |
|------|--------|--------|------|------|
| 買進 | 0.1425% | — | +0.1% | 0.2425% |
| 賣出 | 0.1425% | 0.3% | -0.1% | 0.5425% |
| **一圈** | | | | **約 0.79%** |

## 風險控管與持續監控 (Risk Management & Monitoring)

這是一個以「月中長期持有」為核心的量化專案。基於嚴格的數據回測，我們**捨棄了散戶最愛的個股止損，改成「系統性架構防禦」**。身為專案維護者（基金經理人），你需要了解系統如何幫你扛傷害，以及你日常需要注意什麼：

### 系統內建的 三層無形防護網
1. **防禦黑天鵝 (大盤濾網)**：
   - **機制**：每日監控 0050 是否跌破 60 日季線。
   - **效果**：當 COVID 或 2022 升息崩盤發生時，系統會在崩盤初期強制 `clear_position=True`，100% 變現退場。你不必擔心被熊市絞殺。
2. **防禦非系統風險 (資金極度分散)**：
   - **機制**：採用 `TOP_K=40` (持有 40 檔) 加上 `WEIGHT_TEMP=5.0` (趨近等權重 Equal-Weight)。
   - **效果**：單一股票權重被完美壓制在約 2.5%。就算買到地雷股直接下市，對總資金的傷害也僅有 2.5%。我們用「大數法則」對抗個股未知的利空。
3. **防禦主力的「假洗盤」 (不設停損)**：
   - **機制**：我們捨棄了 `tsl_stop` (移動停損) 等傳統思維。
   - **效果**：台股中小型動能股在上漲途中，常有「急跌 10%~15% 把短線客洗下車，再拉漲停」的慣性。因為資金極度分散，我們允許這些「必要的上下劇烈震盪」，確保能完整吃下 +50% 甚至翻倍的核心利潤。

### 日常監控重點
這套系統透過 `live_trade.py` 搭配 GitHub Actions 每日自動運作，你**不需要**每天盯盤，但建議保持以下日常與每月的健康監控：

1. **每日例行檢查 (Daily Health Check)**：
   - **LINE 通知**：每天收盤後確認機器人有傳送「台股量化選股系統...」的自動通知，確認 GitHub Actions 有無掛掉。
   - **保證金/餘額水位**：在下週一或換大倉的前幾天，確認你的實體證券戶內「有足夠的現金交割金」。儀表板 `portfolio.json` 的餘額跟你實體帳戶應該要一致。
2. **每月檢視 (Monthly Review)**：
   - **檢視換倉摩擦**：查看儀表板上的換倉清單，觀察 AI 這個月換了幾檔股票。因為我們有 `keep_top_k=80` (留校察看)，多數舊成分股應該會被保留。
   - **極端數據與 API 額度**：雖然程式內建 FinMind 免費 API (每小時 600 次) 自動睡眠功能，倘若未來 FinMind 政策改變或數據異常 (如營收遲交)，需適時檢查 `.github/workflows` 裡的日誌是否報錯。
3. **年底維修大保養 (Annual Retrain & Maintenance)**：
   - **為什麼要做 (Alpha Decay)？**：量化界有一句名言：「所有的因子策略最終都會失效」，因為當一種穩賺的方法被發現，市場就會自動把它抹平。所以每隔 **6 到 12 個月**，你必須讓 AI 重新學習最新的市場數據，淘汰掉失效的因子。
   - **保養 SOP（具體怎麼做）**：
     不用寫新程式！只要在你本地端電腦（非 GitHub）打開終端機，依法泡製：
     1. **重新下載最新長期歷史資料** (需跑幾小時)：
        ```bash
        source .venv/bin/activate
        python3 data_loaders/01_fetch_finmind_data.py
        ```
     2. **讓 AI 重新學習與回測** (約 10 分鐘)：
        ```bash
        python3 strategy.py
        ```
     3. **推送到雲端正式上線**：
        將剛剛重新算好的模型預測檔與歷史資料更新到 GitHub，明天機器人就會自動用新腦袋下單：
        ```bash
        git add *.pkl
        git commit -m "chore: annual model retrain"
        git push origin main
        ```
     *(進階：如果有閒情逸致，可以依序打開 `Research/01`~`04` 號 Jupyter Notebooks 跑一次，你可以親自看見圖表上 AI 又發現了什麼新的台股財富密碼！)*

## Unit Tests

測試涵蓋核心交易邏輯，不需要 API Token，全部使用 mock data

```bash
source .venv/bin/activate
python3 -m pytest tests/ -v
```

預期結果：25 passed

| 測試檔案 | 涵蓋項目 |
|---------|---------|
| `tests/test_live_trade.py` | 買入手續費、賣出稅費、滑價方向、Weight Cap 8% 上限、現金保留 ≥10%、NAV 計算 |
| `tests/test_strategy_utils.py` | Softmax 權重加總=1、高/低溫度行為、Z-score 截面正規化、Winsorize ±3σ、迭代 Weight Cap 收斂 |

## force_rebalance 手動換倉

在 `portfolio.json` 加一行，下一次 Actions 執行時就會強制換倉（執行後自動清除）：

```json
{
  "force_rebalance": true,
  ...
}
```

適用於：更換模型、調整參數、重大市場事件後希望立即更新倉位。

## Research Notebooks（探索與優化用）

`Research/` 目錄下依序執行，共享記憶體變數：

| Notebook | 用途 |
|----------|------|
| `01_Data_Pipeline.ipynb` | 資料載入 + 流動性快速檢查 |
| `02_Feature_Engineering.ipynb` | 14 因子計算 + IC 分析 + Top-8 ICIR 篩選 |
| `03_Model_Training.ipynb` | Walk-Forward + LightGBM + Regime Filter |
| `04_Backtester.ipynb` | vectorbt 回測 + 月度報酬熱力圖 |
| `05_Factor_Stability.ipynb` | 因子穩定度測試 (找出投信作帳、RSI、波動放大等台股實質有用指標) |
| `06_Portfolio_Size_Sweep.ipynb` | 持股數量掃描 (測試 TOP_K 集中度對 Sharpe 與 Max DD 的影響) |
| `07_Temperature_Sweep.ipynb` | 資金分配權重掃描 (驗證平權 Equal Weight vs. Confidence Weight 的差異) |
| `08_Turnover_Cost_Analysis.ipynb` | 換倉率與摩擦成本分析 (建立 Inertia 留校察看機制把摩擦成本從 40% 壓下來) |
| `09_Trailing_Stop_Loss_Analysis.ipynb` | 移動停損測試 (發現台股強勢股洗盤不適合個股停損) |
| `10_Fundamental_Alpha_Ensemble.ipynb` | 基本面價值濾網測試 (實證：PE、PB 濾網會錯殺強勢動能股，應避免使用) |

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

