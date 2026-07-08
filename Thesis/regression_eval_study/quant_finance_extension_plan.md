# 量化金融拓展方案 — metrics_derivations → LinkedIn 系列

> **狀態（2026-07-05）：B1–B4 全部完成。** §8–§12 全文（含推導、worked examples、D6/D7/D8 互動 demo）、7 個 `.finance` bridge、§7.1 金融常數、§7.4 對照詞典、§7.3 雙部曲結語、TOC Part II 連結、og:description 更新——EN/ZH 兩檔逐行平行（各 2094 行），JS 語法與瀏覽器渲染已驗證，16 條引文 + 全部數值宣稱經獨立重算核對通過。§13 依決策降級為 §12 尾聲 remark。
> **待用戶動作：**① repo Settings 啟用 GitHub Pages；② 提供 LinkedIn 個人頁 URL 補進 byline；③ commit；④（可選）跑 make_cover.py 產 ZH 封面。
> **狀態（2026-07-06）：B5 完成。** 審稿回應層（回應 AI_review.md）——§5.2 三明治/FGLS-REML 附註、§12.2 穩健SE、新 §13 分割保形預測+CQR+互動實驗9、尾聲 remark，EN/ZH 逐行平行（各 2282 行），headless Edge 渲染 + demo 9 執行 + 全部數值/交叉引用/術語經三路獨立驗證通過。

主線：**Calibration → Uncertainty → Risk**。同一批數學物件（coverage、PIT、CRPS、NLL、N_eff、censoring），換一個市場語境，就是金融風控的標準工具。每個新章節都以「你已在 §X 推導過 A；金融把 A 叫做 B，並用它做 C」開場。

---

## 一、定位判斷：加層，不改寫

ChatGPT 的評語是對的——這份文件是「Gaussian predictive regression 的評估數學」，不是量化金融教科書。把它改寫成教科書會稀釋兩個模型都稱讚的核心優點（metric 的數學角色 × protocol 前提 × thesis 實數三線縫合）。因此：

- **Part I（現有 §0–§7）基座不動**：保持 thesis-defense 敘事，只加輕量 bridge 框。
- **Part II（新增 §8–§12，+可選 ★§13）做遷移層**：每章一個金融主題、一篇 LinkedIn post。
- `defense_supplement/` 下的舊副本保持凍結不動；所有演進發生在 `regression_eval_study/`。

### 已驗證的技術事實（決定可行性）
| 事實 | 含義 |
|---|---|
| 摺疊 JS 自動包裝所有 `h2[id^="sec"]` | 新章節零 JS 改動即可摺疊 |
| URL hash 自動展開對應章節 + TOC 點擊攔截器已存在 | 每篇 post 直接貼 `…#sec9` 深連結，開箱即用 |
| **新章節必須插在 §7.3 結尾（≈line 1167）與 `<hr>`（line 1169）之間** | sec7 的包裝器以 `<hr>` 為終點；插在 hr 之後會掉出摺疊系統 |
| `<head>` 無 og:/description meta | LinkedIn 連結卡片必須補（見第四節） |
| 檔案 2.3 MB（MathJax 內嵌） | GitHub Pages 無壓力（單檔上限 100MB） |
| §3.5 標題 "Five properties" 實列 P1–P6 | 現成小疵，Batch 1 順手修 |

---

## 二、Part II 章節藍圖

每章遵守既有慣例：`defn/thm/proof(∎)/card/worked/recipe`、`☞ pencil` 檢查點、`★` 進階選讀、式編號 `(8.1)…`、demo 純 canvas JS 零外部依賴。

### §8 Backtest 統計學：有效樣本數、Sharpe 標準誤與多重檢定
*Hook：§0.3 的 N_eff。金融 backtest 最常見的自欺 = 把高度相關的樣本當獨立樣本。*
- **8.1** Sharpe ratio 的標準誤：iid 下 \(SE(\widehat{SR})\approx\sqrt{(1+SR^2/2)/T}\)（delta method 推導，☞pencil）。
- **8.2** 自相關修正（Lo 2002）：年化 \(SR\times\sqrt{252}\) 隱含 iid；有自相關時修正因子的分母結構與式 (0.2) **同構**——「你已經推導過這個公式」時刻，是整個 Part II 的敘事支點。
- **8.3 ★** 多重檢定：測 M 個隨機策略取最大 SR，\(\mathbb{E}[\max]\sim\sqrt{2\ln M}\)；Deflated Sharpe（Bailey & López de Prado 2014）與 White reality check 各一段（敘述+引文，不裝完整證明）。
- **8.4** worked example：1000 個純噪聲策略選出 SR=2 → 校正後不顯著。
- **Demo D6**：sliders φ（自相關）、T、策略數 M → N_eff、SR 的 t-stat、deflated SR。

### §9 Coverage → VaR 例外回測（Kupiec / Christoffersen / Basel 紅綠燈）
*Hook：§2 的 coverage 和 VaR backtest 是同一件事（1−α 覆蓋 ↔ α 例外率）。*
- **9.1** VaR = predictive quantile；hit 序列 \(I_t=\mathbf 1[y_t<-\mathrm{VaR}_t]\)，正確校準 ⟺ \(\mathbb E[I_t]=\alpha\) 且 \(I_t\) iid Bernoulli。
- **9.2** Kupiec POF：binomial LR 檢定完整推導（LR ~ χ²₁，☞pencil，難度與 §6.3 相當）。
- **9.3** Christoffersen 獨立性/條件覆蓋檢定：兩態 Markov chain LR；**例外聚集就是 §0.3 的 ρ_k>0 的 0/1 版**——與 N_eff 敘事閉環。
- **9.4** worked example：Basel traffic light（T=250、99% VaR，綠 ≤4／黃 5–9／紅 ≥10 = binomial CDF 分界，逐個算出區間概率）。
- **Demo D7**：真實分布 Student-t(ν) vs 模型假設 Gaussian，sliders ν/α/T → 模擬 hit 序列、例外計數、紅綠燈判定。

### §10 PIT → 密度預測評估（Diebold–Gunther–Tay、Berkowitz）
*Hook：§4.1 的 PIT 定理在金融文獻裡叫 density forecast evaluation。*
- **10.1** DGT 1998：PIT 序列應 **iid** U(0,1)——比 §4 多出「獨立性」這一維度，正是金融時間序列的重點。
- **10.2** Berkowitz 2001：\(z_t=\Phi^{-1}(u_t)\)，在 Gaussian+AR(1) 備擇下的 3 自由度 LR 檢定（均值/方差/自相關），likelihood 推導完整可行。
- **10.3** 為什麼參數化：小樣本+相關下比 KS 有力——正好接住 §4.5 對 KS 的批評，敘事變成「本文早已論證 KS 不可靠；金融界的工程解是 Berkowitz」。
- **10.4** §4.2 shapes 字典的金融翻譯：fat tails → PIT U 形 → \(z_t\) 超額峰度。

### §11 分位數、pinball 與 elicitability：VaR 可以被「訓練」，ES 不能單獨
*Hook：§3.6 已證 CRPS = ∫ pinball——積木全都在手上了。*
- **11.1** pinball 一致性：\(\arg\min_q \mathbb E[\rho_\tau(y-q)]=\) τ 分位數（推導，☞pencil；正是 §1.3 MAE↔median 的推廣）。
- **11.2** ⇒ VaR elicitable：可直接當 loss 訓練 quantile 模型。
- **11.3** ES 的定義與「不可單獨 elicit」（Gneiting 2011，敘述+直覺反例）；(VaR, ES) 聯合 elicitable（Fissler–Ziegel 2016，給出 scoring function 形式，證明標 ★+引文）。
- **11.4** worked example：10σ 黑天鵝下 NLL（二次爆炸）vs CRPS（線性）具體數字——把 §3.5 P5 + §5.6 對照表變成金融案例。

### §12 NLL → 波動率預測：GARCH、QMLE 與 QLIKE
*Hook：§5.2 的 stationary-point 論證，就是「為什麼 NLL/QLIKE 能學到條件方差」。*
- **12.1** stylized facts：波動聚集 ⇒ 收益率天生異方差；GARCH(1,1) 一頁式（遞迴、無條件方差 \(\omega/(1-\alpha-\beta)\)、EWMA/RiskMetrics λ=0.94 為退化情形）。
- **12.2** Gaussian QMLE：§5.2 的推導逐字遷移 = GARCH 即使真分布非 Gaussian 也能一致估計 σ_t 的直覺。
- **12.3** QLIKE：從 Gaussian NLL 剝離常數 → \(\mathrm{QLIKE}=\frac{\text{proxy}}{\hat\sigma^2}-\ln\frac{\text{proxy}}{\hat\sigma^2}-1\)；Patton 2011「對 noisy proxy 穩健的 loss 只有 MSE 與 QLIKE」；用 §5.2 同款 stationary-point 技巧證 QLIKE 極小值點=真 σ²（☞pencil）。
- **12.4** Student-t NLL：推導、ν 的角色、「Gaussian 假設低估 99% VaR」分位數對照表（正面回應 ChatGPT 點名的 Student-t 缺口）。
- **Demo D8**：QLIKE vs MSE 的非對稱懲罰曲線（slider \(\hat\sigma^2/\sigma^2\)）——低估波動率被罰得更重。

### §13 ★（可選）Censoring → 信用風險與生存分析
*Hook：§6.2 的 censoring lemma = time-to-default 右刪失。*
- **13.1** Kaplan–Meier 一頁 + naive 平均違約時間有偏 = Lemma 6.1 換皮。
- **13.2** hazard rate ↔ CDS 定價一段（survival curve 進 default leg 的折現）。
- **決策點**：若戰線過長，降級為 §12 末尾一個 `.remark` + 引文，不單開章。

---

## 三、Part I 的 bridge 框

新 CSS class **`.finance`**（建議深藍綠底、金色左邊框，label「💹 Quant-finance bridge」，與現有 recap 藍框視覺區隔）。每框 3–6 行，只做「翻譯 + 錨點索引」，**不塞推導**，Part I 節奏不變：

| 位置 | 內容一句話 | 指向 |
|---|---|---|
| §0.3 之後 | 相關樣本 ≠ 獨立樣本 = backtest overfitting 的統計根源 | §8 |
| §1.3 | 重尾下 mean vs median = MSE vs MAE 的金融選擇 | §11 |
| §2.5 | conservative coverage ↔ VaR 例外率過低的監管解讀 | §9 |
| §3.5/3.6 | CRPS 尾部線性 + pinball 對偶 | §11 |
| §4.1/4.5 | PIT = density forecast evaluation；KS 之弊 → Berkowitz | §10 |
| §5.2/5.6 | NLL 學到條件方差 = 波動率建模的損失函數原理 | §12 |
| §6.2 | 右刪失 = 信用風險的 time-to-default | §13/§12 remark |

---

## 四、LinkedIn 發布工程（必做清單）

1. **og meta**：`og:title`、`og:description`、`og:image`（絕對 URL、1200×627 封面圖，可用 demo 截圖拼版）+ `meta name="description"`。LinkedIn 卡片全靠這個；發文前用 LinkedIn Post Inspector 驗證。
2. **標題與開場改 standalone**：`<title>` 縮短（現 >70 字元，lint 已報錯），如 *"Calibration → Uncertainty → Risk: Metric Derivations"*；h1/subtitle 重寫，保留一行「源自碩士論文答辯附錄」+ byline（姓名 + LinkedIn + GitHub 連結）+ 一行 educational disclaimer（非投資建議）。
3. **Hosting**：GitHub Pages。若 thesis repo 為私有，建獨立 public repo（例 `uq-metrics-derivations`），單檔直接放根目錄或 `/docs`。
4. **深連結**：已原生支援，每篇 post 貼 `…/metrics_derivations_en.html#secN` 即可。
5. **系列節奏**（6 篇）：
   - P0 論文主 post → 發 Part I 全文（現有文件）
   - P1 §8「10 萬幀 ≠ 10 萬次實驗；1000 天回測 ≠ 1000 個樣本」（金融共鳴最強，先發）
   - P2 §9 VaR 紅綠燈 → P3 §10 Berkowitz → P4 §11 elicitability → P5 §12 QLIKE + Student-t
   - 每篇格式：金融痛點 hook（1–2 句）→ demo 錄屏 GIF 或公式卡圖 → 3 個要點 → 深連結；篇末互鏈前篇。
6. **語言**：EN 檔為發布主體；ZH 檔維持 line-parallel，**每批次結束時一次性同步**（不必逐 edit 鏡像），發小紅書/知乎可選。

---

## 五、品質守則（硬實力的可信度來源）

- 每個金融結果二選一：**完整最小推導**（Kupiec、Christoffersen、pinball 一致性、QLIKE stationary point、Berkowitz LR 均可做到），或**老實標 ★ + 引文**（DSR、Fissler–Ziegel、White RC）。不放半吊子偽證明——這是被 LinkedIn 上的從業者挑錯 vs 建立信譽的分界線。
- **引文清單**（發布前用 sonnet subagent 逐條核對出處/年份）：Lo 2002 (FAJ)；Kupiec 1995；Christoffersen 1998 (IER)；Berkowitz 2001 (JBES)；Diebold–Gunther–Tay 1998 (IER)；Gneiting 2011 (JASA)；Fissler & Ziegel 2016 (AoS)；Patton 2011 (J. Econometrics)；Bailey & López de Prado 2014；Basel Committee 1996（traffic light）。
- **§7 收尾配套**：7.1 常數表補金融常數（Φ⁻¹(0.99)=2.326、Φ⁻¹(0.975)=1.960、λ=0.94 等）；新增 **7.4「Part I ↔ Part II 對照表」**（每個 metric ↔ 金融名字 ↔ 檢定/損失）；TOC 加 §8–§13；§7.3 結語改雙部曲版本。
- 全程維持 EN/ZH 行平行、tag 配對、demo console 零錯誤。

---

## 六、執行批次（4 個 PR-sized）

| 批次 | 內容 | 產出 |
|---|---|---|
| **B1 底座** | og meta + title/byline/disclaimer、`.finance` class、TOC 擴充、§3.5 "Five"→"Six" 修正、§7.4 對照表骨架 | 可立即發 P0 |
| **B2** | §8 + §9 全文 + D6/D7 demo + 對應 bridge 框 | P1、P2 |
| **B3** | §10 + §11 + bridges | P3、P4 |
| **B4** | §12（+§13 決策）+ D8 + §7 收尾 + 引文核對 + ZH 全量同步 + 全檔校對 | P5 |

規模估計：每章 150–250 行 HTML，Part II 全部 +900–1200 行、檔案 +~80KB；每批結束跑結構自檢（tag 平衡、行平行、demo 冒煙測試）。

## 七、留給你的三個決策點

1. §13（信用風險）獨立成章還是降級為 remark？（建議：先降級，系列反響好再擴）
2. GitHub Pages 用哪個 repo？（thesis repo 若私有需另建 public repo）
3. og:image 封面圖風格：demo 截圖拼版 vs 手繪公式卡？（影響 B1 的一個小任務）
