# 中文版去混寫 — 風格規範（metrics_derivations_zh.html）

這份檔案是**唯一的規範來源**。§9、§10、§12 已依此改完，可當範本。

## 目標

把繁體中文正文裡「隨手沒切回中文」的英文改成中文，同時**保留**真正該用英文的術語。
量化目標：每 100 個漢字的可翻譯英文詞 ≤ 3（§13 原本就是 2.2，§9/§10 改後是 1.1 / 0.5）。

---

## 三層術語政策

### A 層 — 保留英文，不加註、不翻譯
- 縮寫：RMSE、MAE、MSE、P95、CRPS、NLL、PIT、ECE、VaR、ES、CQR、QLIKE、GARCH、QMLE、FZ0、CxLS、ULLN、MLE、CDF、KS、FWER、LONO、FOV、MLP、SE、CI
- 人名／專有名詞：Kupiec、Christoffersen、Berkowitz、Basel、BCBS、Diebold、Gunther、Tay、Bollerslev、Wooldridge、Patton、Kaplan–Meier、Cox、Duffie、Lei、Vovk、Gneiting、Ziegel、Fissler、Osband、Rockafellar–Uryasev、Galois、Lipschitz、Bayes、Lebesgue、Clopper–Pearson、Wilson、Sharpe、RiskMetrics、EWMA
- 分布名（人名類）：**Bernoulli、Poisson、Student-t 保留英文**。但 **Gaussian → 高斯**（全書已用 高斯 67 次），**binomial → 二項**。
- `iid` 保留（全書慣例，不要寫成「獨立同分布」）
- 單位與符號：mm、ms、σ、τ、(i)(ii)(iii)

### B 層 — 中文為主，**首次出現**在括號裡附英文，之後只用中文
例：`過濾族（filtration）`、`分割保形預測（split conformal prediction）`、`可交換（exchangeable）`、`非一致性分數（nonconformity score）`
只給少數真的需要回查文獻的術語加註。**同一節內不要重複加註。**

### C 層 — 一律中譯，不留英文
所有普通名詞、動詞、形容詞。這是本次工作的主體（約六成的英文屬於這一層）。

---

## 語法規則（比術語更影響可讀性）

**英文只能當名詞用，絕不能當謂語、形容詞或副詞。**

- ✗ `這個 selected result 並不 compelling`
- ✓ `這個經過挑選的結果並不具說服力`
- ✗ `Relative power 依 alternative、size、sample size 與 calibration 而定`
- ✓ `相對檢定力取決於對立假設、顯著水準、樣本數與校準方式`

一個中文句子裡若出現 ≥3 個 C 層英文詞，八成是整句要重寫，而不是逐詞換掉。

---

## 統一譯名表（**必須照用**，這些是全書已存在的用法或 §9/§10/§12 已定的）

| 英文 | 中文 | 備註 |
|---|---|---|
| coverage | 覆蓋率 | |
| calibration | 校準 | 全書 69 次 |
| likelihood | 概似 | 全書用「概似／負對數概似」 |
| log-likelihood | 對數概似 | |
| score（概似的） | 分數函數（score） | 首次加註即可 |
| scoring rule | 評分規則 | §3 用「評分」 |
| moment | 矩 | 「二階矩」「四階矩」「矩條件」，**不要用「動差」** |
| variance | 變異數 | |
| covariance | 共變異數 | |
| standard error | 標準誤 | |
| long-run variance | 長期變異數 | |
| effective sample size | 有效樣本數 | §0.3 與 §8 標題已用 |
| asymptotic | 漸近 | |
| stationary | 平穩 | |
| ergodic | 遍歷 | |
| null (hypothesis) | 虛無假設 | |
| joint null | 聯合虛無假設 | |
| alternative | 對立假設 | |
| size (檢定的) | 顯著水準 | |
| power | 檢定力 | |
| cutoff / critical value | 臨界值 | |
| threshold | 門檻 | |
| test | 檢定 | 名詞；「test set」→ 測試集 |
| estimator | 估計量 | |
| estimand | 估計目標 | |
| plug-in | 代入式（plug-in） | |
| consistent | 一致 | |
| identifiable / identified | 可識別 | |
| nonsingular | 非奇異 | |
| interior | 內點 | |
| boundary | 邊界 | |
| minimizer / maximizer | 極小化元／極大化元 | 全書已用「極小化元」 |
| population | 母體 | 全書 47 次 |
| pointwise | 逐點 | |
| conditional | 條件 | |
| unconditional | 無條件 | |
| dependence | 相依 | |
| serial dependence | 序列相依 | |
| autocorrelation | 自相關 | |
| clustering | 叢集 | |
| filtration | 過濾族（filtration） | §10 已首次加註 |
| measurable | 可測 | `\(\mathcal F\)-measurable` → `\(\mathcal F\)-可測` |
| atom | 原子 | |
| tie / tie convention | 並列／並列約定 | 全書已用「並列規則」 |
| convention | 約定 | |
| identity | 恆等式 | |
| loss | 損失 | |
| surrogate | 替代目標（surrogate） | |
| proxy | 代理 | |
| robust / robustness | 穩健／穩健性 | |
| censoring / censored | 刪失 | |
| uncensored | 未刪失 | |
| pooled / pooling | 池化 | |
| fold / folds | 摺 | 全書已用「各摺」 |
| holdout | 留出 | |
| split | 分割 | |
| train / calibration / test set | 訓練集／校準集／測試集 | |
| seed / seeds | 亂數種子 | |
| baseline | 基準 | |
| protocol | 協定 | |
| artifact | 產物 | 全書已用「產物」 |
| printed p. / pp. | 印刷頁 | §13 已用 |
| Eq. / Eqs. | 式 | |
| Section / Sec. | 第 N 節 | |
| Theorem / Proposition / Corollary（引文中） | 定理／命題／推論 | |
| Corollary 4.3（本書自己的） | **系 4.3** | 本書環境標籤就叫「系 4.3」，不要寫 Cor. 4.3 |
| naive | 天真 | 全書已用「天真的」；`naive 帶` → `天真帶` |
| cap | 上限 | `FOV cap` → `FOV 上限` |
| gap | 落差／偏離量 | 看語境 |
| head（網路的） | 輸出頭 | `variance head` → 變異數輸出頭 |
| clipping / clipped | 截斷 | |
| network | 網路 | |
| stage | 階段 | Stage 2 → 第 2 階段 |
| step | 步驟 | |
| miss（§3 那個量 \(|y-\mu|\)） | **偏離** | 「miss 距離」→「偏離距離」，與 §7 速查表 P3 一致 |
| miss（§13 預測帶沒蓋住某點） | **漏接** | 「天真帶漏接的點」 | |
| bin | 分箱 | |
| histogram | 直方圖 | |
| ordering | 次序 | |
| uniform / uniformity | 均勻／均勻性 | |
| infimum | 下確界 | |
| bootstrap | 拔靴法（bootstrap） | |
| decomposition | 分解 | |
| mismatch | 不一致 | |
| selection（挑選效應） | 選擇效應 | |
| sharpness | 銳利度 | |
| admissible | 可容許 | 全書已用 |
| proper（評分規則） | 嚴格適當 / 適當 | strictly proper → 嚴格適當 |
| aggregate | 彙總 | |
| forecast | 預測 | |
| density forecast | 密度預測 | |
| hazard | 風險（率） | proportional hazards → 比例風險 |
| influence function | 影響函數 | |
| delta method | delta 方法 | 全書已用 |
| sandwich | 三明治 | 全書已用 |
| information matrix | 資訊矩陣 | |

---

## 硬性約束（違反就會被退回）

1. **不得更動任何數學**：`\( ... \)`、`\[ ... \]` 內容一個字元都不能改（換行位置可變）。
2. **不得更動任何 `\tag{...}`、`id="..."`、`href="..."`、`class="..."`**，數量與內容都要一模一樣。
3. **不得增刪任何 HTML 元素**（`<p> <div> <li> <tr> <td> <b> <i> <a> <span>` … 的開合標籤總數必須不變）。
4. **不得增刪、合併或重排段落／定理／證明**。一段對一段地改寫。
5. **不得改動事實**：數字、引文頁碼、章節交叉引用、限定語（「不」「並非」「僅」「不足以」…）全部照舊。這本書大量使用「這不證明 X」式的限定句，語氣不可弱化。
6. `<div class="demo">` 內的 `.demo-title`、說明 `<p>`、`.trylist` **都算正文，要一起改**；但 `<label>`、`<input>`、`<canvas>` 的屬性不要動。
7. 保留全形標點與原有的空格慣例（中文與英數之間留一個半形空格）。

---

## 已知缺陷模式（改的時候順手抓）

1. **標題／正文不一致**：§8 標題寫「有效樣本數、Sharpe 標準誤」，內文卻寫 `effective sample size` / `standard error`。同一概念在同一頁兩種寫法。
2. **交叉引用指向不存在的名字**：正文寫 `Cor. 4.3`，但書裡的環境標籤是「系 4.3」。
3. **英文當謂語**：`並不 compelling`、`看起來 compelling`。
4. **整句英文語序**：`Relative power 依 A、B、C 而定` — 主語是英文名詞片語。
5. **表格欄位比正文更嚴重**：§7 速查表整欄是英文短語。
6. **demo 的 `.trylist` 是重災區**：正文改乾淨了，「試試看」卻還是英文。
