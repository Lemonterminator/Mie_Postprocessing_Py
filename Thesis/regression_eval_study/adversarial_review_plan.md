# metrics_derivations 對抗式數學審計 — 分輪工作流計劃

**對象**：`metrics_derivations_en.html`（canonical）+ `metrics_derivations_zh.html`（line-parallel 鏡像）
**規模**：33 個編號 claim（14 Thm / 17 Lem / 2 Cor）+ 7 Def + 35 個 proof block，另有摘要卡、數值示例、應用性保證（conformal、Christoffersen、CRPS 等）
**已知種子錯誤（校準用）**：§1.2 median/MAE 原子點 DCT gap、Thm 11.1 缺 E|Y|<∞ 與 quantile 唯一性、Thm 5.1 缺 0<Var<∞、Christoffersen T vs T−1、CRPS moment-match 過度宣稱、conformal N_eff 宣稱
**約束**：一輪 = 一個對話；Claude 配額有限；混用 ChatGPT。

## 核心設計原則

1. **跨平臺不是妥協，是去相關手段。** 文檔由 Fable 5 + Sonnet 5 寫成——同家族模型審自己的數學有 common-mode blind spot。因此：*判斷型*輪次（重推導、審證明）放 ChatGPT 最強推理檔；*需要 repo / 檔案系統 / 跑代碼*的輪次放 Claude Code。
2. **磁盤上的 artifact 是唯一介面。** 每輪從 `audit/` 目錄讀入，結束時寫回嚴格格式的 JSONL/腳本。任何一輪都不依賴上一個對話的記憶，換平臺零損耗。
3. **否決不對稱。** 一個「假設全部滿足、結論可執行驗證失敗」的反例 = 自動 BLOCKER；任意數量的 PASS 都不能累積成「已驗證」。PASS 必須附 per-obligation 證據。
4. **每個 LLM 判斷輪都放 canary。** 漏抓已知錯誤的輪次，其 PASS 全部作廢重跑；同時放已知正確命題防「全判錯」作弊。
5. **可執行反例是最高權威。** 反例一律落地為 Python 腳本（假設檢查器 + 結論斷言），成為永久回歸測試，修正後與釋出前重跑。

## 目錄約定

```
Thesis/regression_eval_study/audit/
  freeze.json            # 兩檔 SHA-256 + 時間戳
  claims.jsonl           # 全量 claim 清單（含非正式宣稱）
  packs/stmt/C##.md      # 自足陳述包（含所有被引用定義，無證明）——給盲推導者
  packs/full/C##.md      # 陳述+證明包——給證明審計者
  obligations.jsonl      # R1 輸出
  incoming/r?_response.json   # ChatGPT 各輪回貼的 JSON
  rederivations.jsonl    # R2
  proof_audit.jsonl      # R3
  battery/*.py           # R4 反例腳本（永久回歸測試）
  battery_results.jsonl
  findings_final.jsonl   # R5
  fix_plan.md
  patch_log.jsonl        # R6
  parity.jsonl           # R7
  gate_report.json       # R8
```

## ChatGPT 交接機制（每個 ChatGPT 輪通用）

- Claude 側前一輪產出一個**自足 handoff pack**（單一 .md，≤ ~120k 字元；定義全部內聯，不引用 repo 路徑），你上傳給 ChatGPT。
- 要求 ChatGPT **只輸出 fenced JSON**（給定 schema），你存為 `audit/incoming/rX_response.json`。
- 下一個 Claude 輪以該檔為輸入做機械校驗（schema、canary、覆蓋數），不合格即宣告該輪作廢。

---

## R0 — 凍結 + Claim 普查（Claude Code；Sonnet 子代理 + 腳本；成本：低）

**做什麼**：SHA-256 凍結兩檔 → Python 腳本剝 HTML 抽出全部 block → Sonnet 分段成 claims.jsonl。**普查範圍不只 33 個編號 block**：摘要卡結論、行內公式宣稱（如 Christoffersen 分母）、數值示例、應用保證（conformal coverage 陳述、CRPS 性質）都編入，各給穩定 ID（C01…）。為每個 claim 生成兩種 pack（stmt / full），定義閉包內聯。

**產出**：`freeze.json`, `claims.jsonl`, `packs/`。
**糾錯**：
- 覆蓋數由 grep 決定論核對（35 proof、33 編號 block 必須一一對上）；
- Haiku 抽 5 個 pack 驗「自足性」（不看原文能否理解陳述）；
- claims.jsonl 記每條的行號錨點，供 R6 定位。

## R1 — Proof-obligation 編譯 + 分級（ChatGPT Thinking 檔；成本：中，省 Claude 配額）

**做什麼**：上傳「陳述-only」handoff pack + 固定義務清單模板。對每條 claim 列出：量詞與定義域、有限性（E|Y|、Var）、原子/CDF 平台敏感性、普通導數 vs 單側導數、DCT 的 a.e. 極限與 dominator、存在性/唯一性、iid/exchangeability、漸近條件、來源可核性。同時給**分級**：
- **T1**（高危，預期 8–12 條）：進 R2 盲推導 + R3 證明審計 + R4 反例炮台；
- **T2**：R3 + R4；
- **T3**（例行代數）：只進 R4 炮台。

**產出**：`obligations.jsonl`（含 tier）。
**糾錯**：混入 5 個 canary（3 錯 2 對：缺 dominator 的 DCT、非嚴格凸宣稱唯一、原子點普通微分；+2 條正確命題）。已知 6 個種子錯誤所在 claim 若未被評 T1 → 該輪校準失敗，重跑。

**Prompt 骨架**（英文，貼 ChatGPT）：
> You are a proof-obligation compiler. For each claim below (statement only, no proof), output JSON: `{id, quantifiers, domain, finiteness_needed, atom_sensitive, one_sided_derivatives_needed, dct_obligations, existence, uniqueness, dependence_assumptions, tier, tier_reason}`. Do not judge correctness; enumerate what any valid proof must discharge. Output fenced JSON only.

## R2 — 盲重推導（ChatGPT 最強推理檔，全新對話；成本：高，本工作流最貴的一輪）

**做什麼**：只給 T1 claims 的 stmt pack（**無證明、無先前 findings**）。要求逐條給出三選一結論：
- `PROVED`：附證明梗概 + **實際用到的假設清單**；
- `PROVED_UNDER_STRONGER`：明列額外需要的假設 —— 這是本輪的黃金產出，「所需假設 − 原陳述假設」的差集在 R5 機械地變成 MAJOR 候選；
- `REFUTED`：給候選反例（分布 + 參數 + 預期失敗方式），交 R4 落地執行。

**產出**：`rederivations.jsonl`。
**糾錯**：canary 同放；要求每條 PROVED 必須逐項對照 R1 義務清單打勾（拒收「顯然成立」）。

## R3 — 對抗式證明審計（ChatGPT 最強推理檔，另開全新對話；成本：中高）

**做什麼**：給 T1+T2 的 full pack。指令：找出**第一個推不出來的步驟**，引用原句，分類為 {invalid_inference, missing_hypothesis, circular, notation_collision, scope_leak(連續情形結論外推到一般分布)}。明令禁止「整體看起來沒問題」——每條被 R1 標記的義務必須逐項 discharge 或標記缺口。

**產出**：`proof_audit.jsonl`（step-level 引文 + 分類）。
**糾錯**：canary 換皮再放；與 R2 獨立（不同對話、互不可見），R5 才合流。

## R4 — 可執行反例炮台（Claude Code；Sonnet 驅動 Python，Fable 只構造疑難邊例；成本：中）

**做什麼**：
1. 建固定炮台：point mass、兩點分布、CDF 平台、Cauchy（無均值）、零方差、p∈{0,1} 邊界、AR(1) 相依序列、小 n conformal——按 claim 家族自動套用；
2. 把 R2/R3 的候選反例逐個落地為腳本：先跑**假設檢查器**（陳述的 hypotheses 是否全部滿足），再斷言結論失敗。
3. 判決三態：`VALIDATED`（域內且結論失敗）/ `REJECTED_OUT_OF_DOMAIN`（機械駁回，不進 findings）/ `INCONCLUSIVE`（數值不穩，標記給 R5）。

**產出**：`battery/*.py`, `battery_results.jsonl`。
**糾錯**：這一輪本身就是糾錯機制——LLM 判斷退出迴路，`VALIDATED` 是可復現的 ground truth；腳本永久保留，R6 修正後每個對應腳本必須翻轉為 PASS/N-A，R8 全量重跑。

## R5 — 證據仲裁（Claude；Fable 或 Opus 單對話；成本：中）

**做什麼**：合流 R1–R4 四個 JSONL。裁決規則：
- `VALIDATED` 反例 → 自動 **BLOCKER**（不可投票推翻）；
- R2 假設差集 → **MAJOR**，除非原證明實際涵蓋（引文核對）；
- R3 缺口無反例支持 → **MAJOR(PLAUSIBLE)** 或 **MINOR**；
- PASS 只在義務逐項有證據時記入；
- R2/R3 衝突 → 不投票，生成**定向小問題清單**（R5b：一個 15 分鐘 ChatGPT 對話逐題裁決），而非多數決。

**產出**：`findings_final.jsonl`（BLOCKER/MAJOR/MINOR/EDITORIAL，每條附**修正規格**：新增假設的精確措辭、修正後陳述、證明補丁大綱）+ `fix_plan.md`（按嚴重度排序的施工單）。
**糾錯**：每條 CONFIRMED 必須有 battery 腳本或明確的「不可執行化理由」；否則降級 PLAUSIBLE 進 R5b。

## R6 — 修正 canonical EN（Claude Code；Fable 編排 + Sonnet 定位；成本：中）

**做什麼**：按 fix_plan 逐條施工。注意本檔 MathJax 內聯在超長行（Read 需 offset≥61），Sonnet 子代理只負責定位精確編輯範圍，主上下文只 Read 要改的行段（既有工作偏好）。每個 patch 記入 `patch_log.jsonl`（claim id → 舊文 → 新文 → finding id → 行號）。

**產出**：修正後 EN + `patch_log.jsonl`。
**糾錯**：
- 瀏覽器/MathJax 渲染 smoke check（超長行結構不可破壞）；
- **對修正後的陳述重跑對應 battery 腳本**：原 VALIDATED 反例必須因新增假設被域檢查器排除（翻轉為 REJECTED_OUT_OF_DOMAIN）——這是「修對了」的機械定義。

## R7 — ZH 語義同構（Claude Code；Sonnet 施工 + Fable 抽查；成本：低中）

**做什麼**：兩檔 line-parallel——每個 EN patch 按行號鏡像到 ZH，術語走既有 canonical 表（貫穿距離/撞壁等）。`parity.jsonl` 記 EN 行 ↔ ZH 行 ↔ 校驗結果。

**糾錯**：
- 行數差必須為零或兩側對稱；
- 每 3 個 patch 抽 1 個：由**看不到 EN patch 原文**的新 Sonnet/Haiku 呼叫做回譯，Fable 比對語義（假設與量詞不得丟失——「同一個錯誤被忠實翻譯兩次」正是要防的）。

## R8 — 新鮮眼複審 + Release Gate（ChatGPT 新對話 + Claude Code 收尾；成本：低中）

**做什麼**：
1. 對修正後檔案重新生成 stmt pack（新 hash），ChatGPT 全新對話**只複審被改動的 claims**（盲，不給舊 findings）；
2. Claude Code 重跑全量 battery；
3. 生成 `gate_report.json`：claim 覆蓋率、未解 BLOCKER/MAJOR 數、canary 通過率、EN/ZH parity 狀態、前後 hash。
4. **Footer 處置**：line 1628 的「經獨立多代理覆核」聲明，只能由 gate report 自動生成**有範圍、有 run ID**的版本（例如「33 個編號命題經 N 輪跨平臺對抗審計，run 2026-07-XX，反例回歸 M/M 通過」）；gate 不過則移除聲明。

**通過條件**：覆蓋完整；無未解 BLOCKER/MAJOR；全部 PASS 有義務級證據;battery 全綠；EN/ZH 假設與結論一致；修正後 hash 已複審。

---

## 輪次總覽

| 輪　| 平臺 / 模型　　　　　　　　　| 輸入　　　　　　　　　　| 產出　　　　　　　　　　　　| 主要糾錯機制　　　　　　　　　　|
| -----| ------------------------------| -------------------------| -----------------------------| ---------------------------------|
| R0　| Claude Code：Sonnet+腳本　　 | 兩份 HTML　　　　　　　 | freeze, claims.jsonl, packs | grep 決定論覆蓋核對　　　　　　 |
| R1　| ChatGPT Thinking　　　　　　 | stmt packs　　　　　　　| obligations.jsonl + 分級　　| 5 canary + 種子錯誤須進 T1　　　|
| R2　| ChatGPT 最強檔（新對話）　　 | T1 stmt packs　　　　　 | rederivations.jsonl　　　　 | 假設差集機械化；canary　　　　　|
| R3　| ChatGPT 最強檔（另一新對話） | T1+T2 full packs　　　　| proof_audit.jsonl　　　　　 | first-broken-step 引文制;canary |
| R4　| Claude Code：Sonnet+Python　 | claims + R2/R3 候選反例 | battery/*.py + results　　　| 可執行 ground truth；域檢查器　 |
| R5　| Claude：Fable/Opus　　　　　 | R1–R4 全部 JSONL　　　　| findings_final + fix_plan　 | 否決不對稱；衝突→定向追問(R5b)　|
| R6　| Claude Code：Fable+Sonnet　　| fix_plan　　　　　　　　| 修正 EN + patch_log　　　　 | battery 翻轉測試；渲染檢查　　　|
| R7　| Claude Code：Sonnet　　　　　| patch_log　　　　　　　 | 修正 ZH + parity.jsonl　　　| 盲回譯抽查；行數對稱　　　　　　|
| R8　| ChatGPT 新對話 + Claude Code | 修正後 packs　　　　　　| gate_report + 新 footer　　 | 新鮮眼 + battery 全量重跑　　　 |

## 配額分配邏輯

- **ChatGPT 承擔重推理**（R1/R2/R3/R8a）——這正好也是最需要與 Claude 去相關的環節；
- **Claude 承擔機械+落地**（R0/R4/R6/R7）——Sonnet 為主，Fable 只出現在 R5 仲裁與 R6 編排，每次都是短對話；
- 分級（T1/T2/T3）把最貴的盲推導限制在 ~10 條，炮台對其餘 claims 提供廉價覆蓋；
- 若某輪超一個對話的容量（R2 最可能），按 claim 切半：同 schema、同 canary 策略，兩個對話的 JSON 直接 concat，無需彼此可見。
