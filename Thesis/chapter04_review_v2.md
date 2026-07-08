# External Examiner Review (v2) — Chapter 4: Censoring-Aware Penetration Surrogate Modeling and Probabilistic Screening

**Thesis:** Master's thesis, Aalto University / Wärtsilä  
**Chapter reviewed:** 04_trajectory_surrogate_screening.tex, cross-referenced with 05_results.tex  
**Version note:** This is a revised review correcting severity calibration from v1; issues that were overstated as "proven errors" are re-categorised as "protocol ambiguities" or "strengthening opportunities" where the text already contains relevant boundary language.

---

## Overall Assessment

Chapter 4 is technically ambitious and unusually self-aware for a Master's thesis. It documents the ML pipeline with explicit selection biases, a quantified censoring audit, and a LONO ablation structure that genuinely tests generalization rather than in-domain interpolation. The argument structure is mostly sound and the hedging language is calibrated better than most theses at this level. Three issues need to be fixed before the chapter is fully defensible: a protocol ambiguity about which held-out set drove the λ_σ hyperparameter choice; a mismatch between the "conservative coverage" framing in Ch. 4's preamble and the LONO calibration collapse documented in Ch. 5; and an unacknowledged feature asymmetry in the SVGP–MLP comparison. Beyond these, four secondary issues are genuine gaps rather than style notes. Everything else is polish.

---

## CRITICAL ISSUES

### C1. λ_σ selection protocol: ambiguity, not proven contamination

Chapter 4 (line 721) states: *"The test set is never used for model selection or hyperparameter tuning."* Chapter 5 (line 430) then describes scanning λ_σ ∈ {0.5, 1, 2, 5} and selecting λ_σ = 5.0 because *"the near-cap RMSE and bias achieved a joint optimum (4.89mm and −0.66mm)"* — with those figures drawn from a "held-out set" table (Tab. 5.7, single seed s=99). There is no statement anywhere identifying which held-out partition (validation, test, or full-clean diagnostic) drove the near-cap scan.

To be clear: there is no evidence in the text that the test set *was* used for hyperparameter selection. The ambiguity is one of omission, not admission. But the omission matters because an examiner cannot verify the claim in line 721 from the text alone, and near-cap RMSE is precisely the slice on which the deployed model's headline metric is most sensitive.

**Required action:** Add one sentence in §5.4.3 identifying the partition that drove the near-cap ablation and the λ_σ scan — e.g., *"All ablation choices were evaluated on the validation set; the test set was queried only after λ_σ = 5.0 was fixed."* If the answer is more complicated (e.g., a diagnostic re-run on the full-clean set that does not constitute tuning), say so explicitly. The fix is one sentence; the cost of leaving it ambiguous is an examiner's inability to verify protocol integrity.

---

### C2. "Conservative coverage" framing is inconsistent with LONO calibration collapse

Chapter 4, line 10, defines *conservative coverage* as: *"The design intent is for the model's empirical 1σ/2σ coverage on the holdout set to meet or exceed the nominal Gaussian level (68.3%/95.4%)—a deliberate over-dispersion oriented toward early engineering screening rather than a calibration defect."*

Chapter 5, line 197, reports LONO 1σ/2σ coverage of **0.575 / 0.770** — 11 and 18 percentage points *below* nominal. The same chapter explains why (Nozzle0 cross-family extrapolation failure dominates), but this explanation appears in §5.3, several pages after the reader has absorbed the Ch. 4 preamble's conservative-coverage promise.

The LONO result does not invalidate the in-domain coverage finding; it limits it. The problem is that Chapter 4's preamble is read before Chapter 5's qualification, and the stated downstream purpose of the screening interface (§4.7) is precisely the cross-family new-nozzle scenario for which LONO calibration — not in-domain calibration — is the relevant figure. A reader who absorbs Ch. 4 in isolation will conclude that the model is conservatively calibrated for screening; that is not the complete picture.

Chapter 5 handles this correctly once the reader reaches §5.3. The gap is that Chapter 4 does not forward-reference it where the conservative-coverage claim is made.

**Required action:** In the Terminological Conventions preamble (around line 10), append to the conservative coverage definition: *"Whether this target is achieved is quantified in Chapter 5; note that under the LONO protocol, empirical coverage degrades to 0.575/0.770 — below nominal — due to cross-family extrapolation on the Nozzle0 fold."* Also add one sentence at the start of §4.7 (Screening Interface): *"Coverage estimates in this section assume in-domain deployment; cross-family screening calibration is addressed in Chapter 5."*

---

### C3. SVGP–MLP comparison is not feature-matched; the directional implication is untested

Section 4.5 introduces the SVGP on a five-feature set (line 670): *"[t_norm, θ_tilt,z, n_plumes,z, t_i,z, P_cb,z],"* omitting log P_inj,z and log P_ch,z. The MLP uses seven features (line 312). Chapter 4 reports that adding the log-pressure residuals reduced Stage 1 LONO MAE from 8.27mm to 7.46mm for the MLP. The SVGP, running without these features, still achieves lower RMSE on every evaluation protocol (4.193mm vs. 4.265mm in-domain; 4.628mm vs. 4.735mm full-clean; 10.16mm vs. 12.55mm LONO).

The chapter's explanation for the five-feature restriction — that ARD lengthscales handle relevance automatically — is a reasonable engineering argument for *not needing* to hand-tune the feature set, but it is not a substitute for actually testing the seven-feature configuration. For an ARD kernel, adding uninformative features can inflate the lengthscale optimization landscape, increase inducing-point placement difficulty, and interact non-trivially with a fixed M=256 budget. It is therefore not assured that the seven-feature SVGP would be stronger; the margin under equal features is genuinely untested.

What can be said: the comparison is not feature-matched, and the current results give the SVGP a structural disadvantage (fewer features) relative to the MLP. Since the SVGP wins despite this, the stated conclusion — *"the SVGP acts as a stronger point predictor"* — is directionally correct but understates its own robustness: the advantage is demonstrated under an unequal comparison and would likely survive (though not necessarily increase) under matched features.

A secondary asymmetry: the MLP headline metrics use 5-seed means (RMSE 4.265 ± 0.040mm), while the SVGP uses a single seed-42 run. This is acknowledged in the table caption (*"The SVGP row is the trained Stage 3 seed-42 reference model"*) but not foregrounded as a comparison caveat.

**Required action:** Add one sentence in §4.5: *"The SVGP was not evaluated with the full seven-feature set; whether this restriction widens or narrows its point-accuracy advantage over the MLP remains untested."* Add one sentence in §5.2 noting the seed asymmetry: *"The MLP figures are 5-seed averages; the SVGP figure is single-seed-42, so the effective variability of the SVGP estimate is unknown."*

---

## IMPORTANT ISSUES

### I1. Calibration is assessed only marginally — conditional stratification is absent

The thesis reports aggregate 1σ/2σ coverage (88.7%/98.9%) and reliability diagrams described as "binned by predicted standard deviation" (line 62 of Ch. 5). ECE is defined in §4.6 but no numerical ECE value appears in any results table. CRPS is shown in a CRPS–sharpness scatter plot but not tabulated. PIT histograms are mentioned but their shape is not interpreted in the text.

These are marginal calibration checks: they pool all operating conditions, time points, and nozzle families. For engineering screening, the pathology that matters most is calibration within subgroups — particularly the high-penetration tail (where wall impingement occurs), specific time windows (early SOI, post-EOI), and individual nozzle families. The per-folder RMSE spread shown in Fig. 5.7 (folder-level errors ranging ~3mm to ~9mm) confirms substantial heterogeneity that marginal calibration would average away.

This is not a structural gap in the methodology — the model itself is appropriately designed. It is a gap in the *evaluation* reported in the thesis.

**Required action:** Add at minimum: (a) a per-nozzle 1σ coverage table for the in-domain test set (9 values, one per nozzle family); (b) the PIT histogram shape interpreted in one sentence (U-shaped = under-dispersed, hump = over-dispersed, skewed = mean bias); (c) CRPS and ECE values added as columns to Table 5.2. If data exist for time-bin-stratified calibration, include a coverage-vs-time-bin figure — the q1 residual diagnostics already motivate expecting early/late miscalibration.

---

### I2. Gaussian NLL assumption: used without justification, PIT not interpreted

Stages 2 and 3 use a Gaussian NLL. The quantity modeled (spray penetration) is bounded below by zero, partially right-censored, and the q1 residual diagnostics (§4.3.4) show a U-shaped variance envelope with systematic boundary biases. The chapter never checks whether intra-condition residuals are approximately Gaussian after σ(t) conditioning, nor does it explain why Gaussian is preferred over, for example, a log-normal or a heavier-tailed distribution for a positive quantity.

The PIT histogram is the natural tool for this check and is already produced (Fig. 5.4, referenced in Ch. 5 line 176). But the histogram shape is not described or interpreted anywhere in the text; it is shown without comment. The reader cannot judge from the text whether the Gaussian likelihood is approximately satisfied.

**Required action:** In §5.2.2 (Probabilistic Calibration), add one sentence interpreting the PIT histogram shape: does it resemble a uniform distribution (consistent with Gaussian), or show U-shape (under-dispersed), hump (over-dispersed), or skewness (mean bias)? If the shape deviates materially from uniform, add a sentence acknowledging that the Gaussian NLL is approximate and that coverage statistics are affected accordingly. This does not require changing the model; it requires reading a figure that is already there.

---

### I3. LONO protocol definition in Ch. 4 does not match the actual scope in Ch. 5

Chapter 4, line 741, defines LONO as: *"a secondary LONO evaluation is performed: all operating conditions belonging to one nozzle family are removed from the training and validation splits, the model is retrained on the remaining families, and performance is evaluated solely on the held-out family. **This process is repeated for each nozzle family.**"*

Chapter 5, line 24, lists the actual folds: Nozzle0, Nozzle1, Nozzle2, Nozzle4, Nozzle5 — five of nine nozzle families. Nozzles 3, 6, 7, 8 are never held out. The discrepancy between "each nozzle family" (Ch. 4) and "five nozzle families" (Ch. 5) is never explained.

Separately, my v1 review incorrectly characterized Nozzle3 as "the hardest nozzle, never held out." This conflates two things. Nozzle3 is morphologically difficult (step-like trajectories violating the monotonic q1 prior, 11.9% flag rate), but the hardest *LONO failure* belongs to Nozzle0 (RMSE 34.76mm, dominant by point count). What can be said about excluding Nozzle3 from LONO folds is narrower: Nozzle3's elevated flag rate means a fraction of its training trajectories are already excluded by the q1 gate, so its influence on learned representations is attenuated. Holding Nozzle3 out would test whether the model can handle non-monotonic spray dynamics using a teacher trained exclusively on monotonic trajectories — an informative but different question from the Nozzle0 cross-design-family test.

**Required action:** Fix the Ch. 4 definition to match reality: state that LONO is evaluated on five representative nozzle families rather than all nine, and give the reason (e.g., compute budget, or the four excluded families were retained in training to preserve coverage of the supported condition space). If no reason exists, simply drop the phrase "this process is repeated for each nozzle family."

---

### I4. Aleatoric/epistemic uncertainty: existing language needs one sentence of strengthening

The current text (Table 4.6, line 814) already states that σ is *"not an epistemic or safety guarantee."* This is the right boundary but states what σ is *not* rather than what the epistemic gap practically implies. The 5-seed RMSE range (4.217–4.310mm) measures initialization sensitivity in scalar metrics but is not propagated into individual predictions.

**Required action:** Add one sentence to the uncertainty semantics row of Table 4.6: *"Epistemic uncertainty — arising from limited data coverage — is not captured in σ(t); the 5-seed RMSE spread (4.217–4.310mm) is a proxy for initialization sensitivity in aggregate metrics, not a per-prediction epistemic interval."* This is a strengthening of existing language, not a new claim.

---

### I5. Stage 3 interval thresholds: existing acknowledgment needs quantified consequence

The text (line 579) already states the 70%/20% thresholds are *"conservative engineering rules rather than optimal values drawn from a full grid search"* — and acknowledges this as a limitation. The two single-run validation checks (ΔRMSE ≈ −0.003mm and ≈ 0.065mm) are correctly characterized as "directionally consistent rather than statistically resolved." This is good hedging.

What is missing is a quantified statement of the practical consequence: with alternative boundaries differing by ±10 percentage points, RMSE could plausibly shift by ~0.065mm — larger than the 5-seed standard deviation (0.040mm) but smaller than the MLP-SVGP in-domain gap (0.072mm). Without this, the reader cannot judge whether the threshold choice is negligible or material relative to the comparison margins being argued.

**Required action:** Append one sentence to the existing acknowledgment: *"The single-run sensitivity estimate (~0.065mm RMSE shift) is directionally consistent but statistically unresolved; it is larger than the 5-seed RMSE standard deviation (0.040mm) and comparable to the MLP–SVGP in-domain point-accuracy margin (0.072mm), suggesting threshold uncertainty is non-negligible in close comparisons."*

---

## MINOR ISSUES

### M1. LONO ± notation misrepresents the distribution

Table 5.3 reports MLP LONO RMSE as "12.55 ± 12.46mm." The standard deviation is 99% of the mean, driven entirely by the Nozzle0 fold (34.76mm) against four folds in the 5–8mm range. The "±" notation implies symmetric variability; the actual distribution is highly right-skewed. The text addresses this by decomposing Nozzle0 separately, but the table entry itself is misleading in isolation.

**Suggested fix:** In the table footnote, add: *"The ± figure is cross-fold standard deviation, not a symmetric confidence interval; the distribution is right-skewed, dominated by the Nozzle0 fold (34.76mm) against a 4-fold range of 5–8mm."* Alternatively, report median ± IQR.

### M2. CRPS is a stated headline metric but never tabulated

CRPS is formally defined, cited as a primary evaluation metric alongside coverage, and displayed in a figure — but no numerical CRPS value appears in any table. Table 5.2 includes RMSE, MAE, bias, P95, and coverage, but not CRPS or ECE. If CRPS is positioned as a key probabilistic metric, it should appear in the headline table.

### M3. LONO in the Ch. 4 feature ablation (§4.4.4) uses only 5 folds, not the 9 implied

The Stage 1 LONO ablation in §4.4.4 holds out Nozzles 0, 1, 2, 4, 5, correctly described in the text as "a five-fold split." However, Table 4.5's caption says *"five leave-one-nozzle-family-out folds"* and then lists the five held-out families; this is consistent. The problem arises only in the downstream evaluation definition at line 741 (described above in I3). No fix needed in §4.4.4 itself.

### M4. Figure captions describe rather than conclude

Several captions state what the figure shows rather than what conclusion follows. Examples:

- **Fig. 4.5** (*"Residual structure of the q1 production fit aggregating all 24,316 cleaned trajectories"*): append *"—the central window [0.3–1.0ms] is well-constrained (σ_robust ≈ 0.6mm); boundary biases at early and late time motivate the three-stage curriculum."*
- **Fig. 5.1** (*"The dashed line represents the ideal identity line"*): append *"—scatter is approximately unbiased below ~80mm; the widening residual envelope above 80mm is consistent with the heteroscedastic design and elevated censoring at high penetration."*
- **Fig. 5.4** (calibration figure): the caption should state the shape conclusion, not just describe the axes.

### M5. Amplitude scaling R² comparison lacks reconciliation

§4.4.3 reports that A-scaling drops sparse-feature R² from 0.32 to 0.016, presented as evidence that amplitude scaling successfully removes condition variance. Table 4.3 (time-windowed log-log regressions) shows R² = 0.35–0.42 in the mid-trajectory window. These serve different purposes — the former is a pre-training check on the static features; the latter is a per-time-bin regression on the raw CDF observations — but they are not reconciled, and a reader could interpret 0.016 as evidence the model has almost no signal.

**Suggested fix:** Add one sentence: *"The 0.016 reflects variance in Ŝ not explained by the sparse non-time features alone; it does not indicate the full model has weak signal, since time itself is the dominant predictor of penetration magnitude."*

---

## PRIORITIZED ACTION LIST

**1. Clarify the λ_σ selection partition (C1) — one sentence in §5.4.3.**  
State which held-out set drove the near-cap ablation and λ_σ scan. This resolves the only unverifiable protocol claim in the thesis.

**2. Add conservative-coverage caveat to Ch. 4 preamble and §4.7 (C2) — two sentences.**  
Append to the Terminological Conventions definition and the start of the Screening Interface section that this is an in-domain property; LONO coverage is 0.575/0.770.

**3. Acknowledge the SVGP feature asymmetry (C3) — one sentence in §4.5 and one in §5.2.**  
State that the comparison is not feature-matched and that the margin under equal features is untested. Do not claim a directional outcome.

**4. Fix the LONO protocol definition mismatch (I3) — one edit in §4.6.**  
Change "repeated for each nozzle family" to the actual five-fold scope and give the reason for exclusion of Nozzles 3, 6, 7, 8.

**5. Add conditional calibration and tabulate CRPS/ECE (I1, M2) — one figure and two table columns.**  
Per-nozzle 1σ coverage (9 values) and PIT interpretation (one sentence). CRPS and ECE values in Table 5.2. If time-bin-stratified calibration data exist, add a coverage-vs-time-bin figure. This is the highest-effort item but the most important for substantiating the probabilistic screening claim.
