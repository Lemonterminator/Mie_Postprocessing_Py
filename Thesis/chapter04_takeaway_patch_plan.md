# Chapter 4 Narrative-Coherence Patch Plan (aggregated from 10 parallel evaluations + coherence critic)

**Date:** 2026-07-07
**Input:** ChatGPT's takeaway-per-subsection proposal for `latex/sections_en/04_trajectory_surrogate_screening.tex`
**Method:** one Sonnet evaluator per change point (each read the actual section text, the full `chapter04_review_v2.md`, and the defense deck), then a cross-proposal coherence critic. No edits applied yet.

---

## Headline finding

**ChatGPT's list was written against a stale version of the chapter.** Commit `7f77106` (today, "Ch4/Ch5 revision") already landed the review-v2 required fixes (C2 in-domain caveat ×2, C3 SVGP feature-asymmetry sentence, I3 five-fold LONO definition) *and* most of the takeaways ChatGPT proposes. Of the 10 change points, **6 are skips** (content already present, often verbatim) and **4 are genuine gaps** — all concentrated in the curriculum zone (§4.5 lead, Stage 1 close, Stage 3 close) plus one capping sentence at the end of §4.4.4.

## Verdict table

| # | Change point | Verdict | Why |
|---|---|---|---|
| 1 | Curriculum design-logic lead (§4.5, line 467) | **ADAPT** | ChatGPT's paragraph would near-duplicate lines 89 + 469. Genuine gaps: stage *ordering* rationale never echoed here; Stage 1's warm-start motivation buried at line 495. → 2 new sentences, not ChatGPT's paragraph. |
| 2 | q1 "compressor, not physical law" | SKIP | Already stated 4× (lines 169, 176, 194, 202). The 237-section already ends with the exact Stage-3 foreshadowing ChatGPT wants. A 5th restatement = anti-pattern. |
| 3 | Feature-ablation verdict (§4.4.4) | **ADAPT** | The load-bearing/residuals/diameter recap already exists at line 464 (+ 411). Missing: the *methodological* claim that the feature set is **LONO-licensed, not hand-tuned**. → 1 capping sentence appended to line 464. |
| 4 | Stage 1 takeaway | **ADAPT** | Genuine gap — subsection ends on a bare metric+pointer (line 495). → 2 sentences: MAE non-comparability caveat + warm-start-deliverable verdict. |
| 5 | Stage 2 takeaway + anchor verdict | SKIP | (a) "allocate variability into σ" is already the closing sentence at 556, near-verbatim. (b) **ChatGPT's anchor verdict doesn't belong in Ch4**: no anchor mechanism is explained in Ch4 Stage 2 (first appears line 637 as a cross-ref); the verdict already exists next to its evidence in Ch5 (line 361, `tab:stage2-anchor-ablation`). Also the μ vs μ+σ anchor gap is 0.32 mm ≈ the ~0.3 mm single-seed tie threshold — "σ-anchor over-constrains uncertainty" would overclaim. |
| 6 | Stage 3 scope verdict | **ADAPT** | Genuine gap — nothing in 565–664 says Stage 3's raw/teacher blending is orthogonal to cross-family (Nozzle0) generalization. → new bolded "Scope of the Distillation Refinement" paragraph before line 663. Deliberately does NOT cite the N0 few-shot numbers (different model lineage — residual head, not Stage 3 KD). |
| 7 | SVGP conditional verdict | SKIP | C3 sentence already at line 678 verbatim; line 680 already gives a *stronger* conditional verdict naming the three real MLP-retention reasons (onset side-branch, KD/raw curriculum, ablation affordability). ChatGPT's "winner-takes-all" phrasing is deck register and a downgrade. |
| 8 | Screening proxy-not-probability | SKIP | Present **5×** (lines 10, 746, 793, 795, 818). If anything, a future pass should *trim* 793/795/818, never add a 6th. |
| 9 | Ch4→Ch5 handoff + I3 fix | SKIP | All three duties landed in commit 7f77106: chapter-final handoff paragraph (825), five-fold LONO definition with reason (741), per-protocol purpose statements (723, 741). |
| 10 | Deck-language transfer + LLM analogy | Audit only | "same target, different inductive bias" already at 668; "screening proxy" already 5×. The "data→loss→purpose" sentence was **dropped by the critic** (collides with insertion #1). **LLM analogy stays OUT of Ch4**: Ch1:57 + Ch7:58 already form a deliberate, citation-anchored bookend (`Zhao2023LLMSurvey`, `Bommasani2021FoundationModels`); Ch4's curriculum is the concrete instance of Ch1's item 4, so a third echo dilutes the frame. ChatGPT's instinct agrees with what the thesis already does. |

---

## The four approved insertions (post-critic wording — ready to paste)

Apply **bottom-up by anchor line** so insertions don't shift later anchors: P3 → P2 → P1 → P4.

### P1 — Curriculum lead (new paragraph immediately after `\label{sec:mlp-curriculum-en}`, line 467; existing paragraph at 469 stays untouched)

```latex
The ordering of the three stages is deliberate, not incidental, continuing the conclusion drawn
from the censoring audit in Section~\ref{sec:cdf-spatial-censoring-audit}: each stage addresses a
specific limitation of the optical archive established earlier in this chapter---the
trajectory-level noise that demands a stable condition-level mean, the inter-plume variability
that demands a genuine conditional spread instead of a single mean curve, and the field-of-view
censoring itself---so that failures remain diagnosable to a single stage rather than silently
absorbed into one end-to-end fit. Stage 1 exists specifically to give Stage 2 a well-conditioned
starting point, so that the more expressive heteroscedastic fit is not asked to learn the mean
trend and the inter-plume spread from scratch simultaneously.
```

Critic fixes baked in: limitation list now maps 1:1 to the three stages; "deliberate, not incidental"; "well-conditioned starting point" (avoids verbatim echo of line 495's "numerically stable initialisation").

### P2 — Stage 1 close (new paragraph immediately after line 495, before the figure block)

```latex
This test MAE is not directly comparable to the Stage 3 results reported in Chapter 5: it is
measured on a single curated representative trajectory per condition under the amplitude-scaled,
shape-constrained objective, not the full held-out CDF protocol used downstream. Stage 1's
contribution to the curriculum is accordingly the well-behaved warm-start checkpoint from which
Stage 2 learns the conditional spread, not a stand-alone accuracy claim.
```

Critic fixes baked in: warm-start clause de-duplicated (was the 3rd repetition within 30 lines); "headline metrics" → "results"; no "anchor" vocabulary (reserved for the λ_anchor loss term first introduced at line 637).

### P3 — Stage 3 scope (new paragraph after the `\end{figure}` that closes `fig:distillation-loss` (~line 661), BEFORE the existing closing paragraph at 663)

> ⚠️ `\end{figure}` is not unique — match on `\label{fig:distillation-loss}` + following `\end{figure}`.

```latex
\textbf{Scope of the Distillation Refinement.} Stage 3 should be read as a censoring-aware
refinement of the observation window---its raw/teacher blending operates over the same training
population already used by Stages~1 and~2---rather than as a remedy for missing design-family
coverage. No new nozzle-family information enters the pipeline at this stage: the feature design
that carries whatever cross-family generalization the surrogate has was fixed
earlier---motivated by the scaling regressions and licensed by the LONO feature audit of
Section~\ref{sec:stage1-lono-ablation}. Accordingly, the Nozzle0 cross-design-family collapse
that dominates the LONO evaluation (Chapter~5, Section~\ref{sec:lono-transfer-en}) reflects that
earlier feature-and-population design, not a shortcoming this refinement stage is positioned to fix.
```

Critic fixes baked in: attribution aligned with P4's "licensed by, not selected by" framing (was the one genuine cross-proposal contradiction); recap clause compressed. "censoring-aware" is a deliberate first-use coinage in sections_en — use it exactly once.

### P4 — Feature ablation cap (append to the END of the line-464 paragraph, same paragraph, no blank line)

```latex
 This is the ablation's methodological payoff: the production feature set is licensed by the
multi-family LONO evidence in Table~\ref{tab:stage1-lono-ablation-summary}, not by hand-tuning
against the single-split accuracy quoted earlier in this section---only a protocol that holds out
an entire nozzle family can distinguish a feature that memorizes nozzle identity from one that
generalizes across families.
```

Critic fix baked in: "only the leave-one-nozzle-out protocol" softened to "only a protocol that holds out an entire nozzle family".

---

## Cross-cutting guardrails from the critic

- All four insertions verified factually safe: no coverage overclaim, no winner-takes-all SVGP language, no OOD-recovery claim for Stage 3, no nine-fold LONO claim, no sub-0.3 mm ordering claims.
- "rather than" tic: chapter already has 53 occurrences; final wording keeps only 2 new ones ("rather than silently absorbed", "rather than as a remedy"). Don't add more in later passes.
- Ch5-pointer ceiling reached (~8 chapter-wide after P2+P3): no further "quantified/addressed in Chapter 5" sentences.
- "memorize(s) nozzle identity": at 2 occurrences (411 + P4) — a deliberate callback; no third instance.
- All `\ref` targets verified to exist: `sec:cdf-spatial-censoring-audit` (EN:40), `sec:stage1-lono-ablation` (EN:401), `sec:lono-transfer-en` (05_results:207), `tab:stage1-lono-ablation-summary` (EN:436). cleveref not loaded → `\ref`, not `\cref`, is correct.
- **ZH twin:** mirror all four insertions into `latex/sections_zh/04_...tex` at the parallel anchors (~2-line offset per critic spot-check; ZH reuses the same `-en` label names so `\ref` targets carry over). Verify offsets at application time.
- Pre-existing copyedit items surfaced (not this pass): British/American spelling mix (initialisation/initialization); "Nozzle0" vs "nozzle 0" inconsistency.
