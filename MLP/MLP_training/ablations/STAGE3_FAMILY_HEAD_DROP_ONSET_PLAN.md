# Stage-3 Family-Aware Student + Onset-Head Removal — Implementation Plan

**Status:** implemented in code/config on 2026-05-29; pending a fresh production
retrain to generate the first `a_dp050_plus_pressures + family_head` Stage-2
teacher and Stage-3 student artifacts.
**Author/date:** 2026-05-29.
**Decision:** Option **C** (remove the onset auxiliary head) + promote the Stage-3
student from single-head `PenetrationMLP` to `FamilyAwarePenetrationMLP`.

---

## 1. Motivation

The Tier-3A LONO ablation (deck:
`Thesis/slides/slides_tier3a_family_head_ablation/`) established two facts:

1. **N0 and N1–N8 are empirically distinct** even after A-scale normalisation
   (678 k clean points; ratio `S*_N0 / S*_N1-N8` > 1 at *every* `t*`, peaking at
   2.4×). The family label carries non-redundant information beyond orifice
   geometry. → motivation slide (page 4).
2. The **family-aware μ-head** cuts the catastrophic OOD-family folds from
   ~30–37 mm RMSE to ~8–12 mm at no cost to in-family folds.

**The gap this plan closes:** that ablation was measured on the **Stage-2 teacher**.
Production deploys the **Stage-3 student**, which is *always* rebuilt as a
single-head `PenetrationMLP`
([`train_stage3...py:1776-1777`](../train_stage3_distillation_plus_raw_series.py#L1776)):

```python
# Student is always a PenetrationMLP regardless of teacher architecture.
student_train_config["architecture_mode"] = "single"
```

So distillation collapses the two μ-heads back into one, and the family routing
that wins the LONO folds never reaches the deployed model — `family_id` survives
only as an ordinary input feature into a single decoder. **We make the deployed
Stage-3 student family-aware so the proven OOD gain is actually shipped.**

### Why drop the onset head (Option C)

The onset auxiliary head (`output_dim=3`, channel 2 = `onset_logit`) was added to
supply a soft injection-onset anchor when the **time-axis onset alignment was
poor** (frame-to-frame onset detection drifted by ±~50 µs). Onset alignment has
since improved substantially, so the auxiliary anchor is no longer needed.

Two further facts make removal clean:

- **Inference already ignores the onset channel.** `split_mu_logvar` silently
  drops channel 2
  ([`efc/objectives.py:50-56`](../efc/objectives.py#L50)). The onset head only ever
  contributed an auxiliary *training* gradient; it never fed the deployed
  prediction.
- **`FamilyAwarePenetrationMLP` is hard-wired to `output_dim=2`**
  ([`efc/models.py:155-158`](../efc/models.py#L155)). Removing the onset channel
  is therefore a *precondition* for using it — the two changes are naturally
  coupled, not independent.

---

## 2. Precondition: the full family_head chain

`FamilyAwarePenetrationMLP.forward` treats the **last feature channel as
`family_id`**, strips it, and routes on it
([`efc/models.py:219-237`](../efc/models.py#L219)). For the Stage-3 student to
route correctly:

- The teacher's `feature_columns` **must already end with `family_id`**. That only
  happens when the feature table was built with `add_family_id=True`
  ([`efc/feature_engineering.py:588-589`](../efc/feature_engineering.py#L588)),
  which is driven by `architecture_mode=family_head`.
- The student warm-starts its trunk from the teacher's `trunk.*` keys
  ([`train_stage3...py:818-826`](../train_stage3_distillation_plus_raw_series.py#L818)).
  This only matches if the teacher is itself a `FamilyAwarePenetrationMLP`.

Architecture mode chains **Stage 1 → Stage 2** by inheritance + warm-start
([`trainers/stage2.py:195-206`](../trainers/stage2.py#L195)); a single-head Stage 1
cannot warm-start a family_head Stage 2 (head-shape mismatch).

**⇒ Hard requirement:** the production pipeline must run **Stage 1 and Stage 2 in
`family_head` mode** before this plan's Stage-3 change has any effect. This is also
the concrete answer to "should stages 1/2/3 share one architecture?": **yes** — the
`family_id` channel and warm-start chain force all three to agree.

### 2.1 Production feature variant: `a_dp050_plus_pressures`, not `a_only`

Do **not** build the production family-head chain on `a_only`. The thesis feature
screening section
(`Thesis/latex/sections_en/04_trajectory_surrogate_screening.tex`, around the
"final design adopted a seven-feature scaling-plus-residual-pressure
formulation" paragraph and the LONO ablation tables) promotes
`a_dp050_plus_pressures` as the production variant:

- `A = ΔP^0.5 · ρ_a^-0.25 · d_n^0.5` remains the amplitude scaling prior.
- `log_injection_pressure_bar_z` and `log_chamber_pressure_bar_z` are retained as
  residual pressure inputs, because pressure structure is not fully explained by
  the fixed global `ΔP^0.5` scaling.
- `diameter_mm_z` stays out of the direct MLP input; diameter remains absorbed in
  `A` because the diameter axis is sparsely supported and confounded.

Therefore the production Stage 1/2/3 teacher-student chain should use the exact
base feature list:

```python
[
    "time_norm_0_5ms",
    "tilt_angle_radian_z",
    "plumes_z",
    "injection_duration_us_z",
    "control_backpressure_bar_z",
    "log_injection_pressure_bar_z",
    "log_chamber_pressure_bar_z",
]
```

With `architecture_mode=family_head`, Stage 1 appends the routing channel:

```python
[
    "time_norm_0_5ms",
    "tilt_angle_radian_z",
    "plumes_z",
    "injection_duration_us_z",
    "control_backpressure_bar_z",
    "log_injection_pressure_bar_z",
    "log_chamber_pressure_bar_z",
    "family_id",
]
```

So the deployed family-aware Stage-3 student should have `input_dim=8`,
`trunk_in_dim=7`, and `output_dim=2`. The backward-compatible alias
`a_plus_pressures` maps to the same feature set, but new production configs should
prefer the explicit canonical token `a_dp050_plus_pressures`.

### 2.2 Verified state (2026-05-29) — precondition is NOT met by production

Audited `MLP/runs_mlp/`:

- **The standard production pipeline produces a SINGLE-HEAD teacher.**
  `run_full_pipeline.py` + the default `config/full_pipeline_config.json` has no
  `feature_ablations` / `stage2_ablations` block, so it takes the simple
  `run_one_seed` path, which calls `run_stage1` **without** `--architecture-mode`
  ([`run_full_pipeline.py:326-345`](../run_full_pipeline.py#L326),
  [`:977-993`](../run_full_pipeline.py#L977)). Stage 1 then defaults to `single`
  ([`trainers/stage1.py:88-89,150`](../trainers/stage1.py#L88)), so `family_id` is
  **not** appended to `feature_columns`.
- **The family_head teachers in `runs_mlp/` are Tier-3A sweep artifacts, not
  production.** `run_family_head_sweep.py` drives `run_lono_pipeline.py` with an
  explicit `--architecture-mode family_head`
  ([`ablations/run_family_head_sweep.py:130-138`](run_family_head_sweep.py#L130)).
  That is the **LONO leave-one-nozzle-out** harness (holds a nozzle out), not the
  full-data production pipeline.
- The latest full-data student
  (`distill_cdf_onset_v2_ablate_anchor_off_20260528_224217`, `lono_holdout=None`)
  *happened* to be distilled from a family_head sweep teacher
  (`stage2_..._223755`, `architecture_mode=family_head`, `feature_columns[-1]==
  "family_id"`), but that teacher is still an `a_only` artifact and did not come
  from `run_full_pipeline.py`. The *deployed* model, if regenerated via the
  standard pipeline, would be single-head and would also keep the wrong
  production feature variant unless the pipeline config is changed to
  `a_dp050_plus_pressures`.

**⇒ Required pre-work (implemented for new production runs):** make the **production** Stage 1/2 run in
`family_head` mode **on `a_dp050_plus_pressures`**. Two options:

| Opt | Change | Trade-off |
|-----|--------|-----------|
| **P1 (config-only, fastest)** | Use a config whose `variant` / feature entry is `a_dp050_plus_pressures`, and add a one-entry `feature_ablations` block with `variant: "a_dp050_plus_pressures"` and `args: {"architecture_mode": "family_head", "n_families": 2}`. This routes through `run_multilayer_ablation_pipeline`, which *does* pass `stage1_args` ([`run_full_pipeline.py:656-667`](../run_full_pipeline.py#L656)); Stage 2 inherits the mode automatically. | No code change. But switches the run to the multilayer output layout (`bootstrap_by_branch`, `seed_<n>/<feature>/...`) instead of the flat single-mode layout. |
| **P2 (first-class, recommended for clean "production model" semantics)** | Use `config/full_pipeline_dp050_pressures_config.json` (or set `variant: "a_dp050_plus_pressures"`), add `--architecture-mode` / `--n-families` to `run_full_pipeline.py` argparse, and thread them into the simple-path `run_stage1`/`run_stage2` calls ([`:949`](../run_full_pipeline.py#L949), [`:961`](../run_full_pipeline.py#L961), [`:446`](../run_full_pipeline.py#L446)). | Small code edit; keeps the single-seed "produces the production model" path and its flat output layout. |

`family_head_dims` (default `[128]`) and `fallback_family_id` (default `1`) need
not be passed — their `build_model` defaults match the sweep
([`efc/models.py:240-252`](../efc/models.py#L240)).

**Implementation gate:** before running Stage 3, the teacher run feeding the
deployed Stage 3 must satisfy:

1. `variant == "a_dp050_plus_pressures"` (or the equivalent legacy alias
   `a_plus_pressures`, but prefer the canonical token for new runs);
2. `architecture_mode == "family_head"`;
3. `feature_columns == ["time_norm_0_5ms", "tilt_angle_radian_z", "plumes_z",
   "injection_duration_us_z", "control_backpressure_bar_z",
   "log_injection_pressure_bar_z", "log_chamber_pressure_bar_z", "family_id"]`.

The code/config now implements P2: `run_full_pipeline.py` threads
`architecture_mode` / `n_families` into the simple production path, and the
default full-pipeline configs set `architecture_mode="family_head"`,
`n_families=2`, and `variant="a_dp050_plus_pressures"`. Existing historical run
directories are unchanged; rerun the pipeline and verify the freshly produced
Stage-2 run before launching Stage 3.

---

## 3. Change surface

All edits are in
[`train_stage3_distillation_plus_raw_series.py`](../train_stage3_distillation_plus_raw_series.py)
unless stated otherwise. Grouped by concern.

### 3.1 Make the student family-aware

| Loc | Current | Change |
|-----|---------|--------|
| `build_student_model_from_teacher` (L797-877) | builds `PenetrationMLP(output_dim=3)` and remaps `trunk.*`→`net.*`, splicing `mu_heads.0`/`log_var_head` into a flat final layer | build `FamilyAwarePenetrationMLP(output_dim=2, n_families, family_head_dims, fallback_family_id from teacher_config)`; copy teacher `state_dict` **verbatim** (trunk + mu_heads + log_var_head + `trained_families` buffer all match key-for-key — no remap needed since student == teacher architecture) |
| L1064 | `... .reshape(batch_size, n_points, 3)` | `.reshape(batch_size, n_points, 2)` |
| L1773 | `student_train_config["output_dim"] = 3` | `= 2` |
| L1776-1777 | forces `architecture_mode = "single"` | set `= "family_head"` (and carry `n_families`, `family_head_dims`, `fallback_family_id` through, mirroring `build_model`'s family branch) |

**Note on warm-start simplification:** once the student is the same class as the
teacher, the entire `is_family_head` remap block
([L818-869](../train_stage3_distillation_plus_raw_series.py#L818)) collapses to a
direct `student.load_state_dict(teacher_model.state_dict())`. The `output_dim=3`
splicing logic (L832-874) is deleted with the onset head (§3.2). Keep the
`trained_families` buffer from the teacher so eval-time fallback routing for a
held-out N0 still works ([`efc/models.py:202-217`](../efc/models.py#L202)).

### 3.2 Remove the onset auxiliary head + loss

**Delete (head/loss path):**

| Loc | Item |
|-----|------|
| L750-751, L762-763 | `onset_target`, `onset_loss_mask` tensors built in `_build_sample` |
| L880-882 `split_student_output` | drop the 3-way split; replace with `split_mu_logvar` (2-channel) — or delete and reuse the shared helper |
| L953 | `mu_hat, log_var_hat, onset_logit = split_student_output(...)` → `mu_hat, log_var_hat = split_mu_logvar(...)` |
| L970-971 | reads of `onset_target`, `onset_loss_mask` |
| L990-992 | `onset_bce`, `onset_loss` computation |
| L1007 | `+ float(config["lambda_onset"]) * onset_loss` term in total loss |
| L1017, L1048 | `"onset_bce"` metric keys (dict init + totals init) |
| L1144-1149 | CLI flags `--lambda-onset`, `--onset-ramp-ms`, `--onset-loss-window-ms` |
| L1260-1262 | arg→config wiring for the three onset keys |
| L1540, L1546-1547 | config defaults `lambda_onset`, `onset_ramp_ms`, `onset_loss_window_ms` |

**KEEP (do NOT delete — confusingly named, but unrelated):**

- `_build_onset_bins` (L692-702) and `self.onset_bins` (L672) — these compute, per
  trajectory, the time-bin of the **first observation**, used only by the
  `WeightedRandomSampler` (L788-789) to balance early- vs late-onset trajectories.
  This is sampling balance, not the auxiliary head. Removing it would silently
  change the training sample distribution. Leave it intact. *(Optional, separate:
  rename to `first_obs_bins` for clarity — out of scope here.)*

**Docstring:** delete the "Onset auxiliary head" section (L40-45) and update the
header note that says output is `output_dim=3` (L794).

### 3.3 Run-name / cosmetic

- `--run-name-prefix` default `"distill_cdf_onset_v2"` (L1130) and the sanitiser
  fallback (L1250) reference "onset". Rename to e.g. `"distill_cdf_family_v2"` so
  artifact directories aren't misleading. Low priority; cosmetic only.

---

## 4. Inference compatibility

No inference-side code change required:

- `load_run_artifacts` → `build_model(config)` reads `architecture_mode`; with the
  saved student config now `"family_head"`, it reconstructs a
  `FamilyAwarePenetrationMLP`
  ([`efc/inference.py:220-226`](../efc/inference.py#L226)).
- Output-dim inference is *skipped* for non-single architectures
  ([`efc/inference.py:216-225`](../efc/inference.py#L216)); the saved
  `output_dim=2` is used directly.
- `split_mu_logvar` already returns `(mu, log_var)`; with a 2-channel model it is a
  no-op change from the consumer's side.
- The inference feature matrix is built from the saved `feature_columns`, which
  ends with `family_id` (precondition §2), so `forward`'s `x[..., -1]` routing key
  is present.
- The `trained_families` buffer is part of `state_dict`, so it round-trips through
  the checkpoint and eval-time fallback (held-out N0 → family 1) works.

---

## 5. Validation

1. **Smoke / tiny-overfit:** the existing tiny-overfit block
   (L1730-1748) must still drive loss down with the family-aware student and no
   onset term. Confirm no shape errors at the L1064 reshape (now `2`).
2. **State-dict load:** assert `student.load_state_dict(teacher.state_dict())`
   succeeds with `strict=True` (same class ⇒ exact key match incl.
   `trained_families`).
3. **Parity run (single seed, seed 42):** run Stage 3 on the production teacher,
   compare student eval RMSE vs the previous single-head student. Expect:
   in-family folds within seed noise; **OOD-family fold (N0) materially better**,
   carrying the Tier-3A gain through distillation rather than discarding it.
4. **LONO replay:** re-run the `lono_5fold` Stage-3 student for the N0 fold;
   confirm the held-out-N0 RMSE now tracks the family-head teacher (~11–12 mm)
   rather than regressing toward the single-head collapse (~30 mm).
5. **Onset-removal sanity:** confirm deployed predictions are *bit-for-bit
   unaffected by the onset removal alone* when architecture is held fixed — since
   `split_mu_logvar` already dropped channel 2, removing it changes only the
   training gradient, not the inference graph. (Use this to isolate the onset
   change from the architecture change if the parity run surprises.)

---

## 6. Risks & rollback

- **CONFIRMED (not just a risk): production teacher gate is not satisfied.**
  Verified 2026-05-29 (§2.1-§2.2): `run_full_pipeline.py` + default config
  produces a single-head Stage 1/2 with no `family_id` channel, and the most recent
  family-head artifacts checked were `a_only`, not the thesis production
  `a_dp050_plus_pressures` variant. Switching the Stage-3 student to
  `FamilyAwarePenetrationMLP` against such a teacher would either mis-route (it
  would strip a real feature as `family_id`) or deploy the wrong feature basis.
  *Mitigation:* the §2.2 pre-work (P1 or P2) is **mandatory and must land first**;
  then re-verify the gate on the freshly produced family_head + pressures Stage 2.
- **Risk: onset head was silently helping early-time μ accuracy.** The premise is
  that improved onset alignment makes it redundant; the parity run (§5.3) with an
  early-time RMSE breakdown is the check. *Rollback:* the onset removal is
  self-contained — revert §3.2 to restore `output_dim=3` and the BCE term;
  `FamilyAwarePenetrationMLP` cannot carry channel 3, so a rollback that keeps the
  onset head also reverts to the single-head student (the two are coupled, as
  noted in §1).
- **Risk: capacity bump from per-family heads overfits the tiny family-0 set
  (1 nozzle).** Already mitigated by the shared `log_var_head` design
  ([`efc/models.py:128-135`](../efc/models.py#L128)) and eval-time fallback.

---

## 7. Edit checklist (for implementation)

- [x] **Pre-work (mandatory, §2.1-§2.2): make production Stage 1/2 use
      `a_dp050_plus_pressures` + `family_head`** via P1 (config
      `feature_ablations` entry) or P2 (`run_full_pipeline.py` CLI). The default
      pipeline config now uses the canonical pressures variant and family-head
      architecture.
- [ ] Gate: on the freshly produced Stage 2, verify
      `variant=="a_dp050_plus_pressures"`, `architecture_mode=="family_head"`,
      `input_dim==8`, `output_dim==2`, and exact `feature_columns` from §2.1
      ending in `family_id`.
- [x] `build_student_model_from_teacher`: build `FamilyAwarePenetrationMLP`,
      verbatim `load_state_dict`; delete `is_family_head` remap + onset splice.
- [x] L1064 reshape `3` → `2`.
- [x] L1773 `output_dim` `3` → `2`.
- [x] L1776-1777 `architecture_mode` `"single"` → `"family_head"` + carry
      `n_families` / `family_head_dims` / `fallback_family_id`.
- [x] `refinement_loss`: drop onset unpack, BCE, loss term, metric.
- [x] Dataset `_build_sample`: drop `onset_target` / `onset_loss_mask`.
- [x] CLI + config: drop `lambda_onset`, `onset_ramp_ms`, `onset_loss_window_ms`.
- [x] Metric dicts (L1017, L1048): drop `onset_bce`.
- [x] Docstring: drop onset section (L40-45), fix L794 header comment.
- [x] KEEP `onset_bins` / `WeightedRandomSampler` untouched.
- [x] Rename run prefix `distill_cdf_onset_v2` → `distill_cdf_family_head_v3`.
- [ ] Validation §5 (smoke, state-dict, parity, LONO replay).
