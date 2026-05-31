# Tier-3 Accuracy Ablations — Nozzle-Family-Aware Architecture & Learnable Shape Penalty

**Purpose**: directly attack the two specific failure modes §5 documents:

- **Nozzle-0 cross-design-family failure** — MLP LONO RMSE 34.76 mm; SVGP
  on the same fold reaches 26.99 mm. The thesis explicitly flags the OOD
  claim as "conditional" without explicit family modelling.
- **Nozzle-3 step-like trajectory morphology** — the d2 concavity prior
  creates systematic bias on the wake-driven catch-up regime documented in
  §5 Figs. 12a-d.

Two sub-campaigns:
- **3A — Family-aware head with shared backbone** (factory-fresh vs modified)
- **3B — Learnable / per-family λ_d2 concavity weight**

These are not parameter sweeps — both require code changes to model + losses.
They are higher-risk than Tier 1/2 but offer the largest potential headline
wins. **Run only AFTER Tier-1B has established σ_seed**, otherwise any 3A/3B
delta cannot be interpreted against noise.

This brief is self-contained — a fresh agent / Codex run can pick it up cold.

---

## 1. Context

### 1.1 Nozzle 0 = single-point factory-fresh reference

The dataset has 6 nozzles in the LONO protocol (§5.3):

- **Nozzle 0**: factory-fresh; the only sample of that design family.
- **Nozzles 1–5**: modified injectors (different drill / asymmetric /
  wear states).

In LONO 5-fold (§5 Table 3):
- 4-fold (excluding N0): MLP 7.00 ± 1.18 mm vs SVGP 5.95 ± 1.98 mm.
- N0 fold: MLP 34.76 mm vs SVGP 26.99 mm.
- Aggregate 5-fold: MLP 12.55 vs SVGP 10.16 mm.

The MLP has **no mechanism** to learn that N0 is qualitatively different.
The thesis explicitly notes:

> "Without a second factory-fresh nozzle or explicit cross-family
> regularization, the OOD claim remains conditional." — §5

### 1.2 Nozzle 3 step-like morphology

§5 Figs. 12a-d show N3 trajectories exhibiting a leading liquid mass +
wake-driven catch-up of the main jet. This violates the smooth-monotone-
concave prior baked into Stage-1 loss via:

- `d1_positive_weight = 5e-5` — monotone increasing (rarely violated)
- `d2_concave_weight = 5e-4` — concave after `d2_start_ms = 0.9` ms

Both at `MLP/MLP_training/efc/data_io.py:167-169`. The d2 penalty in
particular forces the model to *smooth out* steps. Result: N3 in-distribution
test RMSE is ~9 mm vs 3-5 mm for other nozzles (§5 Fig. 10).

### 1.3 Strategy

- **3A** introduces a shared backbone + family-specific final head. We
  discussed this in conversation as "Option C" (shared backbone with
  family-routed head). Two LONO protocols are run — see §3.3.
- **3B** makes λ_d2 a learnable per-family parameter so N3-family can opt
  out of the concavity prior locally. The simplest version is a 2-family
  (factory-fresh vs modified) split; a follow-up could go finer.

---

## 2. Goals

| Sub | Output | Acceptance |
|---|---|---|
| 3A | `family_head_lono.csv`: A=single baseline / B=modified-only narrow / C=shared backbone + family head | C reduces in-family LONO mean (5 modified folds) by ≥ 1 mm without degrading any individual fold by > 2 mm. C cannot improve the N0 fold in honest LONO (zero training data for family-0 head) — that's reported as expected. |
| 3B | `learnable_d2_lono.csv`: baseline / `λ_d2=0` / `λ_d2 learnable per family` | N3 fold RMSE improves by ≥ 1 mm without overall LONO 5-fold mean degrading by > 0.3 mm. |

Both bars are high. **Acceptable null result**: documented as future work
with clear evidence of *why* it didn't help.

---

## 3. Sub-campaign 3A — Family-aware head (shared backbone + split head)

### 3.1 Architecture

Goal: most of the backbone is shared across all nozzles; the final 1-2
layers are family-specific. Family ID at training and inference time is
deterministic from `experiment_name`:

```python
family_id = 0 if "Nozzle0" in experiment_name else 1
```

Add a new class to `MLP/MLP_training/efc/models.py` (next to `PenetrationMLP`):

```python
class FamilyAwarePenetrationMLP(nn.Module):
    """Shared trunk + per-family final mu heads; shared log_var head.

    family_id is passed as the LAST channel of the input vector. It is
    stripped before the trunk, used to route mu, and ignored by the shared
    log_var head.

    log_var is SHARED across families on purpose: family-0 has 1 nozzle of
    data, which is not enough to learn variance reliably.
    """

    def __init__(
        self,
        input_dim: int,                                 # includes family_id channel
        hidden_dims: Sequence[int],
        family_head_dims: Sequence[int] = (128,),       # extra family-specific layers
        n_families: int = 2,
        dropout: float = 0.3,
        activation: str = "gelu",
    ):
        super().__init__()
        self.n_families = int(n_families)
        trunk_input_dim = input_dim - 1                 # last channel is family ID
        self.trunk = _build_mlp_trunk(trunk_input_dim, hidden_dims, dropout, activation)
        trunk_out = int(hidden_dims[-1])
        self.mu_heads = nn.ModuleList([
            _build_family_head(trunk_out, family_head_dims, output_dim=1)
            for _ in range(self.n_families)
        ])
        self.log_var_head = _build_family_head(trunk_out, family_head_dims, output_dim=1)

    def forward(self, x):                                # x: [B, T, F+1] or [B, F+1]
        family_id = x[..., -1].long()
        features = x[..., :-1]
        h = self.trunk(features)                         # shared embedding
        mu = torch.zeros(*h.shape[:-1], 1, device=h.device, dtype=h.dtype)
        for f in range(self.n_families):
            mask = (family_id == f)
            if mask.any():
                mu[mask] = self.mu_heads[f](h[mask])
        log_var = self.log_var_head(h)
        return mu.squeeze(-1), log_var.squeeze(-1)
```

Helpers `_build_mlp_trunk` and `_build_family_head` should mirror the existing
trunk-building code in `PenetrationMLP` so dropout/activation/LayerNorm
behaviour is identical.

### 3.2 Data plumbing

Add the `family_id` feature to the canonical table. In
`MLP/MLP_training/engineered_feature_common.py`, locate
`build_canonical_feature_table` and `build_variant_feature_table` (search for
those names). After feature columns are assembled and z-scored:

```python
# DO NOT z-score family_id. Append it AFTER the z-score loop.
df["family_id"] = df["experiment_name"].astype(str).map(
    lambda n: 0 if "Nozzle0" in n else 1
).astype("int8")
feature_columns.append("family_id")   # always last channel
```

Also: add a `--family-aware` boolean CLI flag to all three trainers
(stage1.py, stage2.py, stage3 driver). When set:

- `extra_overrides` returns `{"family_aware": True, "n_families": 2}`.
- `build_model` returns `FamilyAwarePenetrationMLP(...)` instead of
  `PenetrationMLP(...)`.
- The Stage-1 feature table builder appends the `family_id` channel.

### 3.3 The 3 architecture × 2 protocol matrix

| run | architecture | training data |
|---|---|---|
| A_single_baseline | single PenetrationMLP | All 6 nozzles |
| B_modified_only | single PenetrationMLP | Only N1-N5 |
| C_family_head | FamilyAwarePenetrationMLP | All 6 nozzles |

LONO protocols:
- **lono_5fold**: hold out one nozzle at a time among {N0…N5} — the
  current §5 protocol.
- **lono_modified_only**: hold out one nozzle at a time among {N1…N5}.
  N0 always in training. Reports the family-head advantage where it can
  manifest.

Required runs:
- A on lono_5fold (already exists from §5; reuse if results are recent).
- B on lono_modified_only (B never sees N0; 5 folds among modified).
- C on lono_5fold (honest LONO — N0 fold cannot benefit from family-0
  head because family-0 head has no training data when N0 is held out;
  see Pitfall 3 below).
- C on lono_modified_only (where the family head can shine).

Total: **20 fold-runs** (A: 5, B: 5, C: 10) + reuse A if available.

### 3.4 Driver

Extend `run_lono_pipeline.py` with two new CLI flags:

```python
parser.add_argument("--architecture-mode", choices=["single", "family_head"],
                    default="single")
parser.add_argument("--protocol", choices=["lono_5fold", "lono_modified_only"],
                    default="lono_5fold")
```

The `lono_modified_only` protocol filters the available nozzles to N1-N5
before iterating folds:

```python
if args.protocol == "lono_modified_only":
    nozzles = [n for n in nozzles if "Nozzle0" not in n]
```

Run commands:

```bash
# A baseline (use existing §5 results if recent)
python run_lono_pipeline.py --architecture-mode single --protocol lono_5fold --seed 42

# B modified-only narrow scope
python run_lono_pipeline.py --architecture-mode single --protocol lono_modified_only --seed 42

# C family head, both protocols
python run_lono_pipeline.py --architecture-mode family_head --protocol lono_5fold --seed 42
python run_lono_pipeline.py --architecture-mode family_head --protocol lono_modified_only --seed 42
```

### 3.5 Output

`MLP/MLP_training/ablations/family_head_<date>/`:
- `family_head_lono.csv` — columns: `architecture, protocol, fold_nozzle,
  rmse_mm, mae_mm, cov_1sigma, cov_2sigma, params_total, params_per_family`
- `family_head_summary.md` — cross-tabulated comparison:
  - **C vs A on N0 fold**: expected null (no family-0 training data).
  - **C vs A on N1-N5 folds (in lono_5fold)**: does sharing trunk help here?
  - **C vs A on lono_modified_only**: family head's main shot at improving.
  - **B vs A on modified nozzles**: does narrowing scope improve modified-
    family RMSE? (this is the "honest narrowing" alternative framing).

### 3.6 Budget 3A

- A (reuse if available): 0 min, else 5 folds × 13 min = 65 min
- B: 5 folds × ~10 min (smaller training set) = 50 min
- C × 2 protocols × 5 folds × ~13 min = 130 min
- **Total: ~3 - 4 h**

### 3.7 Pitfalls 3A

1. **Parameter count fairness**: `FamilyAwarePenetrationMLP` has more
   parameters than baseline (per-family heads). Report `params_total` and
   `params_per_family_at_inference` (= trunk + 1 family head + log_var head)
   in the CSV. If C wins purely by capacity, the win isn't really
   "family-aware".
2. **Family imbalance in batches**: modified family has ~5× the data. Without
   per-family batch balancing, family-0 head sees minimal gradient per
   batch. **Add a `WeightedRandomSampler`** that draws ~equal counts from
   each family per batch (or a stratified sampler). Without this, the
   family-0 head will under-train.
3. **N0 family-0 head with no data (in `lono_5fold`)**: when N0 is held out,
   the family-0 head sees zero training examples (all N0 → test). At
   inference, predictions from `mu_heads[0]` are whatever random init
   ended up being.

   Two options for inference-time fallback:
   - **Fallback A (recommended)**: detect "family-0 head has 0 training
     samples" → at inference, route ALL test points through `mu_heads[1]`.
     Family-1 head fallback is identical to the single-MLP baseline (modulo
     trunk regularization). Report as "N0 fold equivalent to baseline".
   - **Fallback B**: do not fall back; report the raw garbage from the
     untrained family-0 head as a negative result. Less useful.

4. **Stage-3 KD with family routing**: the Stage-2 teacher now has family
   routing. The Stage-3 student must inherit it. Verify that the Stage-3
   KD loss computes the difference between teacher-family-correct mu and
   student-family-correct mu, NOT a cross-family mixup.
5. **`Nozzle0` substring match**: the heuristic
   `0 if "Nozzle0" in name else 1` is fragile. Verify against all
   `experiment_name` values in the dataset (the canonical table CSV); if
   any non-N0 nozzle has "Nozzle0" as a substring (e.g. "Nozzle0X"),
   switch to a strict equality on the parsed nozzle ID column.

---

## 4. Sub-campaign 3B — Learnable / per-family λ_d2 concavity weight

### 4.1 Hypothesis

The fixed `d2_concave_weight = 5e-4` (after `d2_start_ms = 0.9`) is
appropriate for smooth-evolution nozzles (N1, N2, N4, N5) but mis-specified
for Nozzle 3's step-like wake-catch-up morphology. A learnable per-family
λ_d2 should drop towards 0 on N3-family data and stay near baseline on
others.

### 4.2 Three concrete variants

| variant | implementation |
|---|---|
| baseline | `λ_d2 = 5e-4` (current default) |
| no_d2_penalty | `λ_d2 = 0` globally (does N3 improve? does N2 degrade?) |
| per_family_learned | `λ_d2 = floor + softplus(W_family[family_id])` — learnable scalar per family, initialized to ≈ default |

If 3A is run first, "family" can mean the 3A 2-way split (factory-fresh vs
modified). N3 will be in family-1 (modified) along with N2/N4/N5, so the
2-family scheme will not isolate N3. To actually target N3, the family
scheme needs to be 5-or-6-way (per-nozzle). For v1, use 6-way per-nozzle
λ_d2; the family channel from 3A is replaced by `nozzle_id` ∈ {0…5} here.

### 4.3 Implementation

In `MLP/MLP_training/efc/models.py:PenetrationMLP.__init__` (around the
existing arg list):

```python
class PenetrationMLP(nn.Module):
    def __init__(self, ..., n_families_for_d2: int = 1, learnable_d2: bool = False):
        ...
        if learnable_d2:
            # init at log_lambda such that softplus(log_lambda) + floor ≈ 5e-4
            floor = 1e-5
            target_init = 5e-4 - floor
            inv_softplus_init = math.log(math.exp(target_init) - 1)
            self.log_lambda_d2 = nn.Parameter(
                torch.full((n_families_for_d2,), inv_softplus_init)
            )
            self.lambda_d2_floor = floor
        else:
            self.log_lambda_d2 = None
            self.lambda_d2_floor = 0.0
```

In the Stage-1 loss (locate via `grep -n d2_concave_weight` in
`MLP/MLP_training/`; likely in `efc/losses.py` or `engineered_feature_common.py`):

```python
def stage1_loss(outputs, targets, time_norm, family_ids, model, config):
    mu, log_var = outputs[:2]
    primary = ...
    d1_term = ...
    d2_penalty_per_sample = compute_d2_penalty(mu, time_norm, ...)   # shape [B, T]
    if getattr(model, "log_lambda_d2", None) is not None:
        weights = F.softplus(model.log_lambda_d2) + model.lambda_d2_floor   # [n_families]
        per_sample_weight = weights[family_ids]                              # [B]
        d2_term = (per_sample_weight.unsqueeze(-1) * d2_penalty_per_sample).mean()
    else:
        d2_term = config["d2_concave_weight"] * d2_penalty_per_sample.mean()
    return primary + d1_term + d2_term
```

Add CLI flags:
- `--learnable-d2` (bool) on Stage 1 trainer
- `--d2-concave-weight FLOAT` for the global override case (variant `no_d2_penalty` → `--d2-concave-weight 0`).

### 4.4 Data plumbing for per-nozzle `family_ids`

If using per-nozzle (6-way), add a separate `nozzle_id` channel (do NOT
overload the 3A `family_id` channel — that's binary).

```python
nozzle_to_id = {f"Nozzle{i}": i for i in range(6)}
df["nozzle_id"] = df["experiment_name"].astype(str).map(nozzle_to_id).fillna(-1).astype("int8")
```

The loss receives `family_ids = batch["nozzle_id"]`.

### 4.5 Run

```bash
# baseline
python train_stage1_mse.py --variant a_only --seed 42

# no_d2_penalty
python train_stage1_mse.py --variant a_only --seed 42 --d2-concave-weight 0

# per_family_learned (6-way per nozzle)
python train_stage1_mse.py --variant a_only --seed 42 --learnable-d2 --n-families-for-d2 6
# then Stage 2, Stage 3, full LONO for each
```

### 4.6 Output

`MLP/MLP_training/ablations/learnable_d2_<date>/`:
- `learnable_d2_lono.csv` — columns: `variant, fold_nozzle,
  rmse_overall_mm, rmse_t_lt_0p9ms_mm, rmse_t_ge_0p9ms_mm, cov_1sigma`
- `learned_lambda_d2_values.csv` — for `per_family_learned`: the final
  `λ_d2[nozzle=0…5]` per fold. **The headline is whether
  λ_d2[N3] → 0 while λ_d2[N1,N2,N4,N5] stay near 5e-4** —
  that would directly confirm the hypothesis.
- `verdict.md`

### 4.7 Budget 3B

- 3 in-domain runs × ~10 min = 30 min
- 3 × 5-fold LONO × ~10 min = 150 min
- **Total: ~3 h**

### 4.8 Pitfalls 3B

1. **N3 stays in family-1 if the 2-way (3A) scheme is reused for λ_d2**:
   per-family λ_d2 with only 2 families lumps N3 with N2/N4/N5 and cannot
   target it. Always use the **per-nozzle** scheme (6-way) for 3B, even if
   3A uses 2-way.
2. **λ_d2 → 0 globally**: if all learnable λ_d2 collapse to zero, the
   concavity prior disappears entirely and post-onset behaviour may
   degrade on smooth-evolution nozzles. The `floor = 1e-5` prevents true
   zero. Verify final values are not all at the floor.
3. **d2 indirectly stabilises log_var**: the d2 penalty acts as a smoothness
   regulariser on the mean head, which indirectly tightens the variance
   head's job. Removing d2 may inflate `cov_1σ` (over-coverage). Not a
   failure per se, but report it.
4. **Onset interaction**: `d2_start_ms = 0.9` means the prior is *off*
   during onset (t < 0.9 ms). So 3B primarily affects late-time behaviour.
   Pair with Tier-2C only after both have been run independently.
5. **LONO + per-nozzle λ_d2**: when fold = N3, no N3 training data exists,
   so `λ_d2[3]` never gets a gradient. It stays at init. This is fine —
   means the inference-time λ_d2[3] is the default, equivalent to baseline
   behaviour on that fold. The expected gain on N3-held-out is therefore
   **zero** by construction. To actually test the N3 hypothesis, also run
   an **in-distribution** evaluation where N3 is in training, and see if
   N3's per-trajectory RMSE on the in-distribution test split drops.

---

## 5. Total budget

| Campaign | Runs | Wall-clock |
|---|---|---|
| 3A A+B+C × 2 protocols × LONO folds | ~25 fold-runs | ~3-4 h |
| 3B 3 variants × 5-fold LONO + in-domain | ~18 | ~3 h |
| **TOTAL** | **~43 runs** | **~6-7 h** |

Larger than Tier 1 + Tier 2 combined. **Pre-requisite: Tier 1B σ_seed** —
without it, any 3A/3B delta can be dismissed as seed noise.

---

## 6. Files to read first

1. `MLP/MLP_training/efc/models.py` — `PenetrationMLP` definition; modify
   for both 3A (add `FamilyAwarePenetrationMLP`) and 3B (add `log_lambda_d2`
   parameter).
2. `MLP/MLP_training/engineered_feature_common.py` — `build_canonical_feature_table`
   and `build_variant_feature_table`; the `family_id` / `nozzle_id` channels
   are added here.
3. `MLP/MLP_training/efc/data_io.py:167-169` — `d2_concave_weight` default
   and the adjacent shape-penalty config dataclass.
4. `MLP/MLP_training/efc/losses.py` (or wherever Stage-1 loss is — search
   for `d2_concave_weight` in the repo) — modify d2 weighting for 3B.
5. `MLP/MLP_training/ood_lono/OOD_LONO_PLAN.md` — LONO infrastructure
   (orchestrator pattern, split function); 3A extends it with `--protocol`.
6. `MLP/MLP_training/trainers/stage1.py:53-71` — `parse_args` (add
   `--family-aware`, `--learnable-d2`, `--n-families-for-d2`,
   `--d2-concave-weight`).
7. `Thesis/latex/sections_en/05_results.tex` — §5.3 N0 fold discussion +
   Fig. 10 per-nozzle bar chart + Figs. 12a-d N3 evidence; informs verdict
   framing.

---

## 7. Paper paragraph hooks

If 3A wins on `lono_modified_only` (Option β) but not on `lono_5fold`:

> "A family-aware architecture with a shared backbone and per-family final
> heads improves the in-family LONO mean RMSE from 7.00 to X mm without
> degrading any individual modified fold; however, the held-out factory-
> fresh nozzle remains beyond the reach of the modified family's training
> distribution and is not addressed by architectural changes alone. We
> therefore propose this as the operational scope for the MLP surrogate:
> it is a strong predictor within a known design family, not a universal
> cross-family extrapolator."

If 3B shows `λ_d2[N3] → 0` while others stay near 5e-4:

> "Allowing the concavity penalty weight λ_d2 to vary per nozzle reveals
> that Nozzle 3 — which exhibits wake-driven step-like morphology — drives
> λ_d2[3] → 0 (floor 1e-5), recovering an unbiased fit on this nozzle
> (in-distribution test RMSE X mm, baseline Y mm). Other nozzles retain
> λ_d2 ≈ 5e-4, confirming that the smooth-concave prior is appropriate as
> a default but should not be globally enforced."

If both 3A and 3B fail (acceptable null result):

> "Two architectural interventions targeting documented worst-case folds
> (family-aware routing for Nozzle 0; learnable concavity weight for
> Nozzle 3) were tested. Neither produced gains beyond seed variance,
> suggesting that the corresponding failure modes are dominated by data
> limitations (single factory-fresh nozzle; under-sampled step-morphology
> regime) rather than inductive bias. Both are left as future work pending
> additional data acquisition."
