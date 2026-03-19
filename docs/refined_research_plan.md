# Refined Research Plan: Data-Driven Prediction of Time-Evolving Spray Envelopes and Collision Events

## 1. Thesis Focus

This thesis will develop a data-driven surrogate workflow that predicts the time-evolving 2D liquid spray envelope from injector and operating conditions, quantifies prediction uncertainty, and detects likely collisions with piston and cylinder-wall geometry fast enough for engineering iteration. The plan is grounded in the existing codebase rather than a generic spray-ML concept.

The current repository already provides:

- image-based spray extraction from `.cine` videos, including plume alignment, penetration, cone angle, area, and optional boundary-point export
- experiment metadata integration from nozzle test matrices
- plume-wise penetration cleaning, hydraulic-delay alignment, and compact 4-parameter curve fitting
- Stage-1 and Stage-2 MLP baselines for penetration prediction, including heteroscedastic Gaussian NLL and manual GUI inference

The main research gap is therefore no longer "can spray data be extracted?" or "can penetration be predicted?" The real gap is how to move from the existing 1D penetration surrogate to a robust 2D envelope surrogate that can be mapped back into chamber geometry and used for uncertainty-aware collision detection.

## 2. Code-Grounded Baseline

The implemented baseline defines the practical starting point for the thesis:

- `main.py` processes calibrated multi-hole spray videos, reads operating-condition metadata from JSON test matrices, computes plume-wise penetration and binarized geometric metrics, and can export boundary points for each plume.
- `mie_multihole_pipeline.py` already converts full spray videos into plume-aligned segments, which is important because it naturally suggests learning in local plume coordinates instead of learning directly on full raw frames.
- `OSCC_postprocessing/binary_ops/feature_extraction.py` and related boundary utilities already represent each plume frame through top/bottom boundary point clouds and scalar geometric descriptors.
- `MLP/fit_raw_data.py` aligns hydraulic delay, cleans penetration traces, fits a two-regime sigmoid-transition penetration model, and flags bad fits. This creates compact, quality-controlled curve parameters and a strong penetration baseline.
- `MLP/median_penetration_MSE.ipynb` and `MLP/median_penetration_NLL.ipynb` already implement a curriculum from deterministic mean prediction to heteroscedastic penetration prediction using condition features and time.
- `Model_GUI.py` provides a usable manual inference front end for the current penetration surrogate.

This means the thesis should treat penetration prediction as the baseline subsystem and focus its novelty on 2D envelope prediction, uncertainty propagation, and geometry-aware collision analysis.

## 3. Finalized Research Problem

In modern internal combustion engine development, spray-wall and spray-piston interactions strongly affect mixture preparation, wall wetting, emissions, and efficiency. High-speed optical diagnostics provide detailed spray evolution, but they are expensive to run and do not directly support fast virtual design studies. Full CFD with moving boundaries remains too computationally expensive for rapid iteration.

The practical problem addressed in this thesis is therefore:

> How can the existing image-processing and penetration-ML workflow be extended into a fast, uncertainty-aware surrogate that predicts plume-resolved 2D spray envelopes over time and detects collision events with engine geometry under unseen operating conditions?

The problem should be framed as a surrogate-modeling problem in aligned plume coordinates, followed by geometric reintegration into chamber coordinates. This is a better fit to the existing repository than attempting end-to-end generation of raw spray images.

## 4. Final Research Questions

### Main RQ1

Can a data-driven model, conditioned on injector geometry, operating conditions, and time, accurately predict the plume-aligned 2D spray envelope together with uncertainty for unseen operating points?

### Main RQ2

Can the predicted 2D plume envelopes be mapped back into chamber geometry to detect wall or piston collision events with sufficient accuracy and runtime advantage over CFD-based workflows?

### Main RQ3

How much do uncertainty modeling and explicit handling of right-censored optical measurements improve the reliability of envelope and collision predictions?

### Supporting Questions

1. Which target representation is most effective for the existing pipeline: binary mask, radial-distance profile, or upper/lower boundary functions on a fixed axial grid?
2. Is a direct conditional model `f(condition, time, x)` more robust than an autoregressive frame-to-frame model for this dataset?
3. How should current penetration-side uncertainty modeling be extended from scalar tip penetration to spatial envelope uncertainty?
4. How should right-censored or optically saturated measurements be encoded during training?
5. How sensitive is collision detection to envelope uncertainty, and how should ambiguous cases be presented to users?
6. How well does the method generalize across nozzle families, chamber pressures, injection durations, and backpressure settings already represented in the test matrices?

## 5. Central Thesis Claim

The thesis should make one central claim:

> A plume-aligned, uncertainty-aware surrogate trained on boundary-derived targets can predict 2D spray envelopes and collision events much faster than CFD while retaining engineering usefulness across unseen operating conditions.

This is a stronger and more coherent thesis focus than claiming a full raw-image spray generator.

## 6. Recommended Target Representation

Based on the current codebase, the recommended primary representation is:

- upper and lower plume boundary functions on a fixed axial grid in aligned plume coordinates

That is, after plume rotation and centering, each frame is represented as:

- `y_upper(x, t)`
- `y_lower(x, t)`

for a fixed normalized axial grid `x in [0, 1]` or a physical axial grid in mm.

This representation is the best fit to the current pipeline because:

- boundary points already exist in the repository
- plume alignment is already done
- collision detection only needs a reconstructed envelope or mask, not photo-realistic spray texture
- the current penetration MLP already uses a coordinate-conditioned pattern over time; adding `x` is a natural extension
- this approach is cheaper and more data-efficient than training a full image generator

Binary masks should still be reconstructed from the predicted boundaries for visualization and collision testing, but masks should be treated as a derived product, not necessarily the primary regression target.

## 7. Methodology

### 7.1 Data Acquisition and Calibration

1. Use the existing GUI-based nozzle calibration workflow to define plume center, calibration radius, and annular processing region.
2. Use `main.py` and the current multi-hole pipeline to process all available `.cine` videos.
3. Enable plume-boundary export for all usable experiments so the dataset contains:
   - operating-condition metadata
   - plume index
   - frame/time index
   - penetration, cone angle, area, and other scalar features
   - top/bottom boundary point clouds per frame

### 7.2 Data Representation and Dataset Construction

Construct three nested datasets:

1. Scalar baseline dataset:
   - existing penetration targets from the current MLP workflow
2. Boundary-function dataset:
   - convert boundary point clouds into fixed-grid `y_upper(x,t)` and `y_lower(x,t)` targets
3. Mask/collision dataset:
   - rasterize predicted or measured boundaries into binary plume masks
   - transform plume masks back to global chamber coordinates

The boundary-function dataset should be the main thesis dataset.

### 7.3 Preprocessing and Quality Control

Reuse and extend the current cleaning logic:

- hydraulic-delay alignment from `MLP/fit_raw_data.py`
- removal of implausible penetration jumps
- fit-quality and outlier masking for poor traces
- per-condition metadata resolution using the JSON test matrices

Add a new preprocessing stage for envelope learning:

- resample each boundary to a common axial grid
- remove frames with insufficient boundary support
- attach censor flags when the spray tip or outer envelope exceeds optical visibility or measurement limits

### 7.4 Recommended Model Development Path

#### Stage A: Baseline Scalar Model

Use the current Stage-1 and Stage-2 penetration MLPs as baselines. Their role in the thesis is:

- to establish the predictive value of the current condition features
- to provide a scalar comparison benchmark for the later 2D models

#### Stage B: Deterministic 2D Boundary Surrogate

Train a coordinate-conditioned model with inputs such as:

- time
- axial coordinate `x`
- injection duration
- injection pressure
- chamber pressure
- control backpressure
- nozzle diameter
- plume count
- tilt angle / umbrella-angle-derived feature
- optionally plume index

Outputs:

- `y_upper`
- `y_lower`

This first model should use deterministic regression loss and smoothness constraints.

#### Stage C: Heteroscedastic 2D Boundary Surrogate

Extend the deterministic model so it outputs mean and log variance for each boundary value:

- `mu_upper, log_var_upper`
- `mu_lower, log_var_lower`

Use Gaussian NLL, following the logic already introduced in `MLP/median_penetration_NLL.ipynb`.

#### Stage D: Explicit Censor-Aware Training

The active codebase already supports heteroscedastic NLL, but it does not yet implement censor-aware loss in the current pipeline. This should therefore be a thesis contribution.

Recommended approach:

- use standard Gaussian NLL for uncensored targets
- use a right-censored likelihood or survival-style term for censored boundary/tip observations
- apply censor-aware loss primarily where visibility truncation affects the downstream geometry, especially near the tip region

#### Stage E: Optional Latent or Mask Model

Only if the boundary-function model proves insufficient, add a second-generation model:

- latent autoencoder for boundary fields or masks, with a condition-to-latent predictor

This should be explicitly secondary, not the starting point.

### 7.5 Physics and Shape Constraints

Retain the spirit of the current penetration model constraints and extend them spatially:

- monotonic envelope growth in the axial direction where physically justified
- nonnegative width constraints
- temporal smoothness
- limited curvature or slope regularization for upper/lower boundaries
- optional consistency constraints between predicted boundary and predicted penetration tip

The thesis should present these as soft physical priors, not full physics-informed PDE enforcement.

### 7.6 Collision Detection Module

The collision module should operate after plume prediction, not inside the network.

Recommended workflow:

1. Predict plume-aligned boundaries for each plume and time frame.
2. Reconstruct local binary masks.
3. Map each plume back into global coordinates using the calibrated center and plume angles.
4. Generate piston-head and cylinder-wall geometry masks from parametric geometry definitions and piston position over time.
5. Compute:
   - first collision time
   - collision area
   - collision location
   - ambiguous collision zone under uncertainty

Uncertainty should be propagated in one of two ways:

- deterministic bands from `mean +/- k sigma`
- Monte Carlo boundary samples leading to collision probability maps

The output shown in the GUI should distinguish:

- no collision
- likely collision
- possible collision under uncertainty

### 7.7 GUI Integration

The final tool should extend the current penetration-inference concept into an engineering viewer with:

- manual input of operating conditions
- optional nozzle selection
- time animation of the predicted plume envelope
- piston/wall overlay
- collision indicators
- uncertainty display
- export of plots, masks, and summary tables

The thesis should describe GUI integration as the deployment layer, not as the methodological core.

## 8. Validation Plan

### 8.1 Scalar Metrics

Use the existing extracted scalar metrics as baseline checks:

- penetration RMSE and relative error
- cone angle error
- area/volume proxy error

### 8.2 Spatial Envelope Metrics

Evaluate predicted boundaries and masks with:

- boundary MAE or RMSE over the axial grid
- Hausdorff distance
- symmetric average surface distance
- mask IoU / Jaccard
- Dice score

### 8.3 Collision Metrics

Evaluate geometry interaction with:

- collision/no-collision classification accuracy
- first-collision-time error
- collision-region IoU
- precision/recall for possible-collision warning zones

### 8.4 Uncertainty Quality

Evaluate uncertainty using:

- prediction-interval coverage
- calibration curves
- negative log-likelihood
- collision-probability calibration

### 8.5 Runtime

Benchmark:

- extraction preprocessing time
- per-case surrogate prediction time
- collision-check time
- total runtime versus a CFD reference workflow

## 9. Hypotheses

The thesis can be organized around the following hypotheses:

- H1: A plume-aligned boundary surrogate predicts 2D spray envelopes more accurately than scalar-only extrapolation from penetration and cone angle.
- H2: Heteroscedastic uncertainty modeling improves calibration of envelope and collision predictions relative to deterministic regression.
- H3: Explicit censor-aware loss improves prediction quality under optically truncated conditions.
- H4: Envelope-based collision detection provides sufficiently accurate early design feedback at a fraction of CFD runtime.

## 10. Expected Contributions

The thesis contributions should be stated as:

1. A complete pipeline from calibrated spray imaging to plume-aligned boundary-learning targets.
2. A new surrogate model for time-evolving 2D spray envelope prediction under operating-condition changes.
3. An uncertainty-aware extension of the current penetration MLP toward spatial envelope prediction.
4. A censor-aware training strategy for optically limited spray measurements.
5. A geometry-aware collision-detection module for piston and wall interaction.
6. A practical GUI-oriented decision-support tool for engineering studies.

## 11. Scope Boundaries

To keep the thesis focused, it should explicitly exclude:

- full CFD replacement for internal droplet-scale physics
- direct raw-image generation as the primary modeling goal
- fully 3D volumetric reconstruction as the main thesis target
- wall-film thickness prediction beyond collision/contact detection

The thesis target is a 2D envelope surrogate with collision awareness, not a universal spray simulator.

## 12. Risks and Mitigation

### Risk 1: Boundary targets are too noisy

Mitigation:

- use resampled boundary functions
- smooth only after preserving penetration tip
- train first on deterministic mean targets before uncertainty modeling

### Risk 2: Dataset size is insufficient for full mask prediction

Mitigation:

- prioritize boundary-function learning over full image generation
- exploit plume-wise alignment and per-plume decomposition to enlarge usable sample count

### Risk 3: Censoring labels are difficult to define

Mitigation:

- start with simple visibility-threshold labels at the tip region
- perform ablations with and without censor-aware loss

### Risk 4: Collision truth is limited

Mitigation:

- use a combination of experimental overlays, manually labeled cases, and geometry-consistency checks
- define conservative "possible collision" categories when hard truth is uncertain

## 13. Recommended 12-Month Work Plan

### Phase 1: Months 1-2

- finalize literature review
- freeze operating-condition metadata schema
- export plume boundary datasets from the current processing pipeline

### Phase 2: Months 3-4

- build boundary resampling and dataset-generation code
- establish scalar and deterministic boundary baselines

### Phase 3: Months 5-6

- train and evaluate heteroscedastic 2D boundary models
- perform ablations on feature design and target representation

### Phase 4: Months 7-8

- implement censor-aware loss and uncertainty calibration studies
- compare deterministic, NLL, and censor-aware variants

### Phase 5: Months 9-10

- implement chamber geometry mapping and collision detection
- validate collision predictions and runtime

### Phase 6: Months 11-12

- integrate the GUI demonstrator
- finalize thesis writing, figures, and defense material

## 14. Final Thesis Positioning

The strongest final positioning is:

This thesis is not about replacing CFD with an unconstrained black-box image generator. It is about converting an already functional spray-image-processing and penetration-ML workflow into an uncertainty-aware, plume-resolved 2D envelope surrogate that can support real engineering questions: where the spray goes, when it reaches a boundary, and how confidently that interaction can be predicted.
