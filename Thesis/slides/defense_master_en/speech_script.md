# Defense speech script — full master deck (tightened)

*Spoken English, ~1 slide/minute. Bracketed cues are delivery notes, not to be read aloud.*
*Slide numbers match the 55-page `defense_master_en.pdf`. LLM-lab analogy kept light at Slides 1, 3, and 53 (opening frame + closing callback).*

---

### Slide 1 — Title
Good morning/afternoon, and thank you for being here. My thesis builds an AI-driven screening surrogate for spray–wall impingement — from raw Mie scattering videos all the way to a deployed app. One person, one machine, following the same engineering methodology used by leading AI labs. [Pause.]

---

### Slide 2 — The result, up front
The deployed screener offers two interchangeable solvers: a production neural network and a sparse Gaussian process. Same inputs, same physical outputs — different prior on how the function should behave.

Both land around **four millimetres** RMSE on the uncensored evaluation — roughly **58 and 54 percent** below the Hiroyasu–Arai and Naber–Siebers correlations. The neural network gives up about seven hundredths of a millimetre to the GP, but its uncertainty is far more conservative — 89 and 99 percent coverage. For a *screening* tool, that conservative bias is exactly what you want. [Pause.]

---

### Slide 3 — One-person frontier-lab pipeline
My contribution isn't a single architecture — it's reproducing, end to end, the engineering pipeline that frontier AI labs use.

Six stages. Data mining — mine is computer vision plus CUDA acceleration; theirs is web-scale data acquisition. Data cleaning — censoring-aware curation versus quality filtering. Target construction, training, evaluation, and deployment all map the same way. [Pause.] The vocabulary is literally shared — distillation, curriculum training, out-of-distribution benchmarks, calibration, deployment-time adapters. [Gesture to the right column.]

---

### Slide 4 — Why a surrogate: screen, don't replace
Why build this? Cost. A single high-fidelity reacting-spray simulation can burn forty-eight thousand CPU core-hours and twenty days of wall-clock — for just two milliseconds after injection. You cannot sweep a design space with that.

The surrogate is *not* meant to supplant CFD. It's a low-cost upstream layer that narrows the design space before you spend the expensive compute. [Beat.]

---

### Slide 5 — Stage 1: Data mining
Stage one: data mining. The raw material is terabytes of multi-hole Mie-scattering video. None of it is trainable as-is — I first turn it into plume-wise penetration-versus-time curves.

The engine is a streaming assembly line: GPU tensor work, a CPU thread pool, and disk I/O hidden behind computation. The whole archive processes in about twenty-eight minutes. [Point to diagram.]

---

### Slide 6 — The dataflow, end to end
This is that assembly line as one picture — the full dataflow from a raw Mie video to per-plume penetration curves. [Trace left to right.] Every stage I'm about to walk through is one node here, and the whole path from pixels to trainable trajectories is automated and resumable.

---

### Slide 7 — The imaged subject: injector geometry
These are multi-hole injectors in a top-view Mie configuration — all nine to eleven plumes visible in a single shot. Five repetitions per condition gives forty-five to fifty-five penetration series per operating point.

Key detail on the right: hole diameter varies only between 0.333 and 0.384 millimetres — a ratio of just 1.15. In Act 3, that narrow span is why diameter cannot be a standalone input feature. [Beat.]

---

### Slide 8 — CV step 0: manual calibration of the nozzle center
Every setup must be spatially calibrated. I exploit co-centered circular hardware features and fit circles by least squares — the equation is linear in its coefficients, so a standard solver recovers center and radius in closed form. This gives the nozzle center as measurement origin and the pixel-to-millimetre scale. Done once per setup, deliberately manual — a wrong origin contaminates everything downstream.

---

### Slide 9 — GUI: main interface
This calibration lives inside the screening GUI itself. [Let screenshot register.] The same application used later to *predict* wall impingement also handles *calibration*. Same app, both ends of the pipeline.

---

### Slide 10 — GUI: calibration interface
The calibration page: load the spray-image context, mark the circular references, and the tool fits center and scale. This is where human-in-the-loop calibration actually happens.

---

### Slide 11 — GUI: calibration result
The fitted center and scale overlaid on the image for visual check before anything downstream runs. [Beat.] With geometry pinned, let me go inside a single frame.

---

### Slide 12 — CV step 1: log-domain background subtraction
Raw Mie intensity is corrupted by glare, vignetting, and uneven illumination — all multiplicative. So I work in log space, estimate background as the per-pixel median over pre-injection frames, subtract in log domain, and gate to keep only pixels meaningfully above background. [Point to four panels.] Result: plume boundaries reflecting physics, not lighting.

---

### Slide 13 — CV step 2: Sobel high-pass
A Sobel high-pass on the clean foreground brings out the liquid-phase edge. The kernels are separable — cheap to run on every frame on the GPU. The output sharpens the moving spray front while smooth interior gradients fade.

---

### Slide 14 — CV step 3a: FFT-based angle estimation
I don't hard-code plume angles — I estimate them from the FFT. The time-summed angular intensity profile goes through a Fourier transform, and the phase at the harmonic equal to the hole count gives the global orientation. You can see the peak at ten in panel (d), with the recovered offset annotated. That single number gives every plume's guide angle.

---

### Slide 15 — CV step 3b: affine remap
Each plume is rotated and translated into a shared strip frame with the nozzle at the left edge. All downstream measurement is now rotation-agnostic and directly comparable across holes.

---

### Slide 16 — CV step 4: CDF tip definition
How do you pin the spray tip robustly? Not with the farthest-lit pixel — one stray speckle hijacks it. Instead, for every distance from the nozzle, add up the brightness in that column; that gives a 1D profile of where the spray is. Take its running total — the CDF — and the tip is the distance where that total first reaches 99.5 percent: essentially all the spray behind it. Because it's a quantile, a lone speckle adds almost nothing and can't pull the tip out. [Point to the pink penetration trace — the white dashed is hydraulic delay, the black dashed is nozzle closing.] On this time–distance map, every vertical column is one frame's profile, and the pink trace is that 99.5-percent crossing frame by frame — stacked over time, the penetration trajectory the rest of the thesis learns from.

---

### Slide 17 — Segmentation result
Thresholding the high-pass response after morphological repair gives a binary spray support; its outline is the detected boundary, traced on all plumes simultaneously. This segmentation underlies every downstream geometric measurement. [Pause.]

---

### Slide 18 — Segmentation in motion
The same boundary in motion — per-frame overlay, injection through decay, at fifteen fps. [Let it play.] Watch the front advance and the signal fade. Hold onto that fade — a few slides on, it becomes one of two reasons data must be censored.

---

### Slide 19 — Why parallelism
At archive scale, a naive per-frame loop doesn't finish. The pipeline layers three resources so none sits idle: one array abstraction for CPU or GPU; heavy reductions batched on the GPU; a CPU thread pool for per-plume metrics; and prefetch–write-behind I/O hiding the disk. [Beat.] Throughput is an architecture problem — the hard part is keeping every unit fed. With that, the data is mined; next, cleaning it.

---

### Slide 20 — Stage 2: Data cleaning
The raw trajectories are no longer videos, but they're not one clean table yet. Three derived populations, each answering a different question.

First: the raw cleaned per-frame set — about twenty-four thousand trajectories, roughly seven hundred thousand frame observations. Second: the uncensored p50 and q1 oracle set. Third: the reduced-order q1 parameters per trajectory. [Point down rows.] What you keep, discard, and label sets the ceiling for every model trained on it.

---

### Slide 21 — The data ontology
These three populations as one ontology. [Point along arrows.] The raw cleaned set at the root; oracle and per-trajectory parameters branch off. Each downstream model draws from exactly one population — they are not interchangeable.

---

### Slide 22 — Raw CV output: two ways a trajectory ends
Before any cleaning, this is what the vision pipeline hands over — penetration versus time, all ten plumes, the CDF front beside the two binary definitions. The data-cleaning stage works directly on these traces. It first shifts each one along the time axis so injection onset sits at a common origin. Then it truncates two signal-loss patterns you can read straight off the top panel — the unphysical negative-gradient retraction, where the front jumps backward, and the late flattening as the trace decays; both are signal loss, not real penetration, so both get cut. It also rejects the occasional optical-noise spike — rare now that the CV pipeline is mature. [Point to the top panel.] The bottom panel is the other mechanism — the field-of-view ceiling — which the next two slides take on.

---

### Slide 23 — Right-censoring I: field-of-view ceiling
First censoring mechanism: structural. Top-view Mie sees all plumes at once, but the camera window is about 180 millimetres — radius about 90 millimetres — leaving only 80 to 85 millimetres of usable penetration once the near-nozzle region is masked.

That's not enough for a real engine cylinder wall distance. When a trajectory flattens near this level, the spray front left the window — it didn't stop. Almost forty percent of trajectories hit this cap. [Point to saturation band.] Training on capped points teaches camera geometry, not spray physics.

---

### Slide 24 — Right-censoring II: late-time Mie signal decay
Second mechanism: the plume is inside the FOV, but the Mie signal fades — droplets break up, the plume dilutes, number density drops until signal is indistinguishable from background.

This is condition-dependent and plume-dependent. I detect it statistically: smoothed per-bin trajectory count below a fixed fraction of its peak. [Point to frames.] A measurement failure, not a physical zero — it must be censored.

---

### Slide 25 — Two censoring mechanisms, one cleaning rule
Side by side: FOV saturation on the left — quantiles converge at the eighty-one millimetre threshold. Density drop on the right — maximum penetration still below threshold, but trajectory count collapses just as sharply.

Unified rule: if the observation no longer faithfully measures the spray front, remove it before target fitting. If cleaning is wrong, the later uncertainty model becomes honest-looking but physically wrong.

---

### Slide 26 — 5 ms horizon plus q1 extrapolation
After censoring, raw observations are exhausted by about two milliseconds. For screening, I want a uniform five-millisecond horizon. The solution: a sigmoid-gated quarter-root reconstruction extending the lower-quantile trajectory.

This isn't cosmetic. Raw time bins are extremely imbalanced — about sixty-five thousand frames near 0.15 milliseconds collapse to fewer than two hundred by 1.55 milliseconds. After truncation and reconstruction, the imbalance drops by an order of magnitude. [Point to panels.]

---

### Slide 27 — q1 reconstruction, worked example (1)
The fit on a real trajectory — sigmoid-gated quarter-root through the observed points, extended to five milliseconds. [Brief.]

---

### Slide 28 — q1 reconstruction, worked example (2)
A second condition. [Brief.] Same form, different onset and amplitude.

---

### Slide 29 — q1 reconstruction, worked example (3)
A third. [Beat.] Across conditions the fit stays stable past the raw window — that stability licenses using it as a synthetic extrapolation target.

---

### Slide 30 — Stage 3: Target construction
Stage three: don't ask the neural network to rediscover obvious physical scale factors. Classical correlations tell us penetration depends on pressure, density, diameter, and time. I use those priors as a hard scaling divisor, then let the data decide which exponent regime fits.

Concretely: construct an amplitude feature from pressure, density, and diameter; train on scaled penetration rather than raw; validate with regression, ablation, and a collapse check.

Two uncertainties matter. *Aleatoric*: irreducible plume-to-plume scatter, carried by the heteroscedastic sigma head. *Epistemic*: model uncertainty over sparse regions, reducible with more data. A-scaling strips nuisance scale; folding diameter into A blocks nozzle-identity memorization. [Pause.]

---

### Slide 31 — Physical priors meet data-driven regression
Classical forms agree on reference exponents: pressure to one-quarter, density to minus one-quarter, diameter to one-half. But this experiment isn't an asymptotic textbook regime — per-window log-log regressions show the pressure exponent drifting from about 0.68 down to 0.45.

The diameter exponent blows up to 1.7–6.0 — not physically interpretable. The range is too narrow and confounded with injector family. [Point to plots.] Pressure scaling is real; diameter is not identifiable here.

---

### Slide 32 — Reading the exponents: time-binned log-log regression
Method: truncate each condition at censor onset — count CV drops from 0.52 to 0.052 — then fit an independent log-log regression in every tenth-of-a-millisecond bin on the half-million-point uncensored set.

Pressure holds at 0.47–0.52 — the Bernoulli orifice-flow value. Chamber density sits around minus 0.23 to 0.25, matching Hiroyasu–Arai. Diameter: 1.7–6.0, physically impossible. [Point to warning column.]

The pooled per-trajectory fit confirms: pressure near 0.5, density near minus 0.25. So I keep ΔP^0.5 and ρ_a^−0.25 from the data, fold d_n^0.5 in as a fixed prior.

---

### Slide 33 — Why diameter is folded in
Six discrete diameters, ratio 1.15, log-leverage much smaller than injection pressure, confounded with hole count and umbrella angle.

So d_n^0.5 stays inside the amplitude as a prior; the model focuses on the residual. Pressure is different: data supports a value near 0.5. Final amplitude: ΔP^0.5, ρ_a^−0.25, d_n^0.5. [Pause.] Physical prior where data is weak; data where data is strong.

---

### Slide 34 — Ablation: cost of the wrong feature set
If raw diameter is added as an independent input, the predicted mean develops five or six gradient sign reversals in a tiny interval — autodiff gradients spike to ten-to-the-fourth over about 0.05 millimetres.

The network is memorizing injector identity. In LONO testing, adding diameter increases MAE by four to six millimetres. Removing pressure residuals hurts too. Rule: fold diameter into A, keep pressure terms as residual inputs.

---

### Slide 35 — Target reparameterization and collapse check
The model learns S(t)/A. Collapse check: after scaling, do different pressure conditions land on the same curve?

With the classical exponent of 0.25, they don't collapse well. With 0.5, the median collapse ratio drops from 0.378 to 0.057. Five-seed RMSE drops from 10.72 to 8.92 millimetres. [Point to panels.] The scaled target removes nuisance units so the network spends capacity on remaining structure.

---

### Slide 36 — Stage 4: Training curriculum
Production model: a 512-512-128 MLP with dropout, five seeds. Heads output mean, log variance, and onset.

Training is staged. Stage 1: backbone on representative rows — one per condition — with MSE, log-variance prior, and shape penalties. Stage 2: all filtered rows, warm-started from Stage 1, Gaussian NLL with an early-time mean anchor. Stage 3: raw CDF series against the Stage-2 teacher, regime-weighted NLL plus distillation. [Point to architecture.] Pretrain, distill, fine-tune — staged because it's easier to debug than one monolithic objective.

---

### Slide 37 — Stage 1 validation: feature ablation
Stage 1 trains on representative rows in amplitude-scaled space. The LONO ablation validates the feature set. Winner: ΔP^0.5 amplitude plus residual pressure terms, mean MAE 7.46 millimetres across five folds.

The pattern: ΔP^0.5 beats ΔP^0.25; pressure residuals help; adding diameter hurts badly; legacy no-scale set near the bottom. This locks the production feature family — out-of-nozzle validation says it generalizes best.

---

### Slide 38 — Stage 2: early-time physics anchor
Two changes at Stage 2. Data: all filtered rows, warm-started from Stage 1. Loss: Gaussian NLL plus a soft early-time mean anchor over the data-sparse onset window.

Ablation: mean-only anchor gives 12.73 millimetres versus 14.54 for no anchor. On Nozzle2, anchor turns 19.18 into 7.76. Anchoring sigma too does not help — production uses mean anchor only. [Beat.] A little physics in the right place stabilizes OOD behaviour.

---

### Slide 39 — Stage 3: raw-CDF refinement
Stage 3 data: raw per-frame penetration plus Stage-2 teacher. Student warm-starts from teacher, trunk frozen, refining only heads.

Three regimes by coverage: above seventy percent — raw reliable; twenty to seventy — uncertain; below twenty — teacher dominates. Loss: regime-weighted raw NLL plus knowledge distillation. Outcome: Stage-3 variants separated by only 0.21 millimetres. The heavy lifting was done upstream; Stage 3 refines, it doesn't rescue.

---

### Slide 40 — Knowledge distillation
Production form: MSE on student mean plus MSE on log variance, sigma weight five. The student copies both the trajectory and the teacher's uncertainty shape, blended with raw targets where reliable.

Final numbers: uncensored CDF RMSE about 4.265 millimetres; one- and two-sigma coverage 0.887 and 0.989. [Point to table.]

---

### Slide 41 — Stage 5: Evaluation protocols
Four protocols. Full-clean CDF: headline accuracy. Uncensored CDF: censoring-robust check. P50 observed: condition-level robustness. Q1 extrapolated: lower-quantile continuation outside the raw window.

Don't collapse into one number — a model can look good on average but fail on censoring, OOD injectors, or calibration. [Beat.]

---

### Slide 42 — Headline accuracy
Both solvers cut the physics baselines roughly in half. Hiroyasu–Arai: 10.261 millimetres. Naber–Siebers: 9.286. MLP: 4.265. SVGP: 4.193.

Fifty-eight percent below Hiroyasu–Arai, fifty-four below Naber–Siebers. The two solvers land within 0.07 millimetres of each other. [Point to bars.] The GP is slightly sharper as a point predictor; the MLP brings advantages in calibration and flexibility.

---

### Slide 43 — Calibration: the MLP's real edge
The SVGP is slightly better on point RMSE but under-covers its own uncertainty. The MLP over-covers: 0.887 for one-sigma and 0.989 for two-sigma.

For screening, conservative bias is useful. A sharp but overconfident model passes risky designs too early. A conservative one sends more to CFD, but is less likely to miss a dangerous wall impact. [Beat.]

---

### Slide 44 — Qualitative fit and honest failure
Left: excellent fit, sub-millimetre RMSE. Middle: worst trajectory, Nozzle3 T6 plume4, over forty millimetres. Right: Nozzle3 is the residual-risk family.

Fit quality is excellent for most cases, but Nozzle3 is a real failure mode — reason to keep OOD checks in the deployment loop, not to discard the surrogate. [Beat.]

---

### Slide 45 — Anatomy of the failure
The Nozzle3 failure is *physical*. Cross-referencing Phantom high-speed images with plume-by-plume curves: strong inter-hole asymmetry — bottom-right plumes eject a low-momentum detached leading mass that decelerates, then the main jet overtakes it, producing a step-like rise after one millisecond.

My reduced-order target is monotonic and concave by construction — it cannot envelope a trajectory whose second derivative flips sign. Forcing a smooth prior onto a step distorts the target. The failure is identifiable, not mysterious.

---

### Slide 46 — Out-of-distribution: LONO, MLP vs SVGP
Leave-one-nozzle-out: five-fold mean, MLP 12.55 millimetres, SVGP 10.16. Excluding Nozzle0: 7.00 and 5.95. Nozzle0 alone dominates — roughly thirty-five for MLP, twenty-seven for SVGP.

SVGP is the stronger point predictor on four of five folds. MLP's value: conservative coverage, onset head, and speed enabling a large ablation study. Main conclusion: Nozzle0 is genuinely out of design family.

---

### Slide 47 — Stage 6: injector-family conditioning
Even after amplitude scaling, families don't fully collapse. Nozzle0 runs about 2.4–2.5 times higher at early time; the gap shrinks post end-of-injection but never disappears.

A single shared model leaves structured residual error. The fix: condition on injector family with a light adaptation layer — frozen shared trunk plus a per-family mean head. [Point to bullets.]

---

### Slide 48 — Family-aware head
The family-aware head fixes catastrophic folds without penalizing in-family ones.

Aggregate LONO RMSE: 11.44 down to 7.35 millimetres. Nozzle0: 30.58 to 11.57. Nozzle6: 36.91 to 7.61. In-family folds move less than one millimetre. [Point to bar chart.] The gain specifically repairs OOD-family failure.

---

### Slide 49 — Model lineage
The climb: Naber–Siebers at 9.286. Production MLP and SVGP at around 4.2. The family-head variant introduces the adaptation mechanism.

Follow-on variants push further: residual family heads, residual FiLM, residual multitask SVGP, QC-gated retrain. Best numbers around 3.99 and 3.92. Consistent mechanism: identity warm-start, frozen trunk, light family adapter. The residual multitask SVGP also repairs the extrapolated q1 tail.

---

### Slide 50 — Nozzle0 few-shot limit
The boundary condition. Delta head fine-tuned, trunk frozen, Nozzle0 held out. Zero-shot: about 33 millimetres. Two examples: 22. Ten: 20.7. Twenty: 19.2. All examples: roughly 15.7, NLL plateaus around 19.5.

Nozzle0 is genuinely out of design family. A head adapter floors around sixteen millimetres — closing that gap needs trunk capacity, not just a new head.

---

### Slide 51 — From checkpoint to product
Deployment ends with the screening GUI — the same tkinter wizard from the calibration slides. It collects piston-bowl geometry and spray conditions page by page, reproduces the full training feature pipeline at inference, runs the MLP, and overlays the prediction on crank-driven piston kinematics.

Outputs: peak piston-impact probability, cumulative exposure, wall-impact probability. Solver swappable between MLP and SVGP. Checkpoint to product.

---

### Slide 52 — The screening tool, live
Here it is running. [Let it play.] Mean penetration, uncertainty band, and onset overlaid on crank-driven piston motion. The numbers an engineer actually wants. [Pause — don't over-explain.]

This is the practical closure: raw Mie archive became trajectories; trajectories became targets; targets trained solvers; the solver returns here, in a visual interface. The pipeline runs end to end.

---

### Slide 53 — The whole pipeline, one more time
The left column is the thesis pipeline; the right is the frontier-lab analogy. [Let it register.]

To be precise: I'm not saying spray-wall impingement is language modelling. The scale and modality are completely different. The claim is that the engineering discipline transfers: data acquisition, filtering, target construction, staged training, robust evaluation, and deployment adaptation. The final solvers reach about 4.2 millimetres uncensored RMSE — roughly fifty-eight and fifty-four percent below the two physics baselines — with conservative calibration and a deployed GUI.

---

### Slide 54 — Contributions, limitations, future work
Six contributions: a CUDA video-to-observable workflow; a reduced-order q1 fit for right-censored targets; a three-stage uncertainty-aware MLP via knowledge distillation; an onset-CDF head with regime-aware censoring; a probabilistic screening GUI; and a large ablation-by-LONO-by-seed study.

Limitations: OOD families degrade badly, especially Nozzle0. The q1 continuation beyond FOV is unvalidated. GUI impingement geometry is a prototype. Half-cone angle is fixed. Points are correlated — clustered CIs still needed. SVGP is the stronger point predictor in-distribution.

Future work: validate hit probability against matching optical or CFD runs, learn a 2D spray envelope, quantify seed-to-seed reproducibility, add nozzle-conditioned transient priors for hard families.

---

### Slide 55 — Thank you
That is the thesis: one person, one machine, an end-to-end AI pipeline for spray-wall impingement screening.

Thank you for listening. I welcome your questions.
