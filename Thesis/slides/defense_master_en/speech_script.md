# Defense speech script — full master deck (aligned to slides)

*Spoken English, target total presentation time under 30 minutes (~30-40 seconds per slide average; flowcharts have detailed explanations, while plain visualizations are summarized in 1-2 sentences).*

---

### Slide 1 — Title Page
Good afternoon. Thank you for being here today. My thesis presents "An AI-Driven Screening Surrogate for Spray-Wall Impingement." This project demonstrates how a single researcher, using a single workstation, can borrow a small-scale version of the end-to-end engineering workflow used in frontier AI labs to build an upstream screening layer for a classical mechanical design problem. [Pause.]

---

### Slide 2 — Motivation: design diesel pilot injection without a CFD run per candidate
The core engine design challenge is tuning the small diesel pilot injection in dual-fuel marine engines to ensure reliable ignition while avoiding liquid fuel impingement on the piston or cylinder bore. Exploring this design space is severely restricted by the cost wall of high-fidelity reacting-spray simulations, which can take 48,000 CPU hours and up to 20 days of wall-clock time for only 2 milliseconds of spray development. This thesis introduces a fast, heteroscedastic surrogate that predicts penetration mean and variance in milliseconds, transforming wall impingement risk assessment from a costly simulation task into a rapid, rankable scoring pipeline. [Pause.]

---

### Slide 3 — The imaged subject: multi-hole injector geometry & design space
*Plain visualization.* This slide shows the multi-hole nozzle geometry where hole diameter varies only from 0.333 to 0.384 mm, and a sample image sequence of the multi-plume spray. Because the diameter range is so narrow, it cannot be learned as an independent input feature. [Pause.]

---

### Slide 4 — plain: surrogate_modelling_data_flow.png
*Flowchart.* Let's walk through the end-to-end data flow of our surrogate modeling, structured into four horizontal categories.
At the top, we have the **Raw Inputs** consisting of raw high-speed Mie scattering videos, calibration configurations, and experimental testing metadata.
Next is **Image Processing**, where a CUDA-accelerated image processing pipeline registers the raw videos into plume-wise canonical strips to extract 1D boundary points and 1D penetration trajectories using both a binarized-boundary, or BW, method and a robust CDF method.
Moving to **Data Augmentation & Deep Learning**, we perform parameter-free curve-fitting on the CDF-based trajectories to generate clean synthetic data. We apply median filtering to this synthetic data to warm-start our Stage 1 MSE training, which then initializes Stage 2 Gaussian NLL training. The final Stage 3 trained model is a student model supervised via knowledge distillation from the Stage 2 teacher and supervised fine-tuning on the raw CDF data.
Finally, in **Model Deployment**, the Stage 3 model is integrated into our Impingement Screening desktop application along with the target piston design and physical parameters to output collision probabilities and real-time animations of the spray-wall interactions. [Pause.]

---

### Slide 5 — The product: a CFD-free spray-impingement screener
*Plain visualization.* This slide showcases the live impingement screening application in action, depicting the spray's Gaussian envelope against the moving piston bowl and cylinder bore. The underlying MLP or SVGP surrogate acts as the solver, coupling prediction to piston kinematics while ignoring the in-cylinder swirl flow fields. [Pause.]

---

### Slide 6 — What the screener computes
To understand what the screener computes, we read the three-step pipeline bottom-up:
First, at the **bottom**, the deep learning surrogate predicts the plume-wise penetration mean and uncertainty as a function of time.
Second, at the **top**, we represent the spray boundary as a 2D anisotropic Gaussian and calculate the collision integral over the moving piston crown or bore.
Third, in the **middle**, we integrate these probabilities over time to produce a single screening scalar that allows designers to rank candidate injectors. [Pause.]

---

### Slide 7 — Coupled to piston motion, decoupled from in-cylinder swirl
*Plain visualization.* This slide compares three engine setups to demonstrate how different bowl shapes and piston kinematics shift the cumulative collision scalar. The point is narrower than full engine validation: the kinematic coupling to the moving piston is represented, while the induced swirl flow is deliberately outside the model. [Pause.]

---

### Slide 8 — The thesis in one picture: a one-person frontier-lab pipeline
*Flowchart/Table.* This table overlays the stages of my thesis with the workflow used by frontier AI labs to build large language models.
First, **Data mining** maps web-scale data acquisition to our parallel CUDA-based image processing.
Second, **Data cleaning** matches web filtering to our censoring-aware curation.
Third, **Target construction** matches label engineering to our $A$-scaled feature engineering and $q_1$ targets.
Fourth, **Training** aligns the pretrain-distill-finetune paradigm with our 3-stage MLP curriculum.
Fifth, **Evaluation** mirrors LLM benchmarks to our leave-one-nozzle-out validation and calibration metrics.
Lastly, **Deployment** aligns serving with parameter-efficient family adapters and the desktop GUI. Let's look at each of these stages. [Pause.]

---

### Slide 9 — Stage 1 --- Batch Image Processing as Data mining: CUDA/CuPy >> per-frame MATLAB loop (CPU)
*Flowchart.* This processing lane diagram shows how our batch image processing pipeline achieves high throughput by keeping GPU, CPU, and disk I/O busy simultaneously.
**Lane 1** uses a background thread to prefetch and read the next video file $N+1$ from the disk.
**Lane 2** runs the GPU compute thread, performing Host-to-Device transfer, fp16 normalization, background subtraction, and Sobel filtering on file $N$.
**Lane 3** utilizes a CPU thread pool ($\lfloor n_{\mathrm{cores}}/2\rfloor$ threads) to compute additional black-and-white geometric metrics.
**Lane 4** performs asynchronous file writing (CSV, metadata, NPZ, AVI) in the background.
A checkpoint JSON file is updated with `fsync` after every completed video, making the pipeline fully resumable and enabling us to process the entire multi-terabyte database in just 28 minutes. [Pause.]

---

### Slide 10 — CV step 0 --- Manual calibration of Nozzle center
*Plain visualization.* This slide displays the manual spatial calibration process, where we exploit the co-centered circular features of the nozzle hardware. We solve a least-squares circle fitting problem in closed form to recover the center coordinates and pixel-to-millimeter ratio. [Pause.]

---

### Slide 11 — plain: GUI_main_interface.png
*Plain visualization.* This is a screenshot of the main page of our desktop GUI, displaying the integrated control dashboard that houses our calibration and screening tools. [Pause.]

---

### Slide 12 — plain: GUI_calibration_interface.png
*Plain visualization.* This screenshot shows the calibration interface, where the user can overlay circular references on the nozzle image to calibrate the coordinate system. [Pause.]

---

### Slide 13 — plain: GUI_calibration_result.png
*Plain visualization.* This screen displays the completed calibration, showing the fitted center and scale overlaid on the raw hardware before processing. [Pause.]

---

### Slide 14 — plain: Multi-Hole Top-View Mie Scattering Video Processing Pipeline.png
*Flowchart.* This flowchart outlines the full video processing pipeline, divided into five main phases.
1. **Reading**: The raw `.cine` video, calibration configs, and metadata are loaded into RAM, and a ring mask is applied to hide the nozzle structure and chamber walls.
2. **Pre-processing**: The video is log-transformed, has its pre-injection background subtracted, is converted back to linear scale, gated, and scaled by percentiles to yield a clean foreground and high-pass features.
3. **Post-processing**: The foreground is transformed to polar coordinates via angular binning, binarized to get occupied angular bins, and processed with a Fourier transform (FFT) to determine the plume rotation angle. An inverse affine mapping rotates and translates each plume into a canonical horizontal strip of shape `(P, F, H, W)`.
4. **Feature Extraction**: The canonical strips are summed along the column dimensions to build time-distance heatmaps, which are used to extract CDF penetration trajectories. In parallel, a triangular binarization and blob-tracking step extracts the BW boundary. Both are scaled from pixels to millimeters with an umbrella angle correction.
5. **Writing**: The extracted CDF and BW penetration curves, cone angles, areas, volumes, and metadata are saved to disk. [Pause.]

---

### Slide 15 — CV step 1 --- Robust log-domain background subtraction
*Plain visualization.* This slide shows the log-domain background subtraction panels. By subtracting the pre-injection background in log space, we remove multiplicative reflections and vignetting, making the plume boundaries physically meaningful. [Pause.]

---

### Slide 16 — CV step 2 --- Sobel high-pass for boundary contrast
*Plain visualization.* This panel shows the output of our separable Sobel filter running on the GPU. The high-pass filter highlights the moving spray front while suppressing smooth intensity variations inside the plume. [Pause.]

---

### Slide 17 — CV step 3a --- FFT-based plume-angle estimation
*Plain visualization.* This slide displays the angular intensity profile of the spray. The phase of the Fourier transform at the harmonic equal to the plume count gives the global rotational offset of the injector. [Pause.]

---

### Slide 18 — CV step 3b --- Affine remap into a canonical strip
*Plain visualization.* This visualization shows the output of the affine remap. All plumes are rotated and translated into a shared horizontal strip, making downstream geometric measurements rotation-agnostic. [Pause.]

---

### Slide 19 — CV step 4 --- CDF tip definition (robust penetration front)
*Plain visualization.* This slide shows the CDF penetration extraction process. By treating the column intensity as a probability distribution and choosing the 99.5% quantile crossing, the resulting trajectory is highly robust to optical noise and stray droplets. [Pause.]

---

### Slide 20 — CV step 4b --- Segmentation result: the detected spray boundary
*Plain visualization.* This slide shows the segmented boundaries overlaid in red. This outline defines the binary spray support, which is used for downstream geometric calculations. [Pause.]

---

### Slide 21 — CV step 4c --- The detected boundary, in motion
*Plain visualization.* This animation shows the segmented boundary overlays running in real-time across a single injection-decay cycle, illustrating the transition from initial injection to late-time signal fading. [Pause.]

---

### Slide 22 — Raw CV output: two ways a trajectory ends
*Plain visualization.* This slide plots the raw signals to illustrate the two ways a trajectory terminates: late-time signal decay (where binary metrics collapse) and field-of-view censoring (where curves flatten at a common ceiling). [Pause.]

---

### Slide 23 — Stage 2 --- Data cleaning: from raw records to curated populations
*Plain visualization.* This slide lists the three distinct datasets derived from the same raw archive: the raw frame observations (Dataset 1), the uncensored points (Dataset 2), and the per-trajectory fitted parameters (Dataset 3). Each has a specific, non-interchangeable role in downstream training and evaluation. [Pause.]

---

### Slide 24 — Where the training data comes from: three populations, one archive
*Flowchart.* This lineage flowchart shows how our datasets feed into different training and evaluation paths.
Starting with **Dataset 1** (raw cleaned CDF observations), we truncate the trajectories at the onset of density drop or FOV saturation to build **Dataset 2** (uncensored points). Taking the per-bin median of Dataset 2 allows us to reconstruct a synthetic reference curve, the **P50/q1 oracle**, reserved exclusively for evaluation.
Separately, and not through Dataset 2, we fit the reduced-order $q_1$ curve to each individual trajectory in Dataset 1, yielding **Dataset 3** ($q_1$ parameters). Dataset 3 is what the training stages consume; the similarly named P50/q1 oracle is only a reference curve. [Pause.]

---

### Slide 25 — Dataset #1 vs. #3: population scatter & three operating-condition subsets
*Plain visualization.* This slide displays the population scatter alongside three operating conditions, showing how the fitted $q_1$ curves smooth the observed raw points where data exists and then provide a uniform 5ms continuation. That continuation is useful for training, but it remains a reconstruction and extrapolation protocol. [Pause.]

---

### Slide 26 — plain: injector_scatter_all_nozzles.png
*Plain visualization.* This is a full-size scatter plot of the entire database scaled by the physical amplitude $A$, demonstrating the tight grouping achieved before any machine learning is applied. [Pause.]

---

### Slide 27 — plain: q1_fit_example_1.png
*Plain visualization.* This plot shows a clean low-penetration example where the fitted $q_1$ curves track the raw observations over the available window, then continue as dashed reconstructions toward a uniform horizon. [Pause.]

---

### Slide 28 — plain: q1_fit_example_2.png
*Plain visualization.* In this deeper, higher-pressure condition, the fitted curve smoothly follows the trajectory until it is truncated. [Pause.]

---

### Slide 29 — plain: q1_fit_example_3.png
*Plain visualization.* This is an extreme example where the raw signal is exhausted before about 2 milliseconds; here, the $q_1$ fit supplies a model-based tail out to the required 5ms horizon. [Pause.]

---

### Slide 30 — plain: Data_Ontology.png
*Flowchart.* This diagram maps out the data ontology for our surrogate.
We divide the penetration data into two complementary perspectives.
On the left, we look at the data **As Points**, which exposes **Aleatoric Uncertainty** (the physical variation between plumes). This perspective forms a statistical distribution that we use for training and evaluating heteroscedastic models.
On the right, we treat the data **As Trajectories**, which exposes **Epistemic Uncertainty** (model uncertainty). These trajectories can be fitted with curves (carrying bias, smoothing, and extrapolation) or studied with time-series models.
At the bottom, the data is understood as being **Generated by a function**. This function connects to empirical physics (like Hiroyasu-Arai and Naber-Siebers, OLS regression, and physics-informed constraints) which ultimately feed into our MLP and SVGP surrogate models. [Pause.]

---

### Slide 31 — Right-Censoring I: a structural field-of-view ceiling
*Plain visualization.* This slide illustrates the FOV ceiling. Since the chamber window is limited to a 90mm radius, nearly 40% of the trajectories are artificially cut off, representing a diagnostic limitation rather than a physical stop. [Pause.]

---

### Slide 32 — Right-Censoring II: late-time Mie signal decay
*Plain visualization.* Here, we see how the Mie signal decays over time. As fuel droplets dilute and break up, the signal falls below the detection threshold, which we identify when the trajectory count collapses. [Pause.]

---

### Slide 33 — Two censoring mechanisms, one cleaning rule
*Plain visualization.* This slide compares FOV saturation on the left with density drop on the right. Both are distinct measurement failures that are detected and cleaned to prevent biasing the neural network. [Pause.]

---

### Slide 34 — Designing for redundancy: a 5 ms horizon + q1 extrapolation
*Plain visualization.* This slide outlines the mathematical form of the $q_1$ curve. It balances the high temporal imbalance in raw data but introduces parameter redundancy, meaning it must be treated as a trajectory reconstruction tool rather than a physical law. [Pause.]

---

### Slide 35 — Stage 3 --- Target construction: one engineered feature, three decisions
*Plain visualization.* This slide details our target construction decisions. We scale the penetration target by the physical scaling prior $A$ with a data-driven pressure exponent of 0.5, while folding diameter in as a fixed prior and keeping pressures as residual inputs to prevent overfitting. [Pause.]

---

### Slide 36 — Does the feature do its job? Family collapse under A-scaling
*Plain visualization.* These plots compare the family collapse under scaling. Using the data-driven pressure exponent of 0.5 collapses the operating conditions much tighter than the textbook value of 0.25, justifying our choice. [Pause.]

---

### Slide 37 — Stage 4 --- Training: a three-stage curriculum
*Flowchart.* This diagram shows the architecture of our MLP surrogate.
- On the **left**, the model takes a 7-feature input vector containing normalized time, nozzle tilt, plume counts, and residual pressures.
- These inputs feed into the **Trunk** (top right) consisting of three hidden layers of sizes 512, 512, and 128, using LayerNorm and SiLU activations with a 30% dropout rate.
- This trunk outputs to the **A-scaled heads** (middle right) which predict normalized mean $\hat\mu$, log-variance $\log\hat\sigma^2$, and an optional auxiliary onset.
- In parallel, the physical **Condition prior** $A = \Delta P^{0.5}\rho_a^{-0.25}d_n^{0.5}$ is computed.
- Finally, the normalized outputs are multiplied by $A$ to produce the physical outputs $\mu_S$ and $\sigma_S$ in millimeters. [Pause.]

---

### Slide 38 — Three training stages, compressed: data, loss, purpose
*Plain visualization.* This table summarizes our three-stage curriculum. We start with representative rows in Stage 1 to fit the median path, move to NLL training on all filtered data in Stage 2, and use knowledge distillation in Stage 3 to train the final student model on raw CDF data. [Pause.]

---

### Slide 39 — Where the training data goes: feeding the three training stages
*Flowchart.* This flowchart shows how our datasets feed into the three training stages.
- **Dataset 3** ($q_1$ parameters) is filtered using QC checks to feed Stage 2 Gaussian NLL training, while a single representative trajectory per condition is extracted to train Stage 1.
- **Dataset 1** (raw observations) feeds directly into Stage 3's raw NLL loss.
- Crucially, the trained Stage 2 model acts as a teacher, passing its predicted mean and variance via **knowledge distillation** (the red dashed arrow) to guide the Stage 3 student where raw data is missing. [Pause.]

---

### Slide 40 — Knowledge distillation: the same algorithm the labs use
*Plain visualization.* This slide outlines how the student model uses the teacher's soft labels to extrapolate in the late-time censored tail. On the uncensored CDF protocol, the production model reaches about 4.26mm RMSE, with conservative coverage: 0.887 at one sigma and 0.989 at two sigma. [Pause.]

---

### Slide 41 — SVGP alternative: same target, different inductive bias
*Plain visualization.* This slide introduces our alternative SVGP model. By utilizing a Matérn-5/2 kernel with 256 inducing points, it acts as a smooth local averager, achieving slightly better point RMSE than the MLP while under-covering uncertainty. [Pause.]

---

### Slide 42 — Stage 5 --- Evaluation: held-out, out-of-distribution, and calibrated
*Plain visualization.* This table lists the four evaluation protocols we use to stress-test our models under different scenarios: overall accuracy, uncensored check, condition-level mean head check, and out-of-window extrapolation. [Pause.]

---

### Slide 43 — Headline: both solvers crush the physics baselines
*Plain visualization.* This comparison shows that both the MLP and SVGP surrogates cut the prediction error in half compared to classical Hiroyasu-Arai and Naber-Siebers correlations, bringing RMSE down to around 4.2mm. [Pause.]

---

### Slide 44 — Calibration: the MLP's real edge
*Plain visualization.* These calibration curves reveal that the MLP over-covers its uncertainty, reaching 89% for $1\sigma$ against a nominal 68%. That is not perfect calibration; it is a conservative bias, which is preferable for an upstream screening tool because it is less likely to understate risk than the sharper but under-covering SVGP. [Pause.]

---

### Slide 45 — Qualitative fit, including an honest failure
*Plain visualization.* Here we plot typical predictions alongside a worst-case scenario. While the fit quality is excellent overall, Nozzle 3 exhibits a large error, indicating a systematic failure mode. [Pause.]

---

### Slide 46 — Anatomy of the failure: the prior cannot envelope a step
*Plain visualization.* This slide diagnoses the Nozzle 3 failure. High-speed images show asymmetric spray development, causing a step-like trajectory that our monotonic, concave $q_1$ prior cannot physically fit, explaining the mismatch. [Pause.]

---

### Slide 47 — Out-of-distribution: leave-one-nozzle-out, MLP vs SVGP
*Plain visualization.* This table shows the leave-one-nozzle-out cross-validation results. While SVGP has better out-of-distribution point accuracy, Nozzle 0 remains a hard out-of-design family for both models, indicating a need for family adaptation. [Pause.]

---

### Slide 48 — Stage 6 --- Deployment: conditioning on the injector family
*Plain visualization.* These curves show that even after scaling, injector families differ systematically. Nozzle 0 runs up to 2.5 times higher than the population mean, showing why a single global model leaves accuracy on the table. [Pause.]

---

### Slide 49 — The adapter family: a shared frozen trunk + a light per-family residual
*Flowchart.* Let's discuss and compare the three post-training family-conditioning architectures we evaluated.
- **Architecture A** is the **Family Head MLP**. It uses the family ID to select a family-specific mean decoder $\mu_f(h)$ connected to the shared trunk. However, an unseen family requires a fallback policy or a completely new head.
- **Architecture B** is the **Residual Family Head**. We freeze the shared trunk and add a family residual module $\Delta_f(h)$ in parallel with the shared mean. This provides a zero-residual fallback policy ($\Delta = 0$) for unseen families.
- **Architecture C** is the **Residual-FiLM** method. We modulate the representation using a last-block family affine transformation controlled by the family ID, then pass it to the shared log-variance head and add a family residual head.
All three architectures freeze the trunk and only train light family modules, adopting a parameter-efficient adapter paradigm. [Pause.]

---

### Slide 50 — Systematic family difference => a light adapter fixes the catastrophic folds
*Plain visualization.* This comparison shows that a light family adapter substantially reduces the catastrophic errors on out-of-distribution families like Nozzle 0 and Nozzle 6, bringing the overall LONO RMSE down from 11.4mm to 7.3mm. It improves the failure mode, but the next slide shows that Nozzle 0 is not fully closed to in-family performance. [Pause.]

---

### Slide 51 — The boundary, quantified: Nozzle0 few-shot limit
*Plain visualization.* These adaptation curves show that even with few-shot delta-head tuning, Nozzle 0's error floors at 16mm. This is because Nozzle 0's physical design is radically different from the rest of the database, highlighting the boundaries of adapter-only capacity. [Pause.]

---

### Slide 52 — From checkpoint to product: the screener GUI
*Plain visualization.* This slide describes the desktop impingement wizard. It implements the full training feature pipeline at inference, including z-scoring, $A$-scaling, and condition canonicalization, then allows the engineer to swap between the MLP and SVGP solvers. [Pause.]

---

### Slide 53 — The screener, live: wall-impingement prediction
*Plain visualization.* This animation shows the live prototype screener calculating the spray Gaussian envelope and wall-impact probability over the engine's crank cycle, closing the loop from raw data to an upstream design tool. [Pause.]

---

### Slide 54 — The whole pipeline, one more time
*Plain visualization.* This final summary table reinforces the analogy carefully: this is not language modeling, and the scale is completely different. What transfers is the engineering discipline: data mining, censoring-aware filtering, target construction, staged training, OOD evaluation, conservative uncertainty, adapters, and deployment as a prototype screening GUI. [Pause.]

---

### Slide 55 — Contributions, limitations, future work
*Plain visualization.* This slide lists the contributions, but it is equally important as the boundary statement. The surrogate has strong internal CDF metrics, yet OOD families still degrade, the $q_1$ continuation beyond the field of view is unvalidated, and the GUI impingement geometry remains a prototype until matched optical or CFD validation is done. [Pause.]

---

### Slide 56 — Thank you --- Questions welcome
*Plain visualization.* In conclusion, this work demonstrates how modern AI engineering methodology can turn a raw Mie-video archive into a fast, uncertainty-aware upstream screening prototype for spray-wall impingement. The next step is not to replace CFD or experiments, but to validate the screening probabilities against them. Thank you very much for your attention, and I am happy to take any questions. [Pause.]
