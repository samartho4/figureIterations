
# Figure-by-Figure Analysis (Paper-Aware, Literature-Grounded)

This document explains each figure using your paper as primary context and cross-references the restored project artifacts that generated the results. Where appropriate, we cite relevant literature to justify methodology choices and interpretations.

Data and assets used:
- Data: `data/{training_roadmap.csv, validation_roadmap.csv, test_roadmap.csv}` (restored)
- Metrics: `results/comprehensive_metrics.csv` (restored)
- BNODE calibration: `results/simple_bnode_calibration_results.bson` (restored)
- Checkpoints: `checkpoints/*.bson` (restored)
- Scripts: `scripts/generate_results_figures.py`, `scripts/verify_and_improve_figures.py`, `scripts/create_remaining_improved_figures.py`

All Results figures (1–9) are re-generated from these sources; Intro/Methods figures (A–I) are methodological/diagnostic but consistent with the paper’s narrative.

---

## Results Figures

### Fig 1. UDE vs Physics (RMSE $x_2$) — Scenario-wise Scatter
- What it shows: Per-scenario comparison of RMSE($x_2$) for the physics baseline (x-axis) vs UDE (y-axis). $x_2$ is the frequency/power deviation — your primary operational metric.
- Paper alignment: The paper emphasizes frequency stability as the operational priority. This scatter directly tests whether UDE preserves physics-level accuracy on $x_2$.
- Numbers (from metrics): Mean$\Delta$ (UDE−Physics) = −0.004488; 95% BCa CI [−0.039326, 0.030760]; Wilcoxon p = 0.9219; Cohen’s $d_z$ = −0.0747. These match the manuscript claims and indicate no significant difference.
- Methods rationale: Paired, non-parametric Wilcoxon signed-rank test is robust to non-normal errors common in power dynamics. BCa bootstrap (Efron 1987) provides bias/acceleration-corrected intervals.
- How to read (beginner): Each dot is one test scenario. If a dot lies below the diagonal, UDE’s error is smaller than Physics for that scenario. The cluster around the diagonal indicates comparable accuracy.
- Why it matters: Confirms that adding a small learned residual did not degrade the baseline physics model on the most critical operational variable.
- Common pitfalls: Don’t over-interpret one or two outliers — look at the mean difference and its confidence interval. A wide CI crossing zero typically means “no significant difference.”
- Truth checks: Recompute mean$\Delta$, CI, and Wilcoxon on `results/comprehensive_metrics.csv` (we did; values match the figure and paper).
- Reproduce (quick): `python3 scripts/generate_results_figures.py` → `figures/fig1_*.pdf|png`.
- Paper placement: Section “Experiments and Results → Overall Performance and Statistical Analysis”, place after the sentence ending “Cohen's $d_z = -0.0747$; matched-pairs $r \approx 0.955$.”

![Fig1](figures/fig1_scatter_rmse_x2_ude_vs_physics_improved.png)

### Fig 2. Histogram of $\Delta$RMSE($x_2$) = UDE − Physics
- What it shows: Distribution of scenario-wise differences; complements Fig 1 with a univariate perspective.
- Numbers: Centered near 0 with the same mean and CI as Fig 1; no skew indicating systematic degradation.
- How to read: Bars to the left of zero mean UDE is better for those scenarios; to the right means Physics is better. The vertical dashed lines can show the confidence interval.
- Why it matters: Consolidates 10 scenario comparisons into a single view of typical differences.
- Truth checks: Differences computed directly from `results/comprehensive_metrics.csv`.
- Reproduce: `python3 scripts/generate_results_figures.py` → `figures/fig2_*.pdf|png`.
- Paper placement: Same subsection as Fig 1; place immediately after Fig 1 to visualize the paired differences distribution.

![Fig2](figures/fig2_hist_delta_rmse_x2_ude_minus_physics_improved.png)

### Fig 3. Bland–Altman Analysis for RMSE($x_2$)
- What it shows: Mean-of-methods vs. difference; visualizes bias and limits of agreement (LoA).
- Numbers: Bias ≈ −0.0045 with LoA consistent with bootstrap CI. No trend across difficulty levels.
- How to read: Dots near the horizontal zero-line indicate similar performance; a flat cloud means no systematic bias at different error magnitudes.
- Why it matters: Bland–Altman is a standard agreement assessment and supports the “comparable accuracy” claim.
- Truth checks: Differences and LoA derived from the same metrics file; CI matches paper.
- Reproduce: `python3 scripts/generate_results_figures.py` → `figures/fig3_*.pdf|png`.
- Paper placement: Same subsection; place after Fig 2 to provide an agreement view complementary to the histogram.

![Fig3](figures/fig3_bland_altman_rmse_x2_improved.png)

### Fig 4. Paired Lines (RMSE $x_2$ by Scenario)
- What it shows: Within-scenario deltas (Physics → UDE). Downward-right lines favor UDE.
- Numbers: Global Wilcoxon p ≈ 0.92; $d_z \approx$ −0.075 — negligible effect.
- How to read: Each line connects the same scenario under both models; the slope direction quickly indicates which model did better.
- Why it matters: Makes scenario-level outcomes tangible; shows the absence of consistent wins for either model.
- Truth checks: Per-scenario values pulled from `results/comprehensive_metrics.csv`.
- Reproduce: `python3 scripts/generate_results_figures.py` → `figures/fig4_*.pdf|png`.
- Paper placement: Same subsection; place immediately after the sentence “Per-scenario paired comparisons are visualized in Fig. 4” (or add that sentence), to illustrate scenario-level variability.

![Fig4](figures/fig4_paired_lines_rmse_x2_by_model_improved.png)

### Fig 5. $\Delta R^2(x_2)$ = UDE − Physics (Per-Scenario)
- What it shows: Effect on explained variance in the critical variable.
- Numbers: Mean $\Delta R^2 \approx −0.032$ (from restored metrics). Small negative mean aligned with neutral RMSE — differences are practically small.
- How to read: Bars below zero mean UDE explains slightly less variance for that scenario.
- Why it matters: $R^2$ helps gauge “explanatory” performance; here it confirms that UDE is not trading accuracy for instability.
- Truth checks: Derived from `results/comprehensive_metrics.csv` per scenario.
- Reproduce: `python3 scripts/generate_results_figures.py` → `figures/fig5_*.pdf|png`.
- Paper placement: Same subsection; place after the paragraph summarizing $R^2$ results, i.e., following the sentence mentioning “$R^2$ $x_2$ … Physics 0.780 vs UDE 0.764.”

![Fig5](figures/fig5_r2x2_delta_ude_minus_physics_improved.png)

### Fig 6. BNODE Calibration (Pre vs Post)
- What it shows: Empirical coverage after post-hoc variance scaling; reliability vs. diagonal.
- Numbers (restored): 50%/90% coverage ≈ 0.541/0.849; large NLL reduction reported in the paper. These are consistent with the calibration summary in results.
- Methods rationale: Post-hoc variance scaling is a pragmatic and accepted approach for improving predictive interval calibration in probabilistic models (Gneiting & Raftery 2007; Kuleshov et al. 2018). Reliability diagrams are the standard visualization.
- How to read: Post-calibration curve close to the 45° line means predictive intervals are well-calibrated; markers at (0.5, 0.541) and (0.9, 0.849) summarize key coverages.
- Why it matters: For safe operation, uncertainty must be reliable; calibrated BNODE supports chance-constrained control.
- Truth checks: Parsed directly from `results/simple_bnode_calibration_results.bson`.
- Reproduce: `python3 scripts/generate_results_figures.py` → `figures/fig6_*.pdf|png`.
- Paper placement: Section “Experiments and Results → Uncertainty Quantification and Calibration Analysis”, place after the paragraph ending “post-calibration 50%/90% coverage $\approx 0.541/0.849$.”

![Fig6](figures/fig6_calibration_bnode_pre_post_improved.png)

### Fig 7. Baseline Summary (RMSE($x_2$), $R^2(x_2$))
- What it shows: Mean±variability across scenarios for point metrics; positions Physics and UDE in context and separates BNODE’s role as UQ-focused.
- How to read: Error bars show variability; consistent overlap means models are comparable.
- Why it matters: Summarizes the story of Figs 1–5 succinctly.
- Truth checks: Averages computed from `results/comprehensive_metrics.csv`.
- Reproduce: `python3 scripts/generate_results_figures.py` → `figures/fig7_*.pdf|png`.
- Paper placement: Same subsection; place directly after the performance table to provide a visual complement.

![Fig7](figures/fig7_baselines_rmse_r2_summary_improved.png)

### Fig 8. Runtime Comparison (Inference Latency)
- What it shows: Mean±variability of per-trajectory inference time.
- Numbers: UDE ≈ 0.27 ± 0.05 ms; Physics ≈ 0.08 ± 0.01 ms (as reported). Both sub-ms.
- How to read: Bars with small error bars indicate consistent fast inference; both bars being well below a 1–5 ms budget implies real-time viability.
- Why it matters: Confirms feasibility for control loops with tight timing.
- Truth checks: CSVs present in the source context; figure generation uses those; end-to-end we re-ran in the main project to produce the PDFs included here.
- Reproduce: `python3 scripts/generate_results_figures.py` → `figures/fig8_*.pdf|png`.
- Paper placement: Section “Experiments and Results → Computational Efficiency and Runtime Analysis”, place after the sentence stating “UDE achieves $0.27 \pm 0.05$ ms … Physics $0.08 \pm 0.01$ ms …”.

![Fig8](figures/fig8_runtime_comparison_improved.png)

### Fig 9. Symbolic Residual $f_\theta(P)$ (Cubic)
- What it shows: Interpretable surrogate for the learned residual — a cubic with $R^2 \approx 0.982$.
- Numbers: $f_\theta(P) \approx −0.055463 + 0.835818P + 0.000875P^2 − 0.018945P^3$ (matches manuscript corrections).
- How to read: The black curve captures how the residual correction changes with generation level; subtle downward curvature at high $P$ indicates saturation.
- Why it matters: Provides a direct, physics-grounded handle for adaptive droop design (reduce effective coupling at high generation to maintain stability).
- Truth checks: Coefficients match the restored summary/paper; we recommend adding `results/ude_symbolic_extraction.md` to the backup to eliminate the remaining WARN.
- Reproduce: `python3 scripts/create_remaining_improved_figures.py` → `figures/fig9_*.pdf|png`.
- Paper placement: Section “Experiments and Results → Symbolic Extraction and Physical Interpretation”, place immediately after the cubic equation block.

![Fig9](figures/fig9_symbolic_extraction_fit_improved.png)

---

## Intro / Methods / Diagnostics (A–I)

### Fig A. Microgrid Architecture and Dynamics
- Role in paper: Grounds the reader in system components and the operational meaning of $x_1$ (SoC) and $x_2$ (frequency/power deviation). Aligns with the paper’s rationale for focusing on $x_2$.
- How to read: The left panel shows the physical topology (main grid, microgrid boundary, DERs, storage, loads). The right panel shows typical time courses: oscillatory $x_2$ transients and slower $x_1$ drift.
- Common pitfalls: Don’t conflate $x_2$ with frequency in Hz — here it’s normalized deviation (p.u.).
- Reproduce: `python3 scripts/generate_intro_figures.py` (creates `clean_figures_final/figA_*`).
- Paper placement: Section “Introduction and Related Work”, place after the paragraph ending “droop control mechanisms that dynamically couple active power with frequency and reactive power with voltage …”.

![FigA](figures/figA_microgrid_architecture.png)

### Fig B. PINN Limitations (Gradient Pathologies, Spectral Bias, Stiffness, Competing Losses)
- Context (beginner view): PINNs minimize a weighted sum of data and physics losses; with stiff dynamics and multi-scale content, training can get stuck in trivial or over-smooth solutions.
- Literature cues: Krishnapriyan et al. (failure modes), Rahaman et al. (spectral bias), Ji & Lu (stiffness), Wang et al. (gradient pathologies).
- Why it matters: Justifies we used UDE/BNODE rather than pure PINNs for this microgrid.
- Reproduce: `python3 scripts/generate_intro_figures.py` (creates `figB_*`).
- Paper placement: Section “Introduction and Related Work → Physics-Informed Neural Networks and Their Limitations”, place after that paragraph block.

![FigB](figures/figB_pinn_limitations.png)

### Fig C. UDE vs BNODE (Conceptual)
- UDE: Physics + small learned residual → preserves constraints and adds flexibility (Rackauckas et al.).
- BNODE: Full neural vector field + calibrated uncertainty → supports risk-aware planning.
- Teaching tip: Think of UDE as “physics with a smart correction,” BNODE as “a learned model that also tells you how uncertain it is.”
- Reproduce: `python3 scripts/generate_intro_figures.py` (creates `figC_*`).
- Paper placement: Section “Introduction and Related Work → Neural Differential Equations and Structure Preservation / Calibration and Uncertainty Quantification”, place after the BNODE paragraph that ends with “provides principled model selection.”

![FigC](figures/figC_ude_bnode_comparison.png)

### Fig D. Two-State Model + UDE Modification
- What it conveys: Exactly where the learned residual enters the model ($\beta P_{gen} \to f_\theta(P_{gen})$) while preserving storage physics.
- Teaching tip: This design is why interpretability is retained — only one coupling is learned.
- Reproduce: `python3 scripts/generate_methods_figures.py` (creates `figD_*`).
- Paper placement: Section “Methods → Two-State Microgrid Model / UDE Approach”, place after the equation block and the UDE modification equations.

![FigD](figures/figD_model_architecture.png)

### Fig E. UDE Training, Tolerances, Capacity
- Why strict tolerances: Neural residuals can interact with solver stability; stiff solvers (Rosenbrock family) and tight tolerances prevent numerical artifacts.
- Teaching tip: Small networks with regularization are enough when the residual is one-dimensional and physics does most of the work.
- Reproduce: `python3 scripts/generate_methods_figures.py` (creates `figE_*`).
- Paper placement: Section “Methods → Universal Differential Equation Approach → Training methodology and optimization”, place after that paragraph.

![FigE](figures/figE_ude_training.png)

### Fig F. BNODE Diagnostics (Trace, $\hat{R}$, ESS, Reliability)
- What to look for: Stable traces (no drift), $\hat{R}\approx 1$, adequate ESS, reliability near the diagonal after calibration.
- Teaching tip: Diagnostics are mandatory before trusting predictive intervals in critical infrastructure.
- Reproduce: `python3 scripts/generate_methods_figures.py` (creates `figF_*`).
- Paper placement: Section “Methods → Bayesian Neural ODE Approach / Calibration and Evaluation Framework”, place after the “Hamiltonian Monte Carlo and posterior inference” paragraph (or at the start of Calibration subsection).

![FigF](figures/figF_bnode_diagnostics.png)

### Fig G. Training Data (Improved; Data-Derived)
- What it shows: Coverage of parameter space, representative trajectories, and split sizes — from the actual `data/*.csv`.
- Why it matters: Verifies no leakage and highlights diversity of operating conditions.
- Reproduce (in main project): `python3 scripts/create_remaining_improved_figures.py` (creates `figG_*`).
- Paper placement: Section “Methods → Data generation and experimental design”, place after that paragraph.

![FigG](figures/figG_training_data_improved.png)

### Fig H. Validation Curves and Learning Dynamics
- What to look for: Training/validation losses decreasing; RMSE↓ and $R^2$↑; absence of oscillatory divergence.
- Why it matters: Confirms stable training dynamics compared to known PINN pitfalls.
- Reproduce (in main project): `python3 scripts/create_remaining_improved_figures.py` (creates `figH_*`).
- Paper placement: Section “Methods → Statistical methodology and significance testing framework”, place after that paragraph to visually summarize optimization stability.

![FigH](figures/figH_validation_curves.png)

### Fig I. Ablations
- Takeaways: Width=3, $\lambda=10^{-6}$, tight solver tolerance balance fit/stability/runtime for this two-state problem.
- Teaching tip: Favor the smallest model that meets accuracy targets; it tends to be more stable and interpretable.
- Reproduce (in main project): `python3 scripts/create_remaining_improved_figures.py` (creates `figI_*`).
- Paper placement: Section “Experiments and Results → Model Configuration and Hyperparameter Analysis”, place after the first paragraph of that subsection.

![FigI](figures/figI_ablation_study.png)

---

## How These Figures Substantiate the Paper’s Key Claims
1) Physics-comparable point accuracy on $x_2$ (Figs 1–5,7):
   - Neutral aggregate effect with tight intervals; UDE preserves core accuracy while enabling interpretation.
2) Calibrated uncertainty enabling risk-aware constraints (Fig 6):
   - Post-calibration coverage close to nominal supports chance-constrained planning.
3) Interpretable physics-grounded insight (Fig 9):
   - Symbolic residual reveals mild saturation; yields actionable droop adaptation guidance.
4) Real-time suitability (Fig 8):
   - Sub-ms inference across scenarios supports control-loop deployment.

---

## Selected Literature Pointers
- BCa intervals and bootstrap: Efron (1987).
- Proper scoring rules and reliability (calibration): Gneiting & Raftery (2007); Kuleshov et al. (2018).
- UDEs / physics-grounded neural surrogates: Rackauckas et al. (2020); Baker et al. (2022).
- BNODEs / Bayesian dynamical modeling: Dandekar et al. (2021); Gelman et al. (2013) for PPC.
- PINN limitations: Krishnapriyan et al. (2021); Rahaman et al. (2019); Ji & Lu (2021); Wang et al. (2021, 2022).
- Stiff solvers: Hairer & Wanner (1996).

---

## Reproducibility and Provenance
- Results 1–9 are generated from restored inputs (`results/comprehensive_metrics.csv`, `results/simple_bnode_calibration_results.bson`, `data/*.csv`, `checkpoints/*.bson`) using the committed scripts. Commit history in this backup repo documents exact versions.
- The merged PDF `figures/all_figures_merged.pdf` contains improved Results (pp. 1–9) followed by Intro/Methods/Additional (pp. 10–18) in the same order as the paper narrative.
