# scripts/fig_manifest.py
# Canonical list of improved figures to (re)generate and include in PPT.
# name -> (title, caption)
FIGS = {
    "reliability_diagram_improved": (
        "BNODE Reliability (Empirical vs Nominal)",
        "Empirical coverage across quantiles (10–90%) after post-calibration; square axes and diagonal reference.",
    ),
    "posterior_predictive_checks_improved": (
        "Posterior Predictive Checks",
        "Observed x₂(t) with BNODE median and 50/90% intervals; representative scenarios; no divergences; R̂ ≤ 1.01.",
    ),
    "calibration_error_improved": (
        "Calibration Sweep",
        "Coverage error and NLL vs α_cal; optimum near α_cal≈1.8 with ~98.5% NLL reduction.",
    ),
    "ude_residual_cubic_improved": (
        "UDE Residual: Symbolic Cubic",
        "fθ(P_gen) and cubic fit (R²≈0.982); mild saturation at high P_gen; coefficients match paper.",
    ),
    "ude_ablations_improved": (
        "UDE Ablations",
        "Width/λ sweeps; x₂ RMSE variance < 0.01; chosen width=3, λ=1e-6.",
    ),
    "noise_robustness_improved": (
        "Noise Robustness",
        "UDE RMSE and BNODE interval widths vs σ∈{0.01,0.05,0.1}; BNODE widens intervals to preserve coverage.",
    ),
    "runtime_boxplots_improved": (
        "Runtime & Deployment",
        "Sub-ms UDE trajectories; boxplots by environment; banner with CPU/GPU/solver.",
    ),
    "bland_altman_improved": (
        "Bland–Altman (UDE vs Physics)",
        "Mean difference & 95% LoA for x₂; helper ensures LoA lines and counts; units + N in panel.",
    ),
    "coverage_by_scenario_improved": (
        "Coverage by Scenario",
        "Per-scenario 50%/90% coverage with CIs; highlights any drift vs nominal.",
    ),
    "error_scatter_improved": (
        "Scenario-wise Errors",
        "RMSE/MAE scatter for x₂ with jitter; shows small practical Δ favoring UDE.",
    ),
    "error_hist_improved": (
        "Error Distribution",
        "Histogram/KDE for x₂ residuals; tails annotated; consistent binning.",
    ),
    "ece_bins_improved": (
        "Reliability (ECE bins)",
        "Expected calibration error in fixed bins; bar plot with exact percentages.",
    ),
}
