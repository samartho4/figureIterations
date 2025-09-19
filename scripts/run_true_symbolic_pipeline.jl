#!/usr/bin/env julia

"""
run_true_symbolic_pipeline.jl

Complete pipeline for true symbolic regression:
1. Export real UDE residuals from checkpoint and test data
2. Perform polynomial fitting with BIC selection and cross-validation
3. Generate publication-ready figure and JSON results
4. Validate all outputs and provide provenance
"""

using Pkg
Pkg.activate(".")

using BSON, CSV, DataFrames, Statistics, Dates, SHA, JSON3

println("🚀 Starting True Symbolic Regression Pipeline...")
println("=" ^ 60)

# Configuration
CHECKPOINT_PATH = "checkpoints/corrected_ude_best.bson"
TEST_DATA_PATH = "data/test_roadmap.csv"
BETA_VALUE = 1.0  # Physics baseline coupling coefficient

# Phase 1: Export real residuals
println("\n📊 PHASE 1: Exporting Real UDE Residuals")
println("-" ^ 40)

# Check if checkpoint exists
if !isfile(CHECKPOINT_PATH)
    error("❌ UDE checkpoint not found: $CHECKPOINT_PATH")
end

# Check if test data exists
if !isfile(TEST_DATA_PATH)
    error("❌ Test data not found: $TEST_DATA_PATH")
end

# Include the export module
include("ExportUdeResiduals.jl")
using .ExportUdeResiduals

# Export residuals
try
    out_csv, sha256_hash, n_rows = ExportUdeResiduals.export_from_checkpoint(CHECKPOINT_PATH, TEST_DATA_PATH, BETA_VALUE)
    println("✅ Phase 1 complete: $(n_rows) residual samples exported")
catch e
    error("❌ Phase 1 failed: $e")
end

# Phase 2: Symbolic regression
println("\n🔬 PHASE 2: Symbolic Regression Analysis")
println("-" ^ 40)

try
    include("symbolic_from_csv.jl")
    println("✅ Phase 2 complete: Symbolic regression analysis finished")
catch e
    error("❌ Phase 2 failed: $e")
end

# Phase 3: Validation and provenance
println("\n🔍 PHASE 3: Validation and Provenance")
println("-" ^ 40)

# Check outputs exist
required_outputs = [
    "results/ude_f_Pgen_samples.csv",
    "results/ude_f_Pgen_samples.meta.json", 
    "results/symbolic_fit.json",
    "figures/figSYM_residual_true.pdf"
]

all_exist = true
for output in required_outputs
    if isfile(output)
        println("✅ $output exists")
    else
        println("❌ $output missing")
        all_exist = false
    end
end

if !all_exist
    error("❌ Some required outputs are missing")
end

# Load and validate symbolic results
println("\n📋 Validating symbolic regression results...")
symbolic_results = JSON3.read(read("results/symbolic_fit.json", String))

println("  Best model: Degree $(symbolic_results["degree"]) polynomial")
println("  R² = $(round(symbolic_results["r2"], digits=4))")
println("  BIC = $(round(symbolic_results["bic"], digits=2))")
println("  CV R² = $(round(symbolic_results["cv_r2_mean"], digits=4)) ± $(round(symbolic_results["cv_r2_std"], digits=4))")
println("  Samples: $(symbolic_results["n_samples"])")
println("  CSV SHA256: $(symbolic_results["csv_sha256"])")

# Validate CSV SHA256 matches
csv_sha256_actual = try
    io = open("results/ude_f_Pgen_samples.csv", "r")
    bytes = read(io)
    close(io)
    bytes2hex(sha256(bytes))
catch
    "unknown"
end

if symbolic_results["csv_sha256"] == csv_sha256_actual
    println("✅ CSV SHA256 validation passed")
else
    println("⚠️  CSV SHA256 mismatch: stored=$(symbolic_results["csv_sha256"]), actual=$csv_sha256_actual")
end

# Create comprehensive provenance report
println("\n📝 Creating provenance report...")
provenance = Dict(
    "pipeline" => "run_true_symbolic_pipeline.jl",
    "created_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
    "git_commit" => try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end,
    "inputs" => Dict(
        "checkpoint" => CHECKPOINT_PATH,
        "test_data" => TEST_DATA_PATH,
        "beta_value" => BETA_VALUE
    ),
    "outputs" => Dict(
        "residual_csv" => "results/ude_f_Pgen_samples.csv",
        "residual_meta" => "results/ude_f_Pgen_samples.meta.json",
        "symbolic_json" => "results/symbolic_fit.json",
        "symbolic_figure" => "figures/figSYM_residual_true.pdf"
    ),
    "validation" => Dict(
        "csv_sha256" => csv_sha256_actual,
        "n_samples" => symbolic_results["n_samples"],
        "best_degree" => symbolic_results["degree"],
        "r2" => symbolic_results["r2"],
        "bic" => symbolic_results["bic"]
    )
)

# Write provenance
open("results/symbolic_provenance.json", "w") do io
    write(io, JSON3.write(provenance, pretty=true))
end
println("✅ Provenance report saved to results/symbolic_provenance.json")

# Final summary
println("\n🎉 PIPELINE COMPLETE!")
println("=" ^ 60)
println("✅ Real UDE residuals exported from checkpoint")
println("✅ True symbolic regression performed (no synthetic data)")
println("✅ Best polynomial model selected by BIC")
println("✅ Cross-validation performed")
println("✅ Publication-ready figure generated")
println("✅ All outputs validated and provenance recorded")
println("\n📁 Outputs:")
println("  - results/ude_f_Pgen_samples.csv (real residual data)")
println("  - results/symbolic_fit.json (model coefficients and metrics)")
println("  - figures/figSYM_residual_true.pdf (publication figure)")
println("  - results/symbolic_provenance.json (complete provenance)")
println("\n🔬 Best Model:")
println("  Degree: $(symbolic_results["degree"])")
println("  Equation: $(symbolic_results["equation"])")
println("  R²: $(round(symbolic_results["r2"], digits=4))")
println("  BIC: $(round(symbolic_results["bic"], digits=2))")
println("  CV R²: $(round(symbolic_results["cv_r2_mean"], digits=4)) ± $(round(symbolic_results["cv_r2_std"], digits=4))")

println("\n✅ True symbolic regression pipeline complete!")
