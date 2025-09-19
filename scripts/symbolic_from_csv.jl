#!/usr/bin/env julia

"""
symbolic_from_csv.jl

Performs true symbolic regression on real UDE residual data from CSV.
Fits polynomial models, selects best by BIC, generates figure and JSON results.
"""

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, StatsBase, Plots, JSON3, SHA, Dates, LinearAlgebra, Random

println("🔬 Starting True Symbolic Regression Analysis...")

# Configuration
CSV_PATH = "results/ude_f_Pgen_samples.csv"
META_PATH = "results/ude_f_Pgen_samples.meta.json"
OUTPUT_JSON = "results/symbolic_fit.json"
OUTPUT_PDF = "figures/figSYM_residual_true.pdf"

# Check if CSV exists
if !isfile(CSV_PATH)
    error("❌ Required CSV not found: $CSV_PATH. Run ExportUdeResiduals first.")
end

# Load data
println("📊 Loading residual data from $CSV_PATH...")
df = CSV.read(CSV_PATH, DataFrame)
println("  Loaded $(nrow(df)) samples")

# Validate data
required_cols = ["P_gen", "f_residual"]
for col in required_cols
    if !(col in names(df))
        error("❌ Missing required column: $col")
    end
end

# Extract arrays
x = df.P_gen
y = df.f_residual
n = length(x)

println("  P_gen range: [$(minimum(x)), $(maximum(x))]")
println("  f_residual range: [$(minimum(y)), $(maximum(y))]")
println("  Sample size: $n")

# Helper functions
function r2(ŷ, y)
    ss_res = sum((y .- ŷ).^2)
    ss_tot = sum((y .- mean(y)).^2)
    return 1 - ss_res/ss_tot
end

function bic(ŷ, y, k)
    n = length(y)
    σ2 = sum((y .- ŷ).^2)/n
    return n*log(σ2 + eps()) + k*log(n)
end

# Manual polynomial fitting functions
function fit_polynomial(x, y, degree)
    # Create Vandermonde matrix
    n = length(x)
    A = zeros(n, degree + 1)
    for i in 1:n
        for j in 1:(degree + 1)
            A[i, j] = x[i]^(j-1)
        end
    end
    
    # Solve least squares
    coeffs = A \ y
    return coeffs
end

function evaluate_polynomial(x, coeffs)
    y = zeros(length(x))
    for i in 1:length(x)
        for j in 1:length(coeffs)
            y[i] += coeffs[j] * x[i]^(j-1)
        end
    end
    return y
end

function cross_validation(x, y, degree, n_folds=5)
    n = length(x)
    indices = shuffle(1:n)
    fold_size = div(n, n_folds)
    
    cv_scores = Float64[]
    
    for fold in 1:n_folds
        # Split data
        start_idx = (fold-1) * fold_size + 1
        end_idx = fold == n_folds ? n : fold * fold_size
        
        test_indices = indices[start_idx:end_idx]
        train_indices = setdiff(indices, test_indices)
        
        x_train, y_train = x[train_indices], y[train_indices]
        x_test, y_test = x[test_indices], y[test_indices]
        
        # Fit polynomial
        coeffs = fit_polynomial(x_train, y_train, degree)
        y_pred = evaluate_polynomial(x_test, coeffs)
        
        # Compute R²
        r2_score = r2(y_pred, y_test)
        push!(cv_scores, r2_score)
    end
    
    return mean(cv_scores), std(cv_scores)
end

# Fit polynomial models (degrees 1-5)
println("🔧 Fitting polynomial models...")
best_model = nothing
best_bic = Inf

results = Dict[]

for degree in 1:5
    println("  Testing degree $degree...")
    
    # Fit polynomial
    coeffs = fit_polynomial(x, y, degree)
    y_pred = evaluate_polynomial(x, coeffs)
    
    # Compute metrics
    r2_score = r2(y_pred, y)
    bic_score = bic(y_pred, y, degree + 1)  # k = degree + 1 (including constant)
    
    # Cross-validation
    cv_mean, cv_std = cross_validation(x, y, degree)
    
    # Store results
    result = Dict(
        "degree" => degree,
        "coeffs" => coeffs,
        "r2" => r2_score,
        "bic" => bic_score,
        "cv_r2_mean" => cv_mean,
        "cv_r2_std" => cv_std
    )
    push!(results, result)
    
    println("    R² = $(round(r2_score, digits=4))")
    println("    BIC = $(round(bic_score, digits=2))")
    println("    CV R² = $(round(cv_mean, digits=4)) ± $(round(cv_std, digits=4))")
    
    # Check if best
    if bic_score < best_bic
        global best_bic = bic_score
        global best_model = result
    end
end

println("✅ Best model: degree $(best_model["degree"])")
println("  R² = $(round(best_model["r2"], digits=4))")
println("  BIC = $(round(best_model["bic"], digits=2))")
println("  CV R² = $(round(best_model["cv_r2_mean"], digits=4)) ± $(round(best_model["cv_r2_std"], digits=4))")

# Reconstruct best polynomial for plotting
best_coeffs = best_model["coeffs"]
x_plot = range(minimum(x), maximum(x), length=200)
y_plot = evaluate_polynomial(x_plot, best_coeffs)

# Generate figure
println("🎨 Generating symbolic regression figure...")

# Create plot with subplots
p = plot(layout=(2,1), size=(800, 1000))

# Main plot: scatter + fit
scatter!(p[1], x, y, 
         alpha=0.6, color=:blue, markersize=3,
         label="Data", xlabel="P_gen", ylabel="f_residual",
         title="True UDE Residual Symbolic Regression")

plot!(p[1], x_plot, y_plot, 
      color=:red, linewidth=3, label="Best Fit (deg $(best_model["degree"]))")

# Add equation overlay
coeffs_str = [string(round(c, digits=6)) for c in best_model["coeffs"]]
if best_model["degree"] == 1
    eq_text = "f(P) = $(coeffs_str[1]) + $(coeffs_str[2])P"
elseif best_model["degree"] == 2
    eq_text = "f(P) = $(coeffs_str[1]) + $(coeffs_str[2])P + $(coeffs_str[3])P²"
elseif best_model["degree"] == 3
    eq_text = "f(P) = $(coeffs_str[1]) + $(coeffs_str[2])P + $(coeffs_str[3])P² + $(coeffs_str[4])P³"
else
    eq_text = "f(P) = $(coeffs_str[1]) + ... + $(coeffs_str[end])P^$(best_model["degree"])"
end

annotate!(p[1], 0.05, 0.95, eq_text, 
          annotationfontsize=10, annotationcolor=:black,
          bbox=Dict(:boxwidth => 0.3, :boxheight => 0.1, :facecolor => :white, :alpha => 0.8))

# Stats overlay
stats_text = "R² = $(round(best_model["r2"], digits=4))\nBIC = $(round(best_model["bic"], digits=2))\nCV R² = $(round(best_model["cv_r2_mean"], digits=4))"
annotate!(p[1], 0.05, 0.75, stats_text,
          annotationfontsize=10, annotationcolor=:black,
          bbox=Dict(:boxwidth => 0.3, :boxheight => 0.15, :facecolor => :white, :alpha => 0.8))

# Residuals plot
y_pred_full = evaluate_polynomial(x, best_coeffs)
residuals = y .- y_pred_full

scatter!(p[2], x, residuals,
         alpha=0.6, color=:green, markersize=3,
         label="Residuals", xlabel="P_gen", ylabel="Residuals",
         title="Residuals vs P_gen")

hline!(p[2], [0], color=:black, linestyle=:dash, linewidth=2, label="Zero")

# Add residual stats
res_mean = mean(residuals)
res_std = std(residuals)
res_text = "Mean = $(round(res_mean, digits=6))\nStd = $(round(res_std, digits=6))"
annotate!(p[2], 0.05, 0.95, res_text,
          annotationfontsize=10, annotationcolor=:black,
          bbox=Dict(:boxwidth => 0.3, :boxheight => 0.1, :facecolor => :white, :alpha => 0.8))

# Ensure figures directory exists
mkpath(dirname(OUTPUT_PDF))

# Save figure
savefig(p, OUTPUT_PDF)
println("✅ Figure saved to $OUTPUT_PDF")

# Load metadata if available
csv_sha256 = "unknown"
if isfile(META_PATH)
    try
        meta = JSON3.read(read(META_PATH, String))
        csv_sha256 = meta["sha256_csv"]
        println("✅ Loaded CSV SHA256 from metadata: $csv_sha256")
    catch e
        println("⚠️  Could not load metadata: $e")
    end
else
    # Compute SHA256 directly
    try
        io = open(CSV_PATH, "r")
        bytes = read(io)
        close(io)
        csv_sha256 = bytes2hex(sha256(bytes))
        println("✅ Computed CSV SHA256: $csv_sha256")
    catch e
        println("⚠️  Could not compute SHA256: $e")
    end
end

# Create comprehensive results
symbolic_results = Dict(
    "source" => "symbolic_from_csv.jl",
    "csv_path" => CSV_PATH,
    "csv_sha256" => csv_sha256,
    "n_samples" => n,
    "family" => "polynomial",
    "degree" => best_model["degree"],
    "coeffs" => best_model["coeffs"],
    "r2" => best_model["r2"],
    "bic" => best_model["bic"],
    "cv_r2_mean" => best_model["cv_r2_mean"],
    "cv_r2_std" => best_model["cv_r2_std"],
    "equation" => eq_text,
    "all_models" => results,
    "created_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
    "git_commit" => try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end
)

# Save results
open(OUTPUT_JSON, "w") do io
    write(io, JSON3.write(symbolic_results, pretty=true))
end
println("✅ Results saved to $OUTPUT_JSON")

# Print summary
println("\n📊 SYMBOLIC REGRESSION SUMMARY")
println("=" ^ 50)
println("Best Model: Degree $(best_model["degree"]) polynomial")
println("Equation: $eq_text")
println("R² = $(round(best_model["r2"], digits=4))")
println("BIC = $(round(best_model["bic"], digits=2))")
println("CV R² = $(round(best_model["cv_r2_mean"], digits=4)) ± $(round(best_model["cv_r2_std"], digits=4))")
println("Samples: $n")
println("CSV SHA256: $csv_sha256")
println("\n✅ True symbolic regression complete!")
