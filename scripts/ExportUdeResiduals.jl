#!/usr/bin/env julia

"""
ExportUdeResiduals.jl

Module for exporting real UDE residual samples to CSV format.
Provides functions to write P_gen and f_residual arrays with metadata.
"""

module ExportUdeResiduals

using Dates, SHA, CSV, DataFrames, JSON3, BSON, Statistics

"""
write_ude_residual_csv(p_gen::AbstractVector, f_residual::AbstractVector;
                       scenario::Union{Nothing,AbstractVector}=nothing,
                       out_csv::AbstractString="results/ude_f_Pgen_samples.csv")

Writes CSV with columns: P_gen, f_residual, (optional) scenario.
Also writes results/ude_f_Pgen_samples.meta.json with sha256, rows, commit, timestamp.
Raises if length mismatch, NaNs, or <100 valid rows.
"""
function write_ude_residual_csv(p_gen, f_residual; scenario=nothing,
                                out_csv::AbstractString="results/ude_f_Pgen_samples.csv")
    @assert length(p_gen) == length(f_residual) "length mismatch: p_gen=$(length(p_gen)), f_residual=$(length(f_residual))"
    
    # Create DataFrame
    df = DataFrame(P_gen = collect(p_gen), f_residual = collect(f_residual))
    if scenario !== nothing
        @assert length(scenario) == nrow(df) "scenario length mismatch: scenario=$(length(scenario)), df=$(nrow(df))"
        df.scenario = collect(scenario)
    end
    
    # Drop NaNs/Infs
    initial_rows = nrow(df)
    filter!(r -> isfinite(r.P_gen) && isfinite(r.f_residual), df)
    final_rows = nrow(df)
    
    if final_rows < initial_rows
        println("⚠️  Dropped $(initial_rows - final_rows) non-finite rows")
    end
    
    @assert nrow(df) ≥ 100 "too few valid rows for reliable fit: $(nrow(df)) < 100"
    
    # Ensure results/ directory exists
    mkpath(dirname(out_csv))
    
    # Write CSV
    CSV.write(out_csv, df)
    println("✅ Wrote $(nrow(df)) rows to $out_csv")
    
    # Compute SHA256
    io = open(out_csv, "r")
    bytes = read(io)
    close(io)
    h = bytes2hex(sha256(bytes))
    
    # Get git commit (best-effort)
    commit = try 
        readchomp(`git rev-parse --short HEAD`)
    catch
        "unknown"
    end
    
    # Create metadata
    meta = Dict(
        "source" => "ExportUdeResiduals.write_ude_residual_csv",
        "git_commit" => commit,
        "rows" => nrow(df),
        "sha256_csv" => h,
        "created_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        "schema" => Dict("P_gen"=>"Float64", "f_residual"=>"Float64", "scenario"=>"optional String"),
        "validation" => Dict(
            "min_p_gen" => minimum(df.P_gen),
            "max_p_gen" => maximum(df.P_gen),
            "min_f_residual" => minimum(df.f_residual),
            "max_f_residual" => maximum(df.f_residual),
            "mean_p_gen" => mean(df.P_gen),
            "mean_f_residual" => mean(df.f_residual)
        )
    )
    
    # Write metadata
    meta_path = replace(out_csv, ".csv" => ".meta.json")
    open(meta_path, "w") do io
        write(io, JSON3.write(meta, pretty=true))
    end
    println("✅ Wrote metadata to $meta_path")
    
    return out_csv, h, nrow(df)
end

"""
extract_residuals_from_evaluation(test_data::DataFrame, ude_params::Vector, width::Int, β::Float64)

Extracts P_gen and f_residual arrays from test data evaluation.
Computes f_residual = f_θ(P_gen) - β * P_gen where f_θ is the learned UDE function.
"""
function extract_residuals_from_evaluation(test_data::DataFrame, ude_params::Vector, width::Int, β::Float64)
    println("🔍 Extracting residuals from test data evaluation...")
    
    # UDE residual function (matching training script)
    function ftheta(x, θ, width::Int)
        W = reshape(θ[1:width], width, 1)
        b = θ[width+1:width+width]
        h = tanh.(W * [x] .+ b)
        return sum(h)
    end
    
    # Extract P_gen values and compute residuals
    p_gen_vec = Float64[]
    f_residual_vec = Float64[]
    scenario_vec = String[]
    
    scenarios = unique(test_data.scenario)
    println("  Processing $(length(scenarios)) scenarios...")
    
    for scenario in scenarios
        scenario_data = test_data[test_data.scenario .== scenario, :]
        p_gen_vals = scenario_data.Pgen
        
        for p_gen in p_gen_vals
            # Compute learned contribution
            f_learned = ftheta(p_gen, ude_params, width)
            
            # Compute residual (learned - physics baseline)
            f_residual = f_learned - β * p_gen
            
            push!(p_gen_vec, p_gen)
            push!(f_residual_vec, f_residual)
            push!(scenario_vec, scenario)
        end
    end
    
    println("  Extracted $(length(p_gen_vec)) residual samples")
    println("  P_gen range: [$(minimum(p_gen_vec)), $(maximum(p_gen_vec))]")
    println("  f_residual range: [$(minimum(f_residual_vec)), $(maximum(f_residual_vec))]")
    
    return p_gen_vec, f_residual_vec, scenario_vec
end

"""
export_from_checkpoint(checkpoint_path::String, test_data_path::String, β::Float64=1.0)

Convenience function to export residuals from a UDE checkpoint and test data.
"""
function export_from_checkpoint(checkpoint_path::String, test_data_path::String, β::Float64=1.0)
    println("📁 Loading UDE checkpoint from $checkpoint_path...")
    
    # Load checkpoint
    ude_checkpoint = BSON.load(checkpoint_path)
    best_cfg = ude_checkpoint[:best_cfg]
    width, λ, lr, reltol, seed = best_cfg
    ude_params = ude_checkpoint[:best_ckpt]
    
    println("  UDE config: width=$width, λ=$λ, lr=$lr, reltol=$reltol, seed=$seed")
    println("  UDE parameters: $(length(ude_params)) parameters")
    
    # Load test data
    println("📊 Loading test data from $test_data_path...")
    test_data = CSV.read(test_data_path, DataFrame)
    println("  Test data: $(nrow(test_data)) rows, $(length(unique(test_data.scenario))) scenarios")
    
    # Extract residuals
    p_gen_vec, f_residual_vec, scenario_vec = extract_residuals_from_evaluation(test_data, ude_params, width, β)
    
    # Write CSV
    out_csv, sha256_hash, n_rows = write_ude_residual_csv(p_gen_vec, f_residual_vec; scenario=scenario_vec)
    
    println("✅ Export complete!")
    println("  Output: $out_csv")
    println("  Rows: $n_rows")
    println("  SHA256: $sha256_hash")
    
    return out_csv, sha256_hash, n_rows
end

end # module
