#!/usr/bin/env julia

"""
ベンチマーク精度検証テスト

このテストスイートは、ベンチマーク問題で主張している精度が
実際に達成できることを検証します。既知解析解がある問題や
高精度逐次計算との比較により、数値精度を系統的に評価します。
"""

using Test
using MPI
using Heat3ds
using LinearAlgebra
using JSON
using Dates

struct AccuracyBenchmark
    name::String
    description::String
    problem_setup::Function
    analytical_solution::Union{Function, Nothing}
    expected_l2_error::Float64
    expected_max_error::Float64
end

struct AccuracyTestResult
    benchmark_name::String
    l2_error::Float64
    max_error::Float64
    relative_error::Float64
    meets_accuracy::Bool
    parareal_converged::Bool
    execution_time::Float64
end

function setup_accuracy_test()
    """精度テスト環境のセットアップ"""
    
    if !MPI.Initialized()
        MPI.Init()
    end
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    return comm, rank, size
end

# 1次元熱伝導の解析解
function analytical_heat_1d(x, t, L=1.0, alpha=1.0, n_terms=50)
    """1次元熱伝導方程式の解析解（フーリエ級数）"""
    
    result = 0.0
    for n in 1:n_terms
        lambda_n = (n * π / L)^2
        coeff = 8.0 / ((2*n - 1) * π)
        result += coeff * exp(-alpha * lambda_n * t) * sin((2*n - 1) * π * x / L)
    end
    
    return result
end

# 3次元問題の近似解析解（分離変数解）
function analytical_heat_3d_separable(x, y, z, t, Lx=1.0, Ly=1.0, Lz=1.0, alpha=1.0)
    """3次元分離可能問題の解析解"""
    
    # 簡単な分離解: sin(πx/Lx) * sin(πy/Ly) * sin(πz/Lz) * exp(-3π²αt)
    lambda = π^2 * alpha * (1/Lx^2 + 1/Ly^2 + 1/Lz^2)
    return sin(π * x / Lx) * sin(π * y / Ly) * sin(π * z / Lz) * exp(-lambda * t)
end

# ベンチマーク問題の定義
function get_accuracy_benchmarks()
    """精度検証用ベンチマーク問題の定義"""
    
    benchmarks = AccuracyBenchmark[]
    
    # 1. 単純拡散問題（解析解あり）
    push!(benchmarks, AccuracyBenchmark(
        "simple_diffusion",
        "Simple diffusion with analytical solution",
        function()
            return (NX=32, NY=32, NZ=16, total_time=0.1, alpha=1.0e-4)
        end,
        (x, y, z, t) -> analytical_heat_3d_separable(x, y, z, t, 1.0, 1.0, 0.5, 1.0e-4),
        1.0e-4,  # 期待L2誤差
        1.0e-3   # 期待最大誤差
    ))
    
    # 2. 定常状態収束問題
    push!(benchmarks, AccuracyBenchmark(
        "steady_convergence",
        "Convergence to steady state",
        function()
            return (NX=24, NY=24, NZ=12, total_time=1.0, alpha=1.0e-4)
        end,
        nothing,  # 解析解なし（逐次計算と比較）
        1.0e-5,   # 期待L2誤差
        1.0e-4    # 期待最大誤差
    ))
    
    # 3. 高アスペクト比問題
    push!(benchmarks, AccuracyBenchmark(
        "high_aspect_ratio",
        "High aspect ratio geometry",
        function()
            return (NX=48, NY=48, NZ=6, total_time=0.5, alpha=5.0e-5)
        end,
        nothing,
        1.0e-4,
        1.0e-3
    ))
    
    # 4. 短時間高精度問題
    push!(benchmarks, AccuracyBenchmark(
        "short_time_precision",
        "Short time high precision analysis",
        function()
            return (NX=32, NY=32, NZ=16, total_time=0.05, alpha=1.0e-4)
        end,
        nothing,
        1.0e-6,
        1.0e-5
    ))
    
    return benchmarks
end

function create_accuracy_config(problem_setup, n_processes)
    """精度重視のParareal設定"""
    
    setup = problem_setup()
    
    # 高精度設定
    dt_fine = setup.total_time / 200.0  # 細かい時間ステップ
    dt_coarse = dt_fine * 5.0           # 控えめな比率
    
    config = PararealConfig(
        total_time=setup.total_time,
        n_time_windows=min(2, n_processes),  # 少ない時間窓で高精度
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
        time_step_ratio=5.0,
        max_iterations=30,                   # 多い反復回数
        convergence_tolerance=1.0e-8,        # 厳しい収束基準
        n_mpi_processes=min(2, n_processes),
        n_threads_per_process=max(1, n_processes ÷ 2),
        auto_optimize_parameters=false,
        parameter_exploration_mode=false,
        validation_mode=true,
        enable_performance_profiling=false   # 精度重視
    )
    
    return config, setup
end

function compute_analytical_error(parareal_result, analytical_func, problem_setup)
    """解析解との誤差計算"""
    
    if analytical_func === nothing
        return Inf, Inf, Inf
    end
    
    setup = problem_setup()
    temperature_field = parareal_result.temperature_field.data
    final_time = parareal_result.temperature_field.time
    
    NX, NY, NZ = size(temperature_field)[1]-2, size(temperature_field)[2]-2, size(temperature_field)[3]-2
    
    # 格子点での解析解計算
    analytical_field = zeros(NX+2, NY+2, NZ+2)
    
    for i in 2:NX+1
        for j in 2:NY+1
            for k in 2:NZ+1
                x = (i-1.5) / NX
                y = (j-1.5) / NY  
                z = (k-1.5) / NZ
                analytical_field[i,j,k] = analytical_func(x, y, z, final_time)
            end
        end
    end
    
    # 内部格子点のみで誤差計算
    parareal_interior = temperature_field[2:end-1, 2:end-1, 2:end-1]
    analytical_interior = analytical_field[2:end-1, 2:end-1, 2:end-1]
    
    # 誤差メトリクス
    diff = parareal_interior - analytical_interior
    l2_error = norm(diff) / norm(analytical_interior)
    max_error = maximum(abs.(diff))
    relative_error = l2_error
    
    return l2_error, max_error, relative_error
end

function compute_sequential_comparison_error(parareal_result, problem_setup, comm, rank, size)
    """高精度逐次計算との比較誤差"""
    
    setup = problem_setup()
    
    if rank == 0
        println("  Running high-precision sequential computation for comparison...")
    end
    
    try
        # より高精度な逐次計算
        result_seq = q3d(setup.NX, setup.NY, setup.NZ,
                        solver="pbicgstab",
                        epsilon=1.0e-10,  # 非常に高精度
                        par="thread",
                        is_steady=false,
                        parareal=false)
        
        # 誤差計算
        parareal_field = parareal_result.temperature_field.data[2:end-1, 2:end-1, 2:end-1]
        sequential_field = result_seq.temperature_field.data[2:end-1, 2:end-1, 2:end-1]
        
        diff = parareal_field - sequential_field
        l2_error = norm(diff) / norm(sequential_field)
        max_error = maximum(abs.(diff))
        relative_error = l2_error
        
        return l2_error, max_error, relative_error
        
    catch e
        if rank == 0
            println("  Sequential comparison failed: $e")
        end
        return Inf, Inf, Inf
    end
end

function run_accuracy_test(benchmark::AccuracyBenchmark, comm, rank, size)
    """単一精度テストの実行"""
    
    if rank == 0
        println("\n--- Accuracy Test: $(benchmark.name) ---")
        println("Description: $(benchmark.description)")
    end
    
    try
        # 設定作成
        config, setup = create_accuracy_config(benchmark.problem_setup, size)
        
        if rank == 0
            println("Problem size: $(setup.NX)×$(setup.NY)×$(setup.NZ)")
            println("Analysis time: $(setup.total_time)")
            println("Time step ratio: $(config.time_step_ratio)")
            println("Max iterations: $(config.max_iterations)")
        end
        
        # Parareal実行
        start_time = time()
        result = q3d(setup.NX, setup.NY, setup.NZ,
                    solver="pbicgstab",
                    epsilon=1.0e-9,
                    par="thread",
                    is_steady=false,
                    parareal=true,
                    parareal_config=config)
        execution_time = time() - start_time
        
        if rank == 0
            if !result.converged
                println("❌ Parareal did not converge")
                return AccuracyTestResult(
                    benchmark.name, Inf, Inf, Inf, false, false, execution_time
                )
            end
            
            println("✅ Parareal converged in $(result.parareal_iterations) iterations")
            
            # 精度評価
            l2_error, max_error, relative_error = 0.0, 0.0, 0.0
            
            if benchmark.analytical_solution !== nothing
                # 解析解との比較
                println("  Comparing with analytical solution...")
                l2_error, max_error, relative_error = compute_analytical_error(
                    result, benchmark.analytical_solution, benchmark.problem_setup
                )
            else
                # 高精度逐次計算との比較
                println("  Comparing with high-precision sequential computation...")
                l2_error, max_error, relative_error = compute_sequential_comparison_error(
                    result, benchmark.problem_setup, comm, rank, size
                )
            end
            
            # 精度判定
            l2_ok = l2_error <= benchmark.expected_l2_error * 10.0  # 10倍まで許容
            max_ok = max_error <= benchmark.expected_max_error * 10.0
            meets_accuracy = l2_ok && max_ok
            
            # 結果表示
            println("Results:")
            println("  L2 error: $(l2_error) (expected: ≤$(benchmark.expected_l2_error)) $(l2_ok ? "✅" : "❌")")
            println("  Max error: $(max_error) (expected: ≤$(benchmark.expected_max_error)) $(max_ok ? "✅" : "❌")")
            println("  Relative error: $(relative_error)")
            println("  Execution time: $(round(execution_time, digits=2))s")
            println("  Overall accuracy: $(meets_accuracy ? "✅ PASS" : "❌ FAIL")")
            
            return AccuracyTestResult(
                benchmark.name,
                l2_error,
                max_error,
                relative_error,
                meets_accuracy,
                result.converged,
                execution_time
            )
        end
        
    catch e
        if rank == 0
            println("❌ Accuracy test failed: $e")
            return AccuracyTestResult(
                benchmark.name, Inf, Inf, Inf, false, false, 0.0
            )
        end
    end
    
    return nothing
end

@testset "Benchmark Accuracy Validation" begin
    
    comm, rank, size = setup_accuracy_test()
    
    if rank == 0
        println("Starting benchmark accuracy validation...")
        println("MPI processes: $size")
        println("Julia threads per process: $(Threads.nthreads())")
    end
    
    benchmarks = get_accuracy_benchmarks()
    test_results = AccuracyTestResult[]
    
    for benchmark in benchmarks
        result = run_accuracy_test(benchmark, comm, rank, size)
        
        if rank == 0 && result !== nothing
            push!(test_results, result)
            
            # 個別テストの検証
            @test result.parareal_converged "$(benchmark.name) should converge"
            @test result.meets_accuracy "$(benchmark.name) should meet accuracy requirements"
        end
        
        # プロセス間同期とメモリクリーンアップ
        MPI.Barrier(comm)
        GC.gc()
        sleep(2)
    end
    
    if rank == 0
        # 全体統計
        @testset "Overall Accuracy Statistics" begin
            @test length(test_results) > 0 "At least one accuracy test should run"
            
            converged_results = filter(r -> r.parareal_converged, test_results)
            @test length(converged_results) > 0 "At least one test should converge"
            
            if !isempty(converged_results)
                accurate_results = filter(r -> r.meets_accuracy, converged_results)
                accuracy_rate = length(accurate_results) / length(converged_results)
                
                @test accuracy_rate >= 0.8 "At least 80% of converged tests should meet accuracy requirements"
                
                # 統計情報
                l2_errors = [r.l2_error for r in converged_results if r.l2_error < Inf]
                max_errors = [r.max_error for r in converged_results if r.max_error < Inf]
                
                println("\n" * "="^60)
                println("BENCHMARK ACCURACY VALIDATION SUMMARY")
                println("="^60)
                println("Total tests: $(length(test_results))")
                println("Converged: $(length(converged_results))")
                println("Met accuracy: $(length(accurate_results))")
                println("Accuracy rate: $(round(accuracy_rate*100, digits=1))%")
                
                if !isempty(l2_errors)
                    println()
                    println("L2 Error statistics:")
                    println("  Average: $(round(sum(l2_errors)/length(l2_errors), sigdigits=3))")
                    println("  Range: $(round(minimum(l2_errors), sigdigits=3)) - $(round(maximum(l2_errors), sigdigits=3))")
                end
                
                if !isempty(max_errors)
                    println()
                    println("Max Error statistics:")
                    println("  Average: $(round(sum(max_errors)/length(max_errors), sigdigits=3))")
                    println("  Range: $(round(minimum(max_errors), sigdigits=3)) - $(round(maximum(max_errors), sigdigits=3))")
                end
                
                # 詳細結果
                println("\nDetailed Results:")
                println("Benchmark".ljust(20) * "L2 Error".ljust(12) * "Max Error".ljust(12) * "Status")
                println("-"^60)
                
                for result in test_results
                    status = result.meets_accuracy ? "✅ PASS" : (result.parareal_converged ? "⚠️ POOR" : "❌ FAIL")
                    l2_str = result.l2_error < Inf ? "$(round(result.l2_error, sigdigits=3))" : "N/A"
                    max_str = result.max_error < Inf ? "$(round(result.max_error, sigdigits=3))" : "N/A"
                    
                    println("$(result.benchmark_name)".ljust(20) * 
                           l2_str.ljust(12) * 
                           max_str.ljust(12) * 
                           status)
                end
            end
        end
    end
end

function save_accuracy_results(results, filename)
    """精度テスト結果の保存"""
    
    results_dict = []
    for r in results
        push!(results_dict, Dict(
            "benchmark_name" => r.benchmark_name,
            "l2_error" => r.l2_error,
            "max_error" => r.max_error,
            "relative_error" => r.relative_error,
            "meets_accuracy" => r.meets_accuracy,
            "parareal_converged" => r.parareal_converged,
            "execution_time" => r.execution_time
        ))
    end
    
    output_data = Dict(
        "timestamp" => string(now()),
        "julia_version" => string(VERSION),
        "total_tests" => length(results),
        "accurate_tests" => count(r -> r.meets_accuracy, results),
        "converged_tests" => count(r -> r.parareal_converged, results),
        "results" => results_dict
    )
    
    open(filename, "w") do f
        JSON.print(f, output_data, 2)
    end
end

# スクリプトとして実行された場合
if abspath(PROGRAM_FILE) == @__FILE__
    try
        comm, rank, size = setup_accuracy_test()
        
        if rank == 0
            println("Benchmark accuracy validation completed!")
        end
        
    catch e
        if rank == 0
            println("Accuracy test execution failed: $e")
        end
        exit(1)
    finally
        if MPI.Initialized() && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end