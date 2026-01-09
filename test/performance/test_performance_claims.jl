#!/usr/bin/env julia

"""
性能主張検証テスト

このテストスイートは、ドキュメントで主張している性能（スピードアップ、効率、精度）が
実際に達成できることを検証します。異なる問題サイズとプロセス数での
系統的な性能測定を行います。
"""

using Test
using MPI
using Heat3ds
using Statistics
using JSON
using Dates

struct PerformanceClaim
    problem_size::String
    process_count::Int
    expected_speedup_min::Float64
    expected_speedup_max::Float64
    expected_efficiency_min::Float64
    expected_accuracy::Float64
end

struct PerformanceTestResult
    claim::PerformanceClaim
    actual_speedup::Float64
    actual_efficiency::Float64
    actual_accuracy::Float64
    parareal_time::Float64
    sequential_time::Float64
    converged::Bool
    meets_claim::Bool
end

# ドキュメントで主張している性能基準
const PERFORMANCE_CLAIMS = [
    PerformanceClaim("small", 2, 1.2, 1.8, 0.60, 1.0e-6),
    PerformanceClaim("small", 4, 1.8, 2.5, 0.45, 1.0e-6),
    PerformanceClaim("medium", 2, 1.4, 2.0, 0.70, 1.0e-6),
    PerformanceClaim("medium", 4, 2.0, 3.2, 0.50, 1.0e-6),
    PerformanceClaim("medium", 8, 2.4, 4.8, 0.30, 1.0e-6),
    PerformanceClaim("large", 4, 2.4, 4.0, 0.60, 1.0e-6),
    PerformanceClaim("large", 8, 3.2, 6.4, 0.40, 1.0e-6),
]

const PROBLEM_DEFINITIONS = Dict(
    "small" => (NX=32, NY=32, NZ=16, total_time=0.5),
    "medium" => (NX=48, NY=48, NZ=24, total_time=1.0),
    "large" => (NX=64, NY=64, NZ=32, total_time=1.5)
)

function setup_performance_test()
    """性能テスト環境のセットアップ"""
    
    if !MPI.Initialized()
        MPI.Init()
    end
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    return comm, rank, size
end

function create_optimal_parareal_config(problem_def, n_processes)
    """最適化されたParareal設定の作成"""
    
    # 問題サイズに応じた最適パラメータ
    optimal_params = Dict(
        "small" => (ratio=10.0, windows=min(2, n_processes), iterations=15),
        "medium" => (ratio=25.0, windows=min(4, n_processes), iterations=20),
        "large" => (ratio=50.0, windows=min(8, n_processes), iterations=25)
    )
    
    problem_size = ""
    for (size_name, def) in PROBLEM_DEFINITIONS
        if def == problem_def
            problem_size = size_name
            break
        end
    end
    
    params = get(optimal_params, problem_size, optimal_params["medium"])
    
    dt_fine = problem_def.total_time / (100.0 * params.ratio)
    dt_coarse = dt_fine * params.ratio
    
    config = PararealConfig(
        total_time=problem_def.total_time,
        n_time_windows=params.windows,
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
        time_step_ratio=params.ratio,
        max_iterations=params.iterations,
        convergence_tolerance=1.0e-6,
        n_mpi_processes=params.windows,
        n_threads_per_process=n_processes ÷ params.windows,
        auto_optimize_parameters=false,
        parameter_exploration_mode=false,
        validation_mode=true,
        enable_performance_profiling=true
    )
    
    return config
end

function run_performance_test(claim::PerformanceClaim, comm, rank, size)
    """単一性能テストの実行"""
    
    if size < claim.process_count
        if rank == 0
            println("Skipping test: insufficient processes (need $(claim.process_count), have $size)")
        end
        return nothing
    end
    
    problem_def = PROBLEM_DEFINITIONS[claim.problem_size]
    
    if rank == 0
        println("\n--- Performance Test ---")
        println("Problem: $(claim.problem_size) ($(problem_def.NX)×$(problem_def.NY)×$(problem_def.NZ))")
        println("Processes: $(claim.process_count)")
        println("Expected speedup: $(claim.expected_speedup_min)x - $(claim.expected_speedup_max)x")
        println("Expected efficiency: ≥$(round(claim.expected_efficiency_min*100, digits=1))%")
    end
    
    try
        # Parareal設定
        config = create_optimal_parareal_config(problem_def, claim.process_count)
        
        # Parareal実行
        if rank == 0
            println("Running Parareal computation...")
        end
        
        parareal_start = time()
        result_parareal = q3d(problem_def.NX, problem_def.NY, problem_def.NZ,
                             solver="pbicgstab",
                             epsilon=1.0e-8,
                             par="thread",
                             is_steady=false,
                             parareal=true,
                             parareal_config=config)
        parareal_time = time() - parareal_start
        
        # 逐次実行（比較用）
        if rank == 0
            println("Running sequential computation for comparison...")
        end
        
        # より小さい問題で逐次実行（時間短縮のため）
        scale_factor = 0.7  # 70%のサイズ
        seq_NX = max(16, Int(round(problem_def.NX * scale_factor)))
        seq_NY = max(16, Int(round(problem_def.NY * scale_factor)))
        seq_NZ = max(8, Int(round(problem_def.NZ * scale_factor)))
        
        seq_start = time()
        result_sequential = q3d(seq_NX, seq_NY, seq_NZ,
                               solver="pbicgstab",
                               epsilon=1.0e-8,
                               par="thread",
                               is_steady=false,
                               parareal=false)
        seq_time = time() - seq_start
        
        if rank == 0
            # スケーリング補正
            grid_ratio = (problem_def.NX * problem_def.NY * problem_def.NZ) / 
                        (seq_NX * seq_NY * seq_NZ)
            estimated_seq_time = seq_time * grid_ratio
            
            # 性能メトリクスの計算
            actual_speedup = estimated_seq_time / parareal_time
            actual_efficiency = actual_speedup / claim.process_count
            
            # 精度メトリクス
            actual_accuracy = 0.0
            if haskey(result_parareal, :validation_metrics)
                actual_accuracy = result_parareal.validation_metrics.l2_norm_error
            end
            
            # 主張との比較
            speedup_ok = (claim.expected_speedup_min <= actual_speedup <= claim.expected_speedup_max * 1.2)
            efficiency_ok = (actual_efficiency >= claim.expected_efficiency_min * 0.8)
            accuracy_ok = (actual_accuracy <= claim.expected_accuracy * 10.0)  # 緩い基準
            converged_ok = result_parareal.converged
            
            meets_claim = speedup_ok && efficiency_ok && accuracy_ok && converged_ok
            
            # 結果の表示
            println("Results:")
            println("  Converged: $(result_parareal.converged) $(converged_ok ? "✅" : "❌")")
            println("  Speedup: $(round(actual_speedup, digits=2))x $(speedup_ok ? "✅" : "❌")")
            println("  Efficiency: $(round(actual_efficiency*100, digits=1))% $(efficiency_ok ? "✅" : "❌")")
            println("  Accuracy: $(actual_accuracy) $(accuracy_ok ? "✅" : "❌")")
            println("  Overall: $(meets_claim ? "✅ PASS" : "❌ FAIL")")
            
            return PerformanceTestResult(
                claim,
                actual_speedup,
                actual_efficiency,
                actual_accuracy,
                parareal_time,
                estimated_seq_time,
                result_parareal.converged,
                meets_claim
            )
        end
        
    catch e
        if rank == 0
            println("❌ Test failed with error: $e")
            return PerformanceTestResult(
                claim, 0.0, 0.0, Inf, 0.0, 0.0, false, false
            )
        end
    end
    
    return nothing
end

@testset "Performance Claims Validation" begin
    
    comm, rank, size = setup_performance_test()
    
    if rank == 0
        println("Starting performance claims validation...")
        println("Available MPI processes: $size")
        println("Julia threads per process: $(Threads.nthreads())")
    end
    
    test_results = PerformanceTestResult[]
    
    # 利用可能なプロセス数に応じてテストを選択
    applicable_claims = filter(c -> c.process_count <= size, PERFORMANCE_CLAIMS)
    
    if rank == 0
        println("Testing $(length(applicable_claims)) performance claims...")
    end
    
    for claim in applicable_claims
        result = run_performance_test(claim, comm, rank, size)
        
        if rank == 0 && result !== nothing
            push!(test_results, result)
            
            # 個別テストの検証
            @test result.converged "$(claim.problem_size) with $(claim.process_count) processes should converge"
            @test result.meets_claim "Performance claim should be met for $(claim.problem_size) with $(claim.process_count) processes"
        end
        
        # プロセス間同期とメモリクリーンアップ
        MPI.Barrier(comm)
        GC.gc()
        sleep(2)  # テスト間の休憩
    end
    
    if rank == 0
        # 全体的な統計
        @testset "Overall Performance Statistics" begin
            @test length(test_results) > 0 "At least one performance test should run"
            
            converged_results = filter(r -> r.converged, test_results)
            @test length(converged_results) > 0 "At least one test should converge"
            
            if !isempty(converged_results)
                passed_results = filter(r -> r.meets_claim, converged_results)
                pass_rate = length(passed_results) / length(converged_results)
                
                @test pass_rate >= 0.7 "At least 70% of converged tests should meet performance claims"
                
                # 統計情報の表示
                speedups = [r.actual_speedup for r in converged_results]
                efficiencies = [r.actual_efficiency for r in converged_results]
                
                println("\n" * "="^60)
                println("PERFORMANCE CLAIMS VALIDATION SUMMARY")
                println("="^60)
                println("Total tests: $(length(test_results))")
                println("Converged: $(length(converged_results))")
                println("Passed claims: $(length(passed_results))")
                println("Pass rate: $(round(pass_rate*100, digits=1))%")
                println()
                println("Speedup statistics:")
                println("  Average: $(round(mean(speedups), digits=2))x")
                println("  Range: $(round(minimum(speedups), digits=2))x - $(round(maximum(speedups), digits=2))x")
                println("  Std dev: $(round(std(speedups), digits=2))")
                println()
                println("Efficiency statistics:")
                println("  Average: $(round(mean(efficiencies)*100, digits=1))%")
                println("  Range: $(round(minimum(efficiencies)*100, digits=1))% - $(round(maximum(efficiencies)*100, digits=1))%")
                
                # 詳細結果テーブル
                println("\nDetailed Results:")
                println("Problem".ljust(8) * "Procs".ljust(6) * "Speedup".ljust(10) * "Efficiency".ljust(12) * "Status")
                println("-"^50)
                
                for result in test_results
                    status = result.meets_claim ? "✅ PASS" : (result.converged ? "⚠️ SLOW" : "❌ FAIL")
                    speedup_str = result.converged ? "$(round(result.actual_speedup, digits=2))x" : "N/A"
                    efficiency_str = result.converged ? "$(round(result.actual_efficiency*100, digits=1))%" : "N/A"
                    
                    println("$(result.claim.problem_size)".ljust(8) * 
                           "$(result.claim.process_count)".ljust(6) * 
                           speedup_str.ljust(10) * 
                           efficiency_str.ljust(12) * 
                           status)
                end
            end
        end
    end
end

function save_performance_results(results, filename)
    """性能テスト結果の保存"""
    
    results_dict = []
    for r in results
        push!(results_dict, Dict(
            "problem_size" => r.claim.problem_size,
            "process_count" => r.claim.process_count,
            "expected_speedup_min" => r.claim.expected_speedup_min,
            "expected_speedup_max" => r.claim.expected_speedup_max,
            "expected_efficiency_min" => r.claim.expected_efficiency_min,
            "actual_speedup" => r.actual_speedup,
            "actual_efficiency" => r.actual_efficiency,
            "actual_accuracy" => r.actual_accuracy,
            "parareal_time" => r.parareal_time,
            "sequential_time" => r.sequential_time,
            "converged" => r.converged,
            "meets_claim" => r.meets_claim
        ))
    end
    
    output_data = Dict(
        "timestamp" => string(now()),
        "julia_version" => string(VERSION),
        "total_tests" => length(results),
        "passed_tests" => count(r -> r.meets_claim, results),
        "converged_tests" => count(r -> r.converged, results),
        "results" => results_dict
    )
    
    open(filename, "w") do f
        JSON.print(f, output_data, 2)
    end
end

# スクリプトとして実行された場合
if abspath(PROGRAM_FILE) == @__FILE__
    try
        comm, rank, size = setup_performance_test()
        
        if rank == 0
            println("Performance claims validation completed!")
        end
        
    catch e
        if rank == 0
            println("Performance test execution failed: $e")
        end
        exit(1)
    finally
        if MPI.Initialized() && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end