#!/usr/bin/env julia

"""
パラメータ最適化サンプル

このスクリプトは、Pararealアルゴリズムの最適パラメータを
自動的に探索する機能のデモンストレーションです。
異なる時間ステップ比率、時間窓数、プロセス数の組み合わせを
系統的にテストし、最適な設定を見つけます。

実行方法:
    mpirun -np 8 julia parameter_optimization_example.jl
"""

using MPI
using Heat3ds
using JSON
using Dates

struct OptimizationResult
    time_step_ratio::Float64
    n_time_windows::Int
    n_processes::Int
    speedup::Float64
    efficiency::Float64
    iterations::Int
    converged::Bool
    execution_time::Float64
    memory_usage::Float64
end

function create_test_problem()
    """テスト問題の設定"""
    return (
        NX = 48,
        NY = 48, 
        NZ = 24,
        total_time = 0.5,
        solver = "pbicgstab",
        epsilon = 1.0e-6
    )
end

function run_parameter_sweep(comm, rank, size)
    """パラメータスイープの実行"""
    
    # テスト対象パラメータ
    time_step_ratios = [5.0, 10.0, 25.0, 50.0, 100.0]
    time_window_counts = [2, 4, 8]
    
    results = OptimizationResult[]
    problem = create_test_problem()
    
    if rank == 0
        println("=== Parameter Optimization Sweep ===")
        println("Problem size: $(problem.NX)×$(problem.NY)×$(problem.NZ)")
        println("Total combinations: $(length(time_step_ratios) * length(time_window_counts))")
        println("Available processes: $size")
        println()
    end
    
    test_count = 0
    total_tests = length(time_step_ratios) * length(time_window_counts)
    
    for ratio in time_step_ratios
        for n_windows in time_window_counts
            
            # プロセス数が時間窓数以上の場合のみテスト
            if n_windows <= size
                test_count += 1
                
                if rank == 0
                    println("Test $test_count/$total_tests: ratio=$ratio, windows=$n_windows")
                end
                
                # 基本時間ステップの計算
                dt_fine = 0.001
                dt_coarse = dt_fine * ratio
                
                # Parareal設定
                config = PararealConfig(
                    total_time=problem.total_time,
                    n_time_windows=n_windows,
                    dt_coarse=dt_coarse,
                    dt_fine=dt_fine,
                    time_step_ratio=ratio,
                    max_iterations=20,
                    convergence_tolerance=1.0e-6,
                    n_mpi_processes=n_windows,
                    n_threads_per_process=size ÷ n_windows,
                    auto_optimize_parameters=false,
                    parameter_exploration_mode=true,
                    enable_performance_profiling=true
                )
                
                try
                    # メモリ使用量測定開始
                    initial_memory = Sys.total_memory() - Sys.free_memory()
                    start_time = time()
                    
                    # Parareal実行
                    result = q3d(problem.NX, problem.NY, problem.NZ,
                                solver=problem.solver,
                                epsilon=problem.epsilon,
                                par="thread",
                                is_steady=false,
                                parareal=true,
                                parareal_config=config)
                    
                    execution_time = time() - start_time
                    final_memory = Sys.total_memory() - Sys.free_memory()
                    memory_usage = (final_memory - initial_memory) / 1024^3  # GB
                    
                    # 結果の記録
                    if rank == 0
                        speedup = haskey(result, :performance_metrics) ? 
                                 result.performance_metrics.overall_speedup : 0.0
                        efficiency = speedup / n_windows
                        
                        opt_result = OptimizationResult(
                            ratio,
                            n_windows,
                            n_windows,  # 実際に使用したプロセス数
                            speedup,
                            efficiency,
                            result.parareal_iterations,
                            result.converged,
                            execution_time,
                            memory_usage
                        )
                        
                        push!(results, opt_result)
                        
                        # 結果の即座出力
                        status = result.converged ? "✓" : "✗"
                        println("  $status Speedup: $(round(speedup, digits=2))x, " *
                               "Efficiency: $(round(efficiency*100, digits=1))%, " *
                               "Iterations: $(result.parareal_iterations), " *
                               "Time: $(round(execution_time, digits=1))s")
                    end
                    
                    # ガベージコレクション
                    GC.gc()
                    
                catch e
                    if rank == 0
                        println("  ✗ Failed: $e")
                    end
                end
                
                # プロセス間同期
                MPI.Barrier(comm)
                
            else
                if rank == 0
                    println("Skipping: ratio=$ratio, windows=$n_windows (insufficient processes)")
                end
            end
        end
    end
    
    return results
end

function analyze_results(results)
    """結果の分析"""
    
    if isempty(results)
        println("No valid results to analyze")
        return
    end
    
    println("\n=== Optimization Results Analysis ===")
    
    # 収束した結果のみを分析
    converged_results = filter(r -> r.converged, results)
    
    if isempty(converged_results)
        println("No converged results found")
        return
    end
    
    println("Converged configurations: $(length(converged_results))/$(length(results))")
    
    # 最高性能の設定
    best_speedup = maximum(converged_results, by=r -> r.speedup)
    best_efficiency = maximum(converged_results, by=r -> r.efficiency)
    fastest_execution = minimum(converged_results, by=r -> r.execution_time)
    
    println("\n--- Best Speedup ---")
    println("Configuration: ratio=$(best_speedup.time_step_ratio), windows=$(best_speedup.n_time_windows)")
    println("Speedup: $(round(best_speedup.speedup, digits=2))x")
    println("Efficiency: $(round(best_speedup.efficiency*100, digits=1))%")
    println("Iterations: $(best_speedup.iterations)")
    println("Execution time: $(round(best_speedup.execution_time, digits=2))s")
    
    println("\n--- Best Efficiency ---")
    println("Configuration: ratio=$(best_efficiency.time_step_ratio), windows=$(best_efficiency.n_time_windows)")
    println("Speedup: $(round(best_efficiency.speedup, digits=2))x")
    println("Efficiency: $(round(best_efficiency.efficiency*100, digits=1))%")
    println("Iterations: $(best_efficiency.iterations)")
    
    println("\n--- Fastest Execution ---")
    println("Configuration: ratio=$(fastest_execution.time_step_ratio), windows=$(fastest_execution.n_time_windows)")
    println("Speedup: $(round(fastest_execution.speedup, digits=2))x")
    println("Execution time: $(round(fastest_execution.execution_time, digits=2))s")
    
    # 統計情報
    speedups = [r.speedup for r in converged_results]
    efficiencies = [r.efficiency for r in converged_results]
    iterations = [r.iterations for r in converged_results]
    
    println("\n--- Statistics ---")
    println("Average speedup: $(round(sum(speedups)/length(speedups), digits=2))x")
    println("Average efficiency: $(round(sum(efficiencies)/length(efficiencies)*100, digits=1))%")
    println("Average iterations: $(round(sum(iterations)/length(iterations), digits=1))")
    println("Speedup range: $(round(minimum(speedups), digits=2))x - $(round(maximum(speedups), digits=2))x")
    
    # 推奨設定の生成
    generate_recommendations(converged_results)
end

function generate_recommendations(results)
    """推奨設定の生成"""
    
    println("\n=== Recommendations ===")
    
    # 効率性重視（効率 > 70%）
    high_efficiency = filter(r -> r.efficiency > 0.7, results)
    if !isempty(high_efficiency)
        best_he = maximum(high_efficiency, by=r -> r.speedup)
        println("For high efficiency (>70%): ratio=$(best_he.time_step_ratio), windows=$(best_he.n_time_windows)")
    end
    
    # 速度重視（最高スピードアップ）
    best_speed = maximum(results, by=r -> r.speedup)
    println("For maximum speed: ratio=$(best_speed.time_step_ratio), windows=$(best_speed.n_time_windows)")
    
    # バランス重視（効率×スピードアップ）
    balanced_scores = [r.efficiency * r.speedup for r in results]
    best_balanced_idx = argmax(balanced_scores)
    best_balanced = results[best_balanced_idx]
    println("For balanced performance: ratio=$(best_balanced.time_step_ratio), windows=$(best_balanced.n_time_windows)")
    
    # 問題サイズ別推奨
    println("\n--- Problem Size Recommendations ---")
    println("Small problems (<100K points): Use ratio=10-25, windows=2-4")
    println("Medium problems (100K-1M points): Use ratio=25-50, windows=4-8") 
    println("Large problems (>1M points): Use ratio=50-100, windows=8+")
end

function save_results(results, filename)
    """結果をJSONファイルに保存"""
    
    # 結果を辞書形式に変換
    results_dict = []
    for r in results
        push!(results_dict, Dict(
            "time_step_ratio" => r.time_step_ratio,
            "n_time_windows" => r.n_time_windows,
            "n_processes" => r.n_processes,
            "speedup" => r.speedup,
            "efficiency" => r.efficiency,
            "iterations" => r.iterations,
            "converged" => r.converged,
            "execution_time" => r.execution_time,
            "memory_usage" => r.memory_usage
        ))
    end
    
    # メタデータの追加
    output_data = Dict(
        "timestamp" => string(now()),
        "julia_version" => string(VERSION),
        "total_tests" => length(results),
        "converged_tests" => count(r -> r.converged, results),
        "results" => results_dict
    )
    
    # ファイル保存
    open(filename, "w") do f
        JSON.print(f, output_data, 2)
    end
    
    println("Results saved to: $filename")
end

function main()
    # MPI初期化
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    if rank == 0
        println("=== Parareal Parameter Optimization ===")
        println("MPI processes: $size")
        println("Julia threads per process: $(Threads.nthreads())")
        println("Start time: $(now())")
        println()
    end
    
    try
        # パラメータスイープの実行
        results = run_parameter_sweep(comm, rank, size)
        
        if rank == 0
            # 結果の分析
            analyze_results(results)
            
            # 結果の保存
            timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
            filename = "parareal_optimization_results_$(size)proc_$timestamp.json"
            save_results(results, filename)
            
            # 設定ファイルの生成
            if !isempty(results)
                generate_config_files(results)
            end
            
            println("\n=== Optimization Complete ===")
            println("Total execution time: $(round(time(), digits=2)) seconds")
        end
        
    catch e
        if rank == 0
            println("Error during parameter optimization:")
            println(e)
        end
        MPI.Abort(comm, 1)
    end
    
    # MPI終了
    MPI.Finalize()
end

function generate_config_files(results)
    """最適設定ファイルの生成"""
    
    converged_results = filter(r -> r.converged, results)
    if isempty(converged_results)
        return
    end
    
    # 最高性能設定
    best_result = maximum(converged_results, by=r -> r.speedup)
    
    config_template = """
# Optimal Parareal Configuration
# Generated on $(now())
# Based on parameter optimization results

using Heat3ds

function create_optimal_parareal_config(total_time=1.0, n_processes=$(best_result.n_processes))
    return PararealConfig(
        total_time=total_time,
        n_time_windows=$(best_result.n_time_windows),
        dt_coarse=0.001 * $(best_result.time_step_ratio),
        dt_fine=0.001,
        time_step_ratio=$(best_result.time_step_ratio),
        max_iterations=20,
        convergence_tolerance=1.0e-6,
        n_mpi_processes=n_processes,
        n_threads_per_process=Threads.nthreads(),
        auto_optimize_parameters=false,
        parameter_exploration_mode=false
    )
end

# Expected performance:
# Speedup: $(round(best_result.speedup, digits=2))x
# Efficiency: $(round(best_result.efficiency*100, digits=1))%
# Iterations: $(best_result.iterations)
"""
    
    open("optimal_parareal_config.jl", "w") do f
        write(f, config_template)
    end
    
    println("Optimal configuration saved to: optimal_parareal_config.jl")
end

# スクリプトとして実行された場合のみmain()を呼び出し
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end