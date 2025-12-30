#!/usr/bin/env julia

"""
Pararealベンチマーク問題集

このスクリプトは、Pararealアルゴリズムの性能評価用の
標準的なベンチマーク問題を提供します。異なる問題サイズ、
物理特性、境界条件での性能を系統的に評価できます。

実行方法:
    mpirun -np 4 julia benchmark_problems.jl [problem_name]
    
利用可能な問題:
    - small: 小規模問題（32³格子）
    - medium: 中規模問題（64³格子）
    - large: 大規模問題（128³格子）
    - ic_package: ICパッケージ熱解析
    - all: 全ての問題を実行
"""

using MPI
using Heat3ds
using JSON
using Dates

# ベンチマーク問題の定義
struct BenchmarkProblem
    name::String
    description::String
    NX::Int
    NY::Int
    NZ::Int
    total_time::Float64
    thermal_diffusivity::Float64
    expected_speedup_range::Tuple{Float64, Float64}
    recommended_config::Dict{String, Any}
end

function get_benchmark_problems()
    """標準ベンチマーク問題の定義"""
    
    problems = BenchmarkProblem[]
    
    # 小規模問題
    push!(problems, BenchmarkProblem(
        "small",
        "Small-scale problem for algorithm validation",
        32, 32, 32,
        0.5,  # 0.5秒
        1.0e-4,  # 標準的な熱拡散率
        (1.2, 2.5),  # 期待スピードアップ範囲
        Dict(
            "time_step_ratio" => 10.0,
            "n_time_windows" => 2,
            "max_iterations" => 15,
            "convergence_tolerance" => 1.0e-6
        )
    ))
    
    # 中規模問題
    push!(problems, BenchmarkProblem(
        "medium",
        "Medium-scale problem for performance evaluation",
        64, 64, 32,
        1.0,  # 1秒
        1.0e-4,
        (2.0, 4.0),
        Dict(
            "time_step_ratio" => 25.0,
            "n_time_windows" => 4,
            "max_iterations" => 20,
            "convergence_tolerance" => 1.0e-6
        )
    ))
    
    # 大規模問題
    push!(problems, BenchmarkProblem(
        "large",
        "Large-scale problem for scalability testing",
        128, 64, 32,
        2.0,  # 2秒
        1.0e-4,
        (3.0, 8.0),
        Dict(
            "time_step_ratio" => 50.0,
            "n_time_windows" => 8,
            "max_iterations" => 25,
            "convergence_tolerance" => 1.0e-6
        )
    ))
    
    # ICパッケージ問題
    push!(problems, BenchmarkProblem(
        "ic_package",
        "IC package thermal analysis benchmark",
        100, 100, 20,
        5.0,  # 5秒（長時間解析）
        1.4e-4,  # シリコンの熱拡散率
        (4.0, 12.0),
        Dict(
            "time_step_ratio" => 75.0,
            "n_time_windows" => 8,
            "max_iterations" => 30,
            "convergence_tolerance" => 1.0e-7
        )
    ))
    
    # 高アスペクト比問題
    push!(problems, BenchmarkProblem(
        "high_aspect",
        "High aspect ratio problem (thin film)",
        128, 128, 8,
        1.0,
        5.0e-5,  # 薄膜材料
        (2.5, 6.0),
        Dict(
            "time_step_ratio" => 40.0,
            "n_time_windows" => 6,
            "max_iterations" => 25,
            "convergence_tolerance" => 1.0e-6
        )
    ))
    
    return problems
end

function create_benchmark_config(problem::BenchmarkProblem, n_processes::Int)
    """ベンチマーク問題用のParareal設定を作成"""
    
    # 基本時間ステップの計算
    characteristic_length = 1.0 / max(problem.NX, problem.NY, problem.NZ)
    dt_fine = 0.1 * characteristic_length^2 / problem.thermal_diffusivity
    dt_coarse = dt_fine * problem.recommended_config["time_step_ratio"]
    
    # 時間窓数の調整（プロセス数に応じて）
    n_windows = min(problem.recommended_config["n_time_windows"], n_processes)
    
    config = PararealConfig(
        total_time=problem.total_time,
        n_time_windows=n_windows,
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
        time_step_ratio=problem.recommended_config["time_step_ratio"],
        max_iterations=problem.recommended_config["max_iterations"],
        convergence_tolerance=problem.recommended_config["convergence_tolerance"],
        n_mpi_processes=n_windows,
        n_threads_per_process=n_processes ÷ n_windows,
        auto_optimize_parameters=false,
        parameter_exploration_mode=false,
        validation_mode=true,
        enable_performance_profiling=true
    )
    
    return config
end

struct BenchmarkResult
    problem_name::String
    n_processes::Int
    n_threads_per_process::Int
    execution_time::Float64
    parareal_iterations::Int
    converged::Bool
    speedup::Float64
    efficiency::Float64
    l2_error::Float64
    max_error::Float64
    memory_usage::Float64
    within_expected_range::Bool
end

function run_benchmark(problem::BenchmarkProblem, comm, rank, size)
    """単一ベンチマーク問題の実行"""
    
    if rank == 0
        println("\n" * "="^60)
        println("Running benchmark: $(problem.name)")
        println("Description: $(problem.description)")
        println("Problem size: $(problem.NX)×$(problem.NY)×$(problem.NZ) = $(problem.NX*problem.NY*problem.NZ) points")
        println("Analysis time: $(problem.total_time) seconds")
        println("Expected speedup: $(problem.expected_speedup_range[1])x - $(problem.expected_speedup_range[2])x")
        println("="^60)
    end
    
    # Parareal設定
    config = create_benchmark_config(problem, size)
    
    if rank == 0
        println("Configuration:")
        println("  Time windows: $(config.n_time_windows)")
        println("  Time step ratio: $(config.time_step_ratio)")
        println("  Fine time step: $(config.dt_fine)")
        println("  Coarse time step: $(config.dt_coarse)")
        println("  Max iterations: $(config.max_iterations)")
    end
    
    # メモリ使用量測定開始
    initial_memory = Sys.total_memory() - Sys.free_memory()
    
    # 実行時間測定
    start_time = time()
    
    try
        # Parareal実行
        result = q3d(problem.NX, problem.NY, problem.NZ,
                    solver="pbicgstab",
                    epsilon=1.0e-8,
                    par="thread",
                    is_steady=false,
                    parareal=true,
                    parareal_config=config)
        
        execution_time = time() - start_time
        final_memory = Sys.total_memory() - Sys.free_memory()
        memory_usage = (final_memory - initial_memory) / 1024^3  # GB
        
        if rank == 0
            # 結果の解析
            speedup = haskey(result, :performance_metrics) ? 
                     result.performance_metrics.overall_speedup : 0.0
            efficiency = speedup / config.n_time_windows
            
            l2_error = haskey(result, :validation_metrics) ? 
                      result.validation_metrics.l2_norm_error : 0.0
            max_error = haskey(result, :validation_metrics) ? 
                       result.validation_metrics.max_pointwise_error : 0.0
            
            # 期待範囲内かチェック
            within_range = (problem.expected_speedup_range[1] <= speedup <= problem.expected_speedup_range[2])
            
            benchmark_result = BenchmarkResult(
                problem.name,
                size,
                Threads.nthreads(),
                execution_time,
                result.parareal_iterations,
                result.converged,
                speedup,
                efficiency,
                l2_error,
                max_error,
                memory_usage,
                within_range
            )
            
            # 結果の表示
            print_benchmark_result(benchmark_result, problem)
            
            return benchmark_result
        end
        
    catch e
        if rank == 0
            println("❌ Benchmark failed: $e")
            
            # 失敗時のダミー結果
            return BenchmarkResult(
                problem.name, size, Threads.nthreads(),
                0.0, 0, false, 0.0, 0.0, Inf, Inf, 0.0, false
            )
        end
    end
    
    return nothing
end

function print_benchmark_result(result::BenchmarkResult, problem::BenchmarkProblem)
    """ベンチマーク結果の表示"""
    
    status = result.converged ? "✅" : "❌"
    range_status = result.within_expected_range ? "✅" : "⚠️"
    
    println("\n--- Results ---")
    println("$status Converged: $(result.converged)")
    println("   Iterations: $(result.iterations)")
    println("   Execution time: $(round(result.execution_time, digits=2)) seconds")
    println("   Speedup: $(round(result.speedup, digits=2))x $range_status")
    println("   Efficiency: $(round(result.efficiency*100, digits=1))%")
    println("   Memory usage: $(round(result.memory_usage, digits=2)) GB")
    
    if result.l2_error < Inf
        println("   L2 error: $(result.l2_error)")
        println("   Max error: $(result.max_error)")
    end
    
    # 性能評価
    if result.converged
        if result.within_expected_range
            println("🎉 Performance: EXCELLENT (within expected range)")
        elseif result.speedup > problem.expected_speedup_range[1] * 0.8
            println("👍 Performance: GOOD (close to expected range)")
        else
            println("⚠️  Performance: POOR (below expected range)")
        end
    else
        println("❌ Performance: FAILED (did not converge)")
    end
end

function run_all_benchmarks(problems, comm, rank, size)
    """全ベンチマーク問題の実行"""
    
    results = BenchmarkResult[]
    
    if rank == 0
        println("Starting comprehensive benchmark suite...")
        println("Total problems: $(length(problems))")
        println("MPI processes: $size")
        println("Threads per process: $(Threads.nthreads())")
        println("Total cores: $(size * Threads.nthreads())")
    end
    
    for (i, problem) in enumerate(problems)
        if rank == 0
            println("\nProgress: $i/$(length(problems))")
        end
        
        result = run_benchmark(problem, comm, rank, size)
        if rank == 0 && result !== nothing
            push!(results, result)
        end
        
        # プロセス間同期とメモリクリーンアップ
        MPI.Barrier(comm)
        GC.gc()
        
        # 問題間の休憩
        sleep(1)
    end
    
    return results
end

function generate_benchmark_report(results)
    """ベンチマークレポートの生成"""
    
    if isempty(results)
        println("No results to report")
        return
    end
    
    println("\n" * "="^80)
    println("BENCHMARK SUMMARY REPORT")
    println("="^80)
    
    # 全体統計
    total_tests = length(results)
    converged_tests = count(r -> r.converged, results)
    within_range_tests = count(r -> r.within_expected_range, results)
    
    println("Total tests: $total_tests")
    println("Converged: $converged_tests ($((converged_tests/total_tests*100) |> x -> round(x, digits=1))%)")
    println("Within expected range: $within_range_tests ($((within_range_tests/total_tests*100) |> x -> round(x, digits=1))%)")
    
    # 収束した結果のみで統計
    converged_results = filter(r -> r.converged, results)
    
    if !isempty(converged_results)
        speedups = [r.speedup for r in converged_results]
        efficiencies = [r.efficiency for r in converged_results]
        
        println("\n--- Performance Statistics (Converged Tests Only) ---")
        println("Average speedup: $(round(sum(speedups)/length(speedups), digits=2))x")
        println("Best speedup: $(round(maximum(speedups), digits=2))x")
        println("Worst speedup: $(round(minimum(speedups), digits=2))x")
        println("Average efficiency: $(round(sum(efficiencies)/length(efficiencies)*100, digits=1))%")
        println("Best efficiency: $(round(maximum(efficiencies)*100, digits=1))%")
    end
    
    # 問題別詳細結果
    println("\n--- Detailed Results ---")
    println("Problem".ljust(15) * "Size".ljust(12) * "Speedup".ljust(10) * "Efficiency".ljust(12) * "Status")
    println("-"^60)
    
    for result in results
        size_str = "$(result.n_processes)×$(result.n_threads_per_process)"
        speedup_str = result.converged ? "$(round(result.speedup, digits=2))x" : "N/A"
        efficiency_str = result.converged ? "$(round(result.efficiency*100, digits=1))%" : "N/A"
        status_str = result.converged ? (result.within_expected_range ? "✅ PASS" : "⚠️ SLOW") : "❌ FAIL"
        
        println(result.problem_name.ljust(15) * 
               size_str.ljust(12) * 
               speedup_str.ljust(10) * 
               efficiency_str.ljust(12) * 
               status_str)
    end
    
    # 推奨事項
    println("\n--- Recommendations ---")
    
    best_problems = filter(r -> r.converged && r.within_expected_range, results)
    if !isempty(best_problems)
        best_overall = maximum(best_problems, by=r -> r.speedup)
        println("Best performing problem: $(best_overall.problem_name)")
        println("  Speedup: $(round(best_overall.speedup, digits=2))x")
        println("  Efficiency: $(round(best_overall.efficiency*100, digits=1))%")
    end
    
    poor_problems = filter(r -> r.converged && !r.within_expected_range, results)
    if !isempty(poor_problems)
        println("\nProblems needing optimization:")
        for p in poor_problems
            println("  $(p.problem_name): Consider adjusting time step ratio or window count")
        end
    end
    
    failed_problems = filter(r -> !r.converged, results)
    if !isempty(failed_problems)
        println("\nFailed problems:")
        for p in failed_problems
            println("  $(p.problem_name): Check convergence parameters or problem setup")
        end
    end
end

function save_benchmark_results(results, filename)
    """ベンチマーク結果の保存"""
    
    # 結果を辞書形式に変換
    results_dict = []
    for r in results
        push!(results_dict, Dict(
            "problem_name" => r.problem_name,
            "n_processes" => r.n_processes,
            "n_threads_per_process" => r.n_threads_per_process,
            "execution_time" => r.execution_time,
            "parareal_iterations" => r.parareal_iterations,
            "converged" => r.converged,
            "speedup" => r.speedup,
            "efficiency" => r.efficiency,
            "l2_error" => r.l2_error,
            "max_error" => r.max_error,
            "memory_usage" => r.memory_usage,
            "within_expected_range" => r.within_expected_range
        ))
    end
    
    # メタデータ
    output_data = Dict(
        "timestamp" => string(now()),
        "julia_version" => string(VERSION),
        "total_tests" => length(results),
        "converged_tests" => count(r -> r.converged, results),
        "system_info" => Dict(
            "total_memory" => Sys.total_memory(),
            "cpu_threads" => Sys.CPU_THREADS
        ),
        "results" => results_dict
    )
    
    # ファイル保存
    open(filename, "w") do f
        JSON.print(f, output_data, 2)
    end
    
    println("Benchmark results saved to: $filename")
end

function main()
    # MPI初期化
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    # コマンドライン引数の処理
    problem_name = length(ARGS) > 0 ? ARGS[1] : "all"
    
    if rank == 0
        println("=== Parareal Benchmark Suite ===")
        println("Target problem: $problem_name")
        println("MPI processes: $size")
        println("Julia threads per process: $(Threads.nthreads())")
        println("Start time: $(now())")
    end
    
    try
        # ベンチマーク問題の取得
        all_problems = get_benchmark_problems()
        
        # 実行する問題の選択
        if problem_name == "all"
            problems = all_problems
        else
            problems = filter(p -> p.name == problem_name, all_problems)
            if isempty(problems)
                if rank == 0
                    println("Error: Unknown problem '$problem_name'")
                    println("Available problems: $(join([p.name for p in all_problems], ", "))")
                end
                MPI.Abort(comm, 1)
            end
        end
        
        # ベンチマーク実行
        results = run_all_benchmarks(problems, comm, rank, size)
        
        if rank == 0
            # 結果の分析とレポート生成
            generate_benchmark_report(results)
            
            # 結果の保存
            timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
            filename = "benchmark_results_$(size)proc_$timestamp.json"
            save_benchmark_results(results, filename)
            
            println("\n=== Benchmark Complete ===")
            println("Total execution time: $(round(time(), digits=2)) seconds")
        end
        
    catch e
        if rank == 0
            println("Error during benchmark execution:")
            println(e)
        end
        MPI.Abort(comm, 1)
    end
    
    # MPI終了
    MPI.Finalize()
end

# スクリプトとして実行された場合のみmain()を呼び出し
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end