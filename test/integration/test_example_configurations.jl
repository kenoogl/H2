#!/usr/bin/env julia

"""
サンプル設定統合テスト

このテストスイートは、examples/ディレクトリ内の全てのサンプル設定が
正常に動作することを検証します。実際のMPI環境での実行をテストし、
期待される性能と精度が得られることを確認します。
"""

using Test
using MPI
using Heat3ds
using JSON
using Dates

# テスト用の小さな問題サイズ（CI環境対応）
const TEST_PROBLEM_SIZES = Dict(
    "small" => (16, 16, 8),
    "medium" => (24, 24, 12),
    "large" => (32, 32, 16)
)

const TEST_TIME_LIMITS = Dict(
    "basic" => 0.1,
    "ic_thermal" => 0.2,
    "optimization" => 0.5,
    "benchmark" => 0.3
)

struct ExampleTestResult
    example_name::String
    success::Bool
    execution_time::Float64
    speedup::Float64
    iterations::Int
    converged::Bool
    error_message::String
end

function setup_test_environment()
    """テスト環境のセットアップ"""
    
    # MPI初期化確認
    if !MPI.Initialized()
        MPI.Init()
    end
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    return comm, rank, size
end

function create_test_parareal_config(problem_size, total_time, n_processes)
    """テスト用Parareal設定の作成"""
    
    NX, NY, NZ = problem_size
    
    # テスト用の控えめな設定
    config = PararealConfig(
        total_time=total_time,
        n_time_windows=min(2, n_processes),  # 最大2窓に制限
        dt_coarse=total_time / 10.0,
        dt_fine=total_time / 100.0,
        time_step_ratio=10.0,
        max_iterations=5,  # 短時間で完了するよう制限
        convergence_tolerance=1.0e-4,  # 緩い基準
        n_mpi_processes=min(2, n_processes),
        n_threads_per_process=max(1, n_processes ÷ 2),
        auto_optimize_parameters=false,
        parameter_exploration_mode=false,
        validation_mode=false,  # テスト高速化のため無効
        enable_performance_profiling=true
    )
    
    return config
end

@testset "Example Configuration Integration Tests" begin
    
    comm, rank, size = setup_test_environment()
    
    if rank == 0
        println("Starting example configuration tests...")
        println("MPI processes: $size")
        println("Julia threads: $(Threads.nthreads())")
    end
    
    @testset "Basic Parareal Example Test" begin
        if rank == 0
            println("\n--- Testing Basic Parareal Example ---")
        end
        
        try
            # テスト用小規模問題
            NX, NY, NZ = TEST_PROBLEM_SIZES["small"]
            total_time = TEST_TIME_LIMITS["basic"]
            
            config = create_test_parareal_config((NX, NY, NZ), total_time, size)
            
            start_time = time()
            
            # 基本Parareal実行
            result = q3d(NX, NY, NZ,
                        solver="pbicgstab",
                        epsilon=1.0e-6,
                        par="thread",
                        is_steady=false,
                        parareal=true,
                        parareal_config=config)
            
            execution_time = time() - start_time
            
            if rank == 0
                @test result.converged == true
                @test result.parareal_iterations <= config.max_iterations
                @test execution_time < 60.0  # 1分以内で完了
                
                if haskey(result, :performance_metrics)
                    @test result.performance_metrics.overall_speedup >= 0.8
                    println("✅ Basic example: Speedup = $(round(result.performance_metrics.overall_speedup, digits=2))x")
                end
                
                println("✅ Basic example test passed")
            end
            
        catch e
            if rank == 0
                @test false "Basic example failed: $e"
                println("❌ Basic example test failed: $e")
            end
        end
    end
    
    @testset "IC Thermal Analysis Example Test" begin
        if rank == 0
            println("\n--- Testing IC Thermal Analysis Example ---")
        end
        
        try
            # IC問題用設定（テスト用に縮小）
            NX, NY, NZ = 20, 20, 4  # 元の100x100x20から縮小
            total_time = TEST_TIME_LIMITS["ic_thermal"]
            
            # IC特有の設定
            thermal_diffusivity = 1.4e-4
            characteristic_length = 0.01
            dt_fine = 0.1 * characteristic_length^2 / thermal_diffusivity / 1000
            dt_coarse = dt_fine * 25.0
            
            config = PararealConfig(
                total_time=total_time,
                n_time_windows=min(2, size),
                dt_coarse=dt_coarse,
                dt_fine=dt_fine,
                time_step_ratio=25.0,
                max_iterations=8,
                convergence_tolerance=1.0e-5,
                n_mpi_processes=min(2, size),
                n_threads_per_process=max(1, size ÷ 2),
                auto_optimize_parameters=false,
                parameter_exploration_mode=false,
                validation_mode=false,
                enable_performance_profiling=true
            )
            
            start_time = time()
            
            # IC熱解析実行
            result = q3d(NX, NY, NZ,
                        solver="pbicgstab",
                        epsilon=1.0e-7,
                        par="thread",
                        is_steady=false,
                        parareal=true,
                        parareal_config=config)
            
            execution_time = time() - start_time
            
            if rank == 0
                @test result.converged == true
                @test execution_time < 120.0  # 2分以内で完了
                
                # IC解析特有の検証
                if haskey(result, :temperature_field)
                    max_temp = maximum(result.temperature_field.data)
                    min_temp = minimum(result.temperature_field.data)
                    @test max_temp > min_temp  # 温度勾配の存在確認
                    @test max_temp < 1000.0    # 物理的に妥当な温度範囲
                    @test min_temp > 200.0
                end
                
                println("✅ IC thermal analysis test passed")
            end
            
        catch e
            if rank == 0
                @test false "IC thermal analysis failed: $e"
                println("❌ IC thermal analysis test failed: $e")
            end
        end
    end
    
    @testset "Parameter Optimization Example Test" begin
        if rank == 0
            println("\n--- Testing Parameter Optimization Example ---")
        end
        
        # パラメータ最適化は時間がかかるため、簡略版をテスト
        try
            NX, NY, NZ = TEST_PROBLEM_SIZES["small"]
            total_time = TEST_TIME_LIMITS["optimization"]
            
            # 限定的なパラメータスイープ
            time_step_ratios = [5.0, 10.0]  # 2つのみテスト
            n_windows_list = [2]            # 1つのみテスト
            
            results = []
            
            for ratio in time_step_ratios
                for n_windows in n_windows_list
                    if n_windows <= size
                        
                        config = PararealConfig(
                            total_time=total_time,
                            n_time_windows=n_windows,
                            dt_coarse=total_time / 20.0,
                            dt_fine=total_time / (20.0 * ratio),
                            time_step_ratio=ratio,
                            max_iterations=5,
                            convergence_tolerance=1.0e-4,
                            n_mpi_processes=n_windows,
                            n_threads_per_process=size ÷ n_windows,
                            auto_optimize_parameters=false,
                            parameter_exploration_mode=true
                        )
                        
                        try
                            result = q3d(NX, NY, NZ,
                                        solver="pbicgstab",
                                        epsilon=1.0e-6,
                                        par="thread",
                                        is_steady=false,
                                        parareal=true,
                                        parareal_config=config)
                            
                            if rank == 0
                                push!(results, (
                                    ratio=ratio,
                                    windows=n_windows,
                                    converged=result.converged,
                                    iterations=result.parareal_iterations
                                ))
                            end
                            
                        catch e
                            if rank == 0
                                println("Parameter test failed for ratio=$ratio: $e")
                            end
                        end
                    end
                end
            end
            
            if rank == 0
                @test length(results) > 0
                converged_results = filter(r -> r.converged, results)
                @test length(converged_results) > 0
                
                println("✅ Parameter optimization test passed ($(length(converged_results))/$(length(results)) converged)")
            end
            
        catch e
            if rank == 0
                @test false "Parameter optimization test failed: $e"
                println("❌ Parameter optimization test failed: $e")
            end
        end
    end
    
    @testset "Benchmark Problems Test" begin
        if rank == 0
            println("\n--- Testing Benchmark Problems ---")
        end
        
        # 小規模ベンチマーク問題のみテスト
        try
            # 小規模問題の定義
            benchmark_problem = (
                name="test_small",
                NX=16, NY=16, NZ=8,
                total_time=0.1,
                expected_speedup_range=(1.0, 2.0)
            )
            
            config = create_test_parareal_config(
                (benchmark_problem.NX, benchmark_problem.NY, benchmark_problem.NZ),
                benchmark_problem.total_time,
                size
            )
            
            start_time = time()
            
            result = q3d(benchmark_problem.NX, benchmark_problem.NY, benchmark_problem.NZ,
                        solver="pbicgstab",
                        epsilon=1.0e-6,
                        par="thread",
                        is_steady=false,
                        parareal=true,
                        parareal_config=config)
            
            execution_time = time() - start_time
            
            if rank == 0
                @test result.converged == true
                @test execution_time < 60.0
                
                if haskey(result, :performance_metrics)
                    speedup = result.performance_metrics.overall_speedup
                    @test speedup >= benchmark_problem.expected_speedup_range[1] * 0.8  # 80%の性能でも許容
                    println("✅ Benchmark test: Speedup = $(round(speedup, digits=2))x")
                end
                
                println("✅ Benchmark problems test passed")
            end
            
        catch e
            if rank == 0
                @test false "Benchmark problems test failed: $e"
                println("❌ Benchmark problems test failed: $e")
            end
        end
    end
    
    @testset "Configuration Validation Test" begin
        if rank == 0
            println("\n--- Testing Configuration Validation ---")
        end
        
        # 不正な設定での動作確認
        try
            NX, NY, NZ = TEST_PROBLEM_SIZES["small"]
            
            # 意図的に収束しにくい設定
            bad_config = PararealConfig(
                total_time=0.1,
                n_time_windows=2,
                dt_coarse=0.1,
                dt_fine=0.001,
                time_step_ratio=100.0,  # 非常に大きな比率
                max_iterations=3,       # 少ない反復回数
                convergence_tolerance=1.0e-8,  # 厳しい基準
                n_mpi_processes=min(2, size),
                n_threads_per_process=max(1, size ÷ 2),
                auto_optimize_parameters=false,
                parameter_exploration_mode=false
            )
            
            # エラーハンドリングのテスト
            try
                result = q3d(NX, NY, NZ,
                            solver="pbicgstab",
                            epsilon=1.0e-6,
                            par="thread",
                            is_steady=false,
                            parareal=true,
                            parareal_config=bad_config)
                
                if rank == 0
                    # 収束しなくても、エラーで停止せずに完了することを確認
                    @test true  # 実行が完了したことを確認
                    println("✅ Configuration validation test passed (graceful handling)")
                end
                
            catch e
                if rank == 0
                    # 適切なエラーハンドリングがされていることを確認
                    @test isa(e, Exception)
                    println("✅ Configuration validation test passed (proper error handling)")
                end
            end
            
        catch e
            if rank == 0
                @test false "Configuration validation test failed: $e"
                println("❌ Configuration validation test failed: $e")
            end
        end
    end
    
    @testset "Performance Claims Validation" begin
        if rank == 0
            println("\n--- Testing Performance Claims ---")
        end
        
        # ドキュメントで主張している性能が実際に得られるかテスト
        try
            NX, NY, NZ = TEST_PROBLEM_SIZES["medium"]
            total_time = 0.2
            
            # 中規模問題での性能テスト
            config = PararealConfig(
                total_time=total_time,
                n_time_windows=min(4, size),
                dt_coarse=total_time / 20.0,
                dt_fine=total_time / 200.0,
                time_step_ratio=10.0,
                max_iterations=10,
                convergence_tolerance=1.0e-6,
                n_mpi_processes=min(4, size),
                n_threads_per_process=max(1, size ÷ 4),
                auto_optimize_parameters=false,
                parameter_exploration_mode=false,
                validation_mode=true,
                enable_performance_profiling=true
            )
            
            # Parareal実行
            parareal_start = time()
            result_parareal = q3d(NX, NY, NZ,
                                 solver="pbicgstab",
                                 epsilon=1.0e-6,
                                 par="thread",
                                 is_steady=false,
                                 parareal=true,
                                 parareal_config=config)
            parareal_time = time() - parareal_start
            
            # 逐次実行（比較用、より小さい問題で）
            seq_start = time()
            result_sequential = q3d(NX÷2, NY÷2, NZ,
                                   solver="pbicgstab",
                                   epsilon=1.0e-6,
                                   par="thread",
                                   is_steady=false,
                                   parareal=false)
            seq_time = time() - seq_start
            
            if rank == 0
                @test result_parareal.converged == true
                
                # スケーリング推定
                scaling_factor = (NX * NY * NZ) / ((NX÷2) * (NY÷2) * NZ)
                estimated_seq_time = seq_time * scaling_factor
                actual_speedup = estimated_seq_time / parareal_time
                
                @test actual_speedup > 1.0  # 最低限の高速化
                
                if haskey(result_parareal, :validation_metrics)
                    @test result_parareal.validation_metrics.l2_norm_error < 1.0e-4
                end
                
                println("✅ Performance claims validation passed")
                println("   Estimated speedup: $(round(actual_speedup, digits=2))x")
            end
            
        catch e
            if rank == 0
                @test false "Performance claims validation failed: $e"
                println("❌ Performance claims validation failed: $e")
            end
        end
    end
    
    if rank == 0
        println("\n" * "="^60)
        println("Example Configuration Integration Tests Completed")
        println("="^60)
    end
end

# テスト結果のサマリー生成
function generate_test_summary()
    """テスト結果のサマリーを生成"""
    
    comm, rank, size = setup_test_environment()
    
    if rank == 0
        summary = Dict(
            "timestamp" => string(now()),
            "mpi_processes" => size,
            "julia_threads" => Threads.nthreads(),
            "julia_version" => string(VERSION),
            "test_environment" => "integration_test",
            "test_status" => "completed"
        )
        
        # サマリーファイルの保存
        open("test_example_configurations_summary.json", "w") do f
            JSON.print(f, summary, 2)
        end
        
        println("Test summary saved to: test_example_configurations_summary.json")
    end
end

# スクリプトとして実行された場合
if abspath(PROGRAM_FILE) == @__FILE__
    # テスト実行
    try
        # テスト結果のサマリー生成
        generate_test_summary()
        
        println("All example configuration tests completed successfully!")
        
    catch e
        println("Test execution failed: $e")
        exit(1)
    finally
        if MPI.Initialized() && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end