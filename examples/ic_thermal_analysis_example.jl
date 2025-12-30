#!/usr/bin/env julia

"""
IC熱解析用Pararealサンプル

このスクリプトは、集積回路（IC）の熱解析に特化した
Parareal設定例を示します。実際のICパッケージの
熱特性を模擬した問題設定を使用します。

実行方法:
    mpirun -np 8 julia ic_thermal_analysis_example.jl
"""

using MPI
using Heat3ds

function setup_ic_thermal_problem()
    """IC熱解析問題の設定"""
    
    # ICパッケージの典型的なサイズ（mm単位）
    package_length = 10.0  # mm
    package_width = 10.0   # mm  
    package_height = 2.0   # mm
    
    # 格子設定（高解像度）
    NX = 100  # X方向格子点数
    NY = 100  # Y方向格子点数
    NZ = 20   # Z方向格子点数
    
    # 時間設定（過渡熱解析）
    total_time = 10.0  # 秒（起動から定常状態まで）
    
    return NX, NY, NZ, total_time, package_length, package_width, package_height
end

function create_ic_parareal_config(n_processes, total_time)
    """IC熱解析用Parareal設定"""
    
    # IC熱解析の特性に基づく設定
    # - 熱拡散率: シリコン ~1.4e-4 m²/s
    # - 特性時間: パッケージサイズ²/熱拡散率
    thermal_diffusivity = 1.4e-4  # m²/s
    characteristic_length = 0.01   # m (10mm)
    characteristic_time = characteristic_length^2 / thermal_diffusivity  # ~0.7秒
    
    # 時間ステップ設定（安定性を考慮）
    dt_fine = characteristic_time / 1000.0    # ~7e-4秒
    dt_coarse = dt_fine * 50.0                # 粗解法は50倍大きく
    
    config = PararealConfig(
        # 時間設定
        total_time=total_time,
        n_time_windows=n_processes,
        
        # 時間ステップ設定（IC熱解析最適化）
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
        time_step_ratio=50.0,  # IC熱解析では50程度が最適
        
        # 収束設定（高精度要求）
        max_iterations=25,
        convergence_tolerance=1.0e-7,  # IC解析では高精度が必要
        
        # 並列化設定
        n_mpi_processes=n_processes,
        n_threads_per_process=Threads.nthreads(),
        
        # IC解析特有の設定
        auto_optimize_parameters=true,
        parameter_exploration_mode=false,
        
        # 検証設定
        validation_mode=true,  # 精度確認を有効化
        
        # 性能監視
        enable_performance_profiling=true
    )
    
    return config
end

function main()
    # MPI初期化
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    if rank == 0
        println("=== IC Thermal Analysis with Parareal ===")
        println("MPI processes: $size")
        println("Julia threads per process: $(Threads.nthreads())")
        println("Total computational cores: $(size * Threads.nthreads())")
    end
    
    try
        # IC熱解析問題の設定
        NX, NY, NZ, total_time, pkg_l, pkg_w, pkg_h = setup_ic_thermal_problem()
        
        if rank == 0
            println("\n=== Problem Setup ===")
            println("IC package dimensions: $(pkg_l)×$(pkg_w)×$(pkg_h) mm")
            println("Grid resolution: $NX×$NY×$NZ = $(NX*NY*NZ) points")
            println("Analysis time: $total_time seconds")
            println("Grid spacing: $(pkg_l/NX) × $(pkg_w/NY) × $(pkg_h/NZ) mm")
        end
        
        # IC用Parareal設定
        config = create_ic_parareal_config(size, total_time)
        
        if rank == 0
            println("\n=== Parareal Configuration ===")
            println("Time windows: $(config.n_time_windows)")
            println("Fine time step: $(config.dt_fine) s")
            println("Coarse time step: $(config.dt_coarse) s")
            println("Time step ratio: $(config.time_step_ratio)")
            println("Max iterations: $(config.max_iterations)")
            println("Convergence tolerance: $(config.convergence_tolerance)")
        end
        
        # 実行時間測定
        start_time = time()
        
        # Parareal実行（IC熱解析）
        result = q3d(NX, NY, NZ,
                    solver="pbicgstab",  # IC解析では高精度ソルバーを使用
                    smoother="",
                    epsilon=1.0e-8,      # 高精度設定
                    par="thread",
                    is_steady=false,     # 過渡解析
                    parareal=true,
                    parareal_config=config)
        
        execution_time = time() - start_time
        
        if rank == 0
            println("\n=== Computation Results ===")
            println("Parareal computation completed!")
            println("Total execution time: $(round(execution_time, digits=2)) seconds")
            println("Parareal iterations: $(result.parareal_iterations)")
            println("Converged: $(result.converged)")
            
            # 性能メトリクス
            if haskey(result, :performance_metrics)
                pm = result.performance_metrics
                println("\n=== Performance Metrics ===")
                println("Overall speedup: $(round(pm.overall_speedup, digits=2))x")
                println("MPI efficiency: $(round(pm.mpi_efficiency * 100, digits=1))%")
                println("Threading efficiency: $(round(pm.threading_efficiency * 100, digits=1))%")
                println("Communication overhead: $(round(sum(pm.communication_time), digits=2)) s")
                println("Load balance factor: $(round(pm.load_balance_factor, digits=3))")
            end
            
            # 精度メトリクス
            if haskey(result, :validation_metrics)
                vm = result.validation_metrics
                println("\n=== Accuracy Validation ===")
                println("L2 norm error: $(vm.l2_norm_error)")
                println("Max pointwise error: $(vm.max_pointwise_error)")
                println("Relative error: $(vm.relative_error)")
                println("Within tolerance: $(vm.is_within_tolerance)")
            end
            
            # 最適化結果
            if haskey(result, :optimal_parameters)
                op = result.optimal_parameters
                println("\n=== Optimization Results ===")
                println("Optimal time step ratio: $(op.time_step_ratio)")
                println("Recommended time windows: $(op.n_time_windows)")
                println("Expected speedup: $(round(op.expected_speedup, digits=2))x")
            end
            
            # 出力ファイル
            println("\n=== Output Files ===")
            println("Temperature field: ic_temperature_$(size)proc.dat")
            println("Convergence history: ic_convergence_$(size)proc.csv")
            println("Performance report: ic_performance_$(size)proc.json")
            
            # IC解析特有の結果
            println("\n=== IC Thermal Analysis Results ===")
            if haskey(result, :temperature_field)
                max_temp = maximum(result.temperature_field.data)
                min_temp = minimum(result.temperature_field.data)
                println("Maximum temperature: $(round(max_temp, digits=2)) K")
                println("Minimum temperature: $(round(min_temp, digits=2)) K")
                println("Temperature range: $(round(max_temp - min_temp, digits=2)) K")
                
                # ホットスポット検出（簡易版）
                hotspot_threshold = min_temp + 0.8 * (max_temp - min_temp)
                hotspot_count = count(x -> x > hotspot_threshold, result.temperature_field.data)
                hotspot_percentage = hotspot_count / length(result.temperature_field.data) * 100
                println("Hotspot regions (>$(round(hotspot_threshold, digits=1))K): $(round(hotspot_percentage, digits=1))%")
            end
        end
        
        # 比較用逐次計算（小規模問題で）
        if rank == 0 && size >= 4
            println("\n=== Sequential Comparison (Reduced Problem) ===")
            try
                # より小さい問題で逐次計算
                seq_start = time()
                result_seq = q3d(NX÷2, NY÷2, NZ,
                               solver="pbicgstab",
                               epsilon=1.0e-8,
                               par="thread",
                               is_steady=false,
                               parareal=false)
                seq_time = time() - seq_start
                
                println("Sequential time (half resolution): $(round(seq_time, digits=2)) s")
                
                # スケーリング推定
                scaling_factor = (NX * NY * NZ) / ((NX÷2) * (NY÷2) * NZ)
                estimated_seq_time = seq_time * scaling_factor
                estimated_speedup = estimated_seq_time / execution_time
                
                println("Estimated sequential time (full resolution): $(round(estimated_seq_time, digits=2)) s")
                println("Estimated speedup: $(round(estimated_speedup, digits=2))x")
                
            catch e
                println("Sequential comparison failed: $e")
            end
        end
        
    catch e
        if rank == 0
            println("Error occurred during IC thermal analysis:")
            println(e)
            
            # デバッグ情報の出力
            println("\n=== Debug Information ===")
            println("Julia version: $(VERSION)")
            println("MPI processes: $size")
            println("Threads per process: $(Threads.nthreads())")
            
            # システム情報
            println("Available memory: $(round(Sys.total_memory() / 1024^3, digits=2)) GB")
            println("Free memory: $(round(Sys.free_memory() / 1024^3, digits=2)) GB")
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