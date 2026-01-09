#!/usr/bin/env julia

"""
基本的なParareal実行例

このスクリプトは、Heat3dsでParareal時間並列化を使用する
最も基本的な例を示します。

実行方法:
    mpirun -np 4 julia basic_parareal_example.jl
"""

using MPI
using Heat3ds

function main()
    # MPI初期化
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    if rank == 0
        println("=== Basic Parareal Example ===")
        println("MPI processes: $size")
        println("Julia threads per process: $(Threads.nthreads())")
    end
    
    try
        # 基本的なParareal設定
        config = PararealConfig(
            # 時間設定
            total_time=1.0,
            n_time_windows=size,  # プロセス数と同じ
            
            # 時間ステップ設定
            dt_coarse=0.01,
            dt_fine=0.001,
            time_step_ratio=10.0,
            
            # 収束設定
            max_iterations=15,
            convergence_tolerance=1.0e-6,
            
            # 並列化設定
            n_mpi_processes=size,
            n_threads_per_process=Threads.nthreads(),
            
            # 最適化設定
            auto_optimize_parameters=false,
            parameter_exploration_mode=false
        )
        
        # 問題設定（中規模問題）
        NX, NY, NZ = 64, 64, 32
        
        if rank == 0
            println("Problem size: $NX × $NY × $NZ = $(NX*NY*NZ) grid points")
            println("Time domain: [0, $(config.total_time)] with $(config.n_time_windows) windows")
            println("Time step ratio: $(config.time_step_ratio)")
        end
        
        # 実行時間測定開始
        start_time = time()
        
        # Parareal実行
        result = q3d(NX, NY, NZ,
                    solver="pbicgstab",
                    smoother="",
                    epsilon=1.0e-6,
                    par="thread",
                    is_steady=false,
                    parareal=true,
                    parareal_config=config)
        
        # 実行時間測定終了
        total_time = time() - start_time
        
        if rank == 0
            println("\n=== Results ===")
            println("Parareal computation completed successfully!")
            println("Total execution time: $(round(total_time, digits=2)) seconds")
            println("Parareal iterations: $(result.parareal_iterations)")
            println("Converged: $(result.converged)")
            
            if haskey(result, :performance_metrics)
                println("Overall speedup: $(round(result.performance_metrics.overall_speedup, digits=2))x")
                println("MPI efficiency: $(round(result.performance_metrics.mpi_efficiency * 100, digits=1))%")
                println("Threading efficiency: $(round(result.performance_metrics.threading_efficiency * 100, digits=1))%")
            end
            
            if haskey(result, :validation_metrics)
                println("L2 norm error: $(result.validation_metrics.l2_norm_error)")
                println("Max pointwise error: $(result.validation_metrics.max_pointwise_error)")
            end
            
            println("\n=== Output Files ===")
            println("Temperature data: temperature_parareal.dat")
            println("Convergence history: convergence_parareal.csv")
        end
        
    catch e
        if rank == 0
            println("Error occurred during Parareal computation:")
            println(e)
            
            # エラー時は逐次計算にフォールバック
            println("\nFalling back to sequential computation...")
            try
                result_seq = q3d(32, 32, 16,  # より小さい問題サイズ
                               solver="pbicgstab",
                               epsilon=1.0e-6,
                               par="thread",
                               is_steady=false,
                               parareal=false)
                println("Sequential computation completed successfully")
            catch seq_error
                println("Sequential computation also failed: $seq_error")
            end
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