# MPI設定・実行ガイド

## MPI環境のセットアップ

### 1. MPIランタイムのインストール

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install mpich libmpich-dev
# または
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
```

#### CentOS/RHEL
```bash
sudo yum install mpich mpich-devel
# または
sudo yum install openmpi openmpi-devel
```

#### macOS (Homebrew)
```bash
brew install mpich
# または
brew install open-mpi
```

#### HPC環境（モジュールシステム）
```bash
module load mpi/mpich/3.4.2
# または
module load mpi/openmpi/4.1.1
```

### 2. Julia MPI.jlパッケージの設定

```julia
# Julia REPL内で実行
using Pkg
Pkg.add("MPI")

using MPI
MPI.install_mpiexec()  # mpiexecのパスを自動設定

# インストール確認
MPI.mpiexec() do cmd
    run(`$cmd --version`)
end
```

### 3. 環境変数の設定

```bash
# .bashrc または .zshrc に追加
export JULIA_MPI_BINARY=system
export JULIA_MPI_PATH=/usr/lib/x86_64-linux-gnu/openmpi

# Intel MPI使用時
export I_MPI_FABRICS=shm:ofi
export I_MPI_OFI_PROVIDER=tcp

# Open MPI使用時
export OMPI_MCA_btl_vader_single_copy_mechanism=none
```

## 実行方法

### 1. 基本的な実行

#### シングルノード実行
```bash
# 4プロセスで実行
mpirun -np 4 julia parareal_example.jl

# プロセス配置を指定
mpirun -np 4 --map-by core julia parareal_example.jl
```

#### マルチノード実行
```bash
# ホストファイルを使用
echo "node1 slots=8" > hostfile
echo "node2 slots=8" >> hostfile
mpirun -np 16 --hostfile hostfile julia parareal_example.jl

# 直接ホストを指定
mpirun -np 8 -H node1:4,node2:4 julia parareal_example.jl
```

### 2. SLURM環境での実行

#### ジョブスクリプト例（run_parareal.sh）
```bash
#!/bin/bash
#SBATCH --job-name=parareal_heat3ds
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=compute

# モジュールロード
module load julia/1.8.5
module load mpi/openmpi/4.1.1

# 環境変数設定
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1

# 実行
srun julia parareal_example.jl
```

#### ジョブ投入
```bash
sbatch run_parareal.sh
```

### 3. PBS/Torque環境での実行

#### ジョブスクリプト例（run_parareal.pbs）
```bash
#!/bin/bash
#PBS -N parareal_heat3ds
#PBS -l nodes=2:ppn=8
#PBS -l walltime=02:00:00
#PBS -q compute

cd $PBS_O_WORKDIR

# モジュールロード
module load julia/1.8.5
module load mpi/openmpi/4.1.1

# 環境変数設定
export JULIA_NUM_THREADS=8
export OMP_NUM_THREADS=1

# 実行
mpirun -np 16 julia parareal_example.jl
```

#### ジョブ投入
```bash
qsub run_parareal.pbs
```

## 実行スクリプト例

### 基本的なParareal実行スクリプト
```julia
# parareal_example.jl
using MPI
using Heat3ds

# MPI初期化
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

if rank == 0
    println("Starting Parareal computation with $size MPI processes")
end

# Parareal設定
config = PararealConfig(
    total_time=1.0,
    n_time_windows=size,  # プロセス数と同じ
    dt_coarse=0.01,
    dt_fine=0.001,
    time_step_ratio=10.0,
    max_iterations=15,
    convergence_tolerance=1.0e-6,
    n_mpi_processes=size,
    n_threads_per_process=Threads.nthreads(),
    auto_optimize_parameters=true,
    parameter_exploration_mode=false
)

# 問題設定
NX, NY, NZ = 64, 64, 32

try
    # Parareal実行
    result = q3d(NX, NY, NZ,
                solver="pbicgstab",
                epsilon=1.0e-6,
                par="thread",
                is_steady=false,
                parareal=true,
                parareal_config=config)
    
    if rank == 0
        println("Parareal computation completed successfully")
        println("Total speedup: ", result.performance_metrics.overall_speedup)
        println("Parareal iterations: ", result.parareal_iterations)
    end
    
catch e
    if rank == 0
        println("Error occurred: ", e)
    end
    MPI.Abort(comm, 1)
end

# MPI終了
MPI.Finalize()
```

### パラメータ探索スクリプト
```julia
# parameter_exploration.jl
using MPI
using Heat3ds

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

# パラメータ探索設定
time_step_ratios = [10.0, 25.0, 50.0, 100.0]
n_time_windows_list = [2, 4, 8]

results = []

for ratio in time_step_ratios
    for n_windows in n_time_windows_list
        if n_windows <= size  # プロセス数以下の時間窓数のみテスト
            
            config = PararealConfig(
                total_time=1.0,
                n_time_windows=n_windows,
                dt_coarse=0.01,
                dt_fine=0.01/ratio,
                time_step_ratio=ratio,
                max_iterations=20,
                convergence_tolerance=1.0e-6,
                n_mpi_processes=n_windows,
                n_threads_per_process=size ÷ n_windows,
                auto_optimize_parameters=false,
                parameter_exploration_mode=true
            )
            
            if rank == 0
                println("Testing: ratio=$ratio, windows=$n_windows")
            end
            
            try
                result = q3d(32, 32, 16,
                            solver="pbicgstab",
                            parareal=true,
                            parareal_config=config)
                
                if rank == 0
                    push!(results, (
                        ratio=ratio,
                        windows=n_windows,
                        speedup=result.performance_metrics.overall_speedup,
                        iterations=result.parareal_iterations,
                        converged=result.converged
                    ))
                end
                
            catch e
                if rank == 0
                    println("Failed for ratio=$ratio, windows=$n_windows: $e")
                end
            end
        end
    end
end

# 結果出力
if rank == 0
    println("\nParameter Exploration Results:")
    println("Ratio\tWindows\tSpeedup\tIterations\tConverged")
    for r in results
        println("$(r.ratio)\t$(r.windows)\t$(round(r.speedup, digits=2))\t$(r.iterations)\t$(r.converged)")
    end
    
    # 最適パラメータの特定
    converged_results = filter(r -> r.converged, results)
    if !isempty(converged_results)
        best_result = maximum(converged_results, by=r -> r.speedup)
        println("\nBest configuration:")
        println("Time step ratio: $(best_result.ratio)")
        println("Number of windows: $(best_result.windows)")
        println("Speedup: $(round(best_result.speedup, digits=2))")
    end
end

MPI.Finalize()
```

## 性能監視とデバッグ

### 1. 性能プロファイリング
```bash
# Intel VTuneを使用
vtune -collect hotspots -r vtune_results -- mpirun -np 4 julia parareal_example.jl

# GNU gprofを使用
mpirun -np 4 julia --track-allocation=user parareal_example.jl
```

### 2. MPI通信の監視
```bash
# Intel MPI Performance Snapshot
mpirun -psc -np 4 julia parareal_example.jl

# Open MPI with profiling
mpirun --mca btl_base_verbose 10 -np 4 julia parareal_example.jl
```

### 3. デバッグ実行
```bash
# GDBを使用したデバッグ
mpirun -np 4 xterm -e gdb --args julia parareal_example.jl

# Valgrindを使用したメモリチェック
mpirun -np 2 valgrind --tool=memcheck julia parareal_example.jl
```

## トラブルシューティング

### 1. MPI初期化エラー
```bash
# エラー: MPI_Init failed
# 解決策: MPIランタイムとJulia MPI.jlの互換性確認
julia -e 'using MPI; MPI.versioninfo()'
```

### 2. プロセス間通信エラー
```bash
# エラー: MPI_ERR_COMM
# 解決策: ファイアウォール設定確認
sudo ufw allow from 192.168.1.0/24 to any port 22
```

### 3. メモリ不足エラー
```bash
# エラー: OutOfMemoryError
# 解決策: プロセス当たりメモリ制限の調整
ulimit -v 8388608  # 8GB制限
```

### 4. ネットワーク設定問題
```bash
# Open MPI設定
export OMPI_MCA_btl_tcp_if_include=eth0
export OMPI_MCA_oob_tcp_if_include=eth0

# Intel MPI設定
export I_MPI_FABRICS=tcp
export I_MPI_TCP_NETMASK=192.168.1.0/24
```

## ベンチマーク実行

### 弱スケーリングテスト
```bash
#!/bin/bash
# weak_scaling_test.sh

for np in 1 2 4 8 16; do
    echo "Testing with $np processes"
    grid_size=$((32 * np))
    
    mpirun -np $np julia -e "
    using Heat3ds
    result = q3d($grid_size, 32, 32, parareal=true, 
                parareal_config=PararealConfig(n_mpi_processes=$np))
    println(\"Processes: $np, Time: \", result.total_time)
    "
done
```

### 強スケーリングテスト
```bash
#!/bin/bash
# strong_scaling_test.sh

for np in 1 2 4 8 16; do
    echo "Testing with $np processes"
    
    mpirun -np $np julia -e "
    using Heat3ds
    result = q3d(64, 64, 32, parareal=true,
                parareal_config=PararealConfig(n_mpi_processes=$np))
    println(\"Processes: $np, Speedup: \", result.performance_metrics.overall_speedup)
    "
done
```

## 最適化のヒント

### 1. プロセス配置の最適化
```bash
# NUMAノード考慮した配置
mpirun -np 8 --map-by numa:pe=4 julia parareal_example.jl

# ソケット単位での配置
mpirun -np 4 --map-by socket julia parareal_example.jl
```

### 2. メモリ使用量の最適化
```julia
# Julia起動時のメモリ設定
julia --heap-size-hint=4G parareal_example.jl
```

### 3. I/O最適化
```bash
# 並列I/Oの設定
export ROMIO_HINTS=romio_cb_write=enable,romio_ds_write=disable
```