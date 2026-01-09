# Parareal時間並列化ユーザーガイド

## 概要

Heat3dsシミュレーションコードのParareal時間並列化機能は、MPI（Message Passing Interface）による時間方向の並列化と、ThreadsXによる空間方向の並列化を組み合わせたハイブリッド並列化を提供します。これにより、大規模な非定常熱伝導解析の計算時間を大幅に短縮できます。

## 基本概念

### Parareal法とは
Parareal法は時間方向の並列化手法で、以下の特徴があります：
- **粗解法（Coarse Solver）**: 大きな時間ステップで高速に予測計算
- **精密解法（Fine Solver）**: 小さな時間ステップで高精度に修正計算
- **反復収束**: 予測と修正を繰り返して収束解を求める

### ハイブリッド並列化
- **MPI並列化**: 時間窓を複数のプロセスに分散
- **ThreadsX並列化**: 各プロセス内で空間方向を並列化
- **スケーラビリティ**: プロセス数×スレッド数の並列度を実現

## セットアップ

### 必要な依存関係
```julia
using MPI
using ThreadsX
using Heat3ds
```

### MPI環境の初期化
```bash
# MPIランタイムの確認
mpirun --version

# Julia MPIパッケージの設定
julia -e 'using Pkg; Pkg.add("MPI"); using MPI; MPI.install_mpiexec()'
```

## 基本的な使用方法

### 1. 基本的なParareal実行
```julia
using Heat3ds

# 基本的なparareal実行
result = q3d(64, 64, 32,
            solver="pbicgstab",
            epsilon=1.0e-6,
            par="thread",
            is_steady=false,
            parareal=true,
            parareal_config=PararealConfig(
                total_time=1.0,
                n_time_windows=4,
                dt_coarse=0.01,
                dt_fine=0.001,
                max_iterations=10,
                convergence_tolerance=1.0e-6,
                n_mpi_processes=4,
                n_threads_per_process=8
            ))
```

### 2. MPI実行コマンド
```bash
# 4プロセスでparareal実行
mpirun -np 4 julia parareal_example.jl

# SLURM環境での実行例
sbatch --nodes=2 --ntasks-per-node=2 --cpus-per-task=8 run_parareal.sh
```

## 設定パラメータ

### PararealConfig構造体
```julia
struct PararealConfig{T <: AbstractFloat}
    # 時間分割設定
    total_time::T              # 総計算時間
    n_time_windows::Int        # 時間窓数（通常はMPIプロセス数と同じ）
    
    # 時間ステップ設定
    dt_coarse::T              # 粗解法の時間ステップ
    dt_fine::T                # 精密解法の時間ステップ
    time_step_ratio::T        # dt_coarse / dt_fine（推奨: 10-100）
    
    # 収束設定
    max_iterations::Int       # 最大反復回数（推奨: 5-20）
    convergence_tolerance::T  # 収束判定基準（推奨: 1e-6）
    
    # 並列化設定
    n_mpi_processes::Int      # MPIプロセス数
    n_threads_per_process::Int # プロセス当たりスレッド数
    
    # 最適化設定
    auto_optimize_parameters::Bool    # 自動パラメータ最適化
    parameter_exploration_mode::Bool  # パラメータ探索モード
end
```

### 推奨パラメータ設定

#### 小規模問題（格子点数 < 100万）
```julia
config = PararealConfig(
    total_time=1.0,
    n_time_windows=2,
    dt_coarse=0.01,
    dt_fine=0.001,
    time_step_ratio=10.0,
    max_iterations=10,
    convergence_tolerance=1.0e-6,
    n_mpi_processes=2,
    n_threads_per_process=4,
    auto_optimize_parameters=true,
    parameter_exploration_mode=false
)
```

#### 中規模問題（格子点数 100万-1000万）
```julia
config = PararealConfig(
    total_time=2.0,
    n_time_windows=4,
    dt_coarse=0.02,
    dt_fine=0.0004,
    time_step_ratio=50.0,
    max_iterations=15,
    convergence_tolerance=1.0e-6,
    n_mpi_processes=4,
    n_threads_per_process=8,
    auto_optimize_parameters=true,
    parameter_exploration_mode=false
)
```

#### 大規模問題（格子点数 > 1000万）
```julia
config = PararealConfig(
    total_time=5.0,
    n_time_windows=8,
    dt_coarse=0.05,
    dt_fine=0.0005,
    time_step_ratio=100.0,
    max_iterations=20,
    convergence_tolerance=1.0e-6,
    n_mpi_processes=8,
    n_threads_per_process=16,
    auto_optimize_parameters=true,
    parameter_exploration_mode=true
)
```

## 性能最適化

### 1. 時間ステップ比率の調整
```julia
# 熱拡散率に基づく推奨比率
function estimate_optimal_ratio(thermal_diffusivity, grid_spacing)
    # 文献ベースの経験式
    base_ratio = sqrt(thermal_diffusivity / grid_spacing^2)
    return clamp(base_ratio, 10.0, 100.0)
end
```

### 2. 自動パラメータ最適化
```julia
# 自動最適化を有効にした実行
config = PararealConfig(
    # ... 他のパラメータ
    auto_optimize_parameters=true,
    parameter_exploration_mode=true
)

result = q3d(NX, NY, NZ, parareal=true, parareal_config=config)

# 最適化結果の確認
println("Optimal time step ratio: ", result.optimal_parameters.time_step_ratio)
println("Expected speedup: ", result.performance_metrics.overall_speedup)
```

### 3. 負荷分散の調整
```julia
# プロセス数とスレッド数のバランス調整
function optimize_process_thread_balance(total_cores, problem_size)
    if problem_size < 1_000_000
        # 小規模問題: スレッド並列化を重視
        n_processes = min(4, total_cores ÷ 4)
        n_threads = total_cores ÷ n_processes
    else
        # 大規模問題: MPI並列化を重視
        n_processes = min(16, total_cores ÷ 2)
        n_threads = total_cores ÷ n_processes
    end
    return n_processes, n_threads
end
```

## 検証と精度確認

### 1. 逐次計算との比較
```julia
# 検証モードでの実行
config = PararealConfig(
    # ... パラメータ設定
    validation_mode=true
)

result = q3d(NX, NY, NZ, parareal=true, parareal_config=config)

# 精度メトリクスの確認
println("L2 norm error: ", result.validation_metrics.l2_norm_error)
println("Max pointwise error: ", result.validation_metrics.max_pointwise_error)
println("Relative error: ", result.validation_metrics.relative_error)
```

### 2. 収束履歴の監視
```julia
# 収束履歴の可視化
using Plots

plot(result.convergence_history, 
     xlabel="Parareal Iteration", 
     ylabel="Residual Norm",
     yscale=:log10,
     title="Parareal Convergence History")
```

## トラブルシューティング

### よくある問題と解決策

#### 1. 収束しない場合
**症状**: Parareal反復が収束しない
**原因**: 時間ステップ比率が大きすぎる
**解決策**:
```julia
# 時間ステップ比率を小さくする
config.time_step_ratio = 10.0  # 100.0から10.0に変更
config.max_iterations = 20     # 反復回数を増加
```

#### 2. 性能が出ない場合
**症状**: 逐次計算より遅い
**原因**: 問題サイズが小さすぎる、または通信オーバーヘッドが大きい
**解決策**:
```julia
# プロセス数を減らす
config.n_mpi_processes = 2    # 4から2に変更
config.n_threads_per_process = 8  # スレッド数を増加
```

#### 3. メモリ不足エラー
**症状**: OutOfMemoryError
**原因**: 各プロセスのメモリ使用量が過大
**解決策**:
```julia
# 空間解像度を動的に削減
config.coarse_spatial_resolution_factor = 0.5  # 粗解法の解像度を半分に
```

#### 4. MPI通信エラー
**症状**: MPI_ERR_COMM または通信タイムアウト
**原因**: ネットワーク問題またはプロセス間同期の問題
**解決策**:
```bash
# MPI設定の調整
export OMPI_MCA_btl_tcp_if_include=eth0
export OMPI_MCA_oob_tcp_if_include=eth0

# タイムアウト時間の延長
mpirun --mca orte_abort_timeout 60 -np 4 julia parareal_example.jl
```

### デバッグ機能

#### 1. 詳細ログの有効化
```julia
# デバッグモードでの実行
ENV["PARAREAL_DEBUG"] = "1"
ENV["PARAREAL_LOG_LEVEL"] = "DEBUG"

result = q3d(NX, NY, NZ, parareal=true, parareal_config=config)
```

#### 2. 性能プロファイリング
```julia
# 性能解析の有効化
config.enable_performance_profiling = true

result = q3d(NX, NY, NZ, parareal=true, parareal_config=config)

# 詳細な性能レポートの出力
generate_performance_report(result.performance_metrics, "performance_report.html")
```

## FAQ

### Q1: Pararealはどのような問題に適していますか？
**A**: 以下の条件を満たす問題に適しています：
- 長時間の非定常解析（時間ステップ数が多い）
- 時間方向の結合が比較的弱い問題
- 空間方向の並列化だけでは不十分な大規模問題

### Q2: 最適な時間窓数はどのように決めますか？
**A**: 一般的には以下の指針があります：
- MPIプロセス数と同じにする
- 総時間ステップ数の1/10～1/100程度
- 自動最適化機能を使用して決定

### Q3: 既存のHeat3dsスクリプトをどのように変更すればよいですか？
**A**: 最小限の変更で済みます：
```julia
# 変更前
result = q3d(64, 64, 32, solver="pbicgstab")

# 変更後
result = q3d(64, 64, 32, solver="pbicgstab", 
            parareal=true, 
            parareal_config=PararealConfig(...))
```

### Q4: どの程度の高速化が期待できますか？
**A**: 問題によりますが、一般的には：
- 理想的な場合: MPIプロセス数に比例した高速化
- 実際の場合: 2-8倍程度の高速化（4-16プロセス使用時）
- 通信オーバーヘッドにより効率は70-90%程度

## 参考文献

1. Lions, J.L., Maday, Y., Turinici, G. (2001). "A parareal in time discretization of PDE's"
2. Gander, M.J., Vandewalle, S. (2007). "Analysis of the parareal time-parallel time-integration method"
3. Heat3ds Documentation: [リンク]
4. MPI.jl Documentation: https://juliaparallel.github.io/MPI.jl/stable/

## サポート

問題が発生した場合は、以下の情報を含めてお問い合わせください：
- Heat3dsのバージョン
- Julia、MPI.jlのバージョン
- 実行環境（OS、MPIランタイム）
- エラーメッセージの全文
- 使用したPararealConfig設定