# Design Document

## Overview

Heat3dsシミュレーションコードにParareal時間並列化機能を実装する。本設計では、MPI（Message Passing Interface）による時間方向の並列化と、各MPIプロセス内でのThreadsXによる空間方向の並列化を組み合わせたハイブリッド並列化アーキテクチャを採用する。

Parareal法は、粗い時間ステップでの予測（Predictor）と細かい時間ステップでの修正（Corrector）を反復的に実行することで、時間方向の並列化を実現する。各時間窓を異なるMPIプロセスに割り当て、プロセス間で温度場データを交換しながら収束解を求める。

## Architecture

### システム全体構成

```
MPI Process 0     MPI Process 1     MPI Process 2     MPI Process N-1
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Time Window │   │ Time Window │   │ Time Window │   │ Time Window │
│   [0, T1]   │   │  [T1, T2]   │   │  [T2, T3]   │   │ [TN-1, TN]  │
│             │   │             │   │             │   │             │
│ ThreadsX    │   │ ThreadsX    │   │ ThreadsX    │   │ ThreadsX    │
│ Spatial     │   │ Spatial     │   │ Spatial     │   │ Spatial     │
│ Parallel    │   │ Parallel    │   │ Parallel    │   │ Parallel    │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
       │                 │                 │                 │
       └─────────────────┼─────────────────┼─────────────────┘
                         │                 │
                    MPI Communication
                   (Temperature Exchange)
```

### 階層化アーキテクチャ

1. **MPI Layer (時間並列化)**
   - 時間領域の分割と各プロセスへの割り当て
   - プロセス間での温度場データ交換
   - Parareal反復の同期制御

2. **Threading Layer (空間並列化)**
   - 各MPIプロセス内でのThreadsX並列化
   - 既存のHeat3ds空間並列化機能の活用
   - ソルバー（PBiCGSTAB, CG）の並列実行

3. **Solver Layer (数値計算)**
   - Coarse Solver: 大きな時間ステップでの高速計算
   - Fine Solver: 小さな時間ステップでの高精度計算
   - 既存のNonUniform.jlソルバーの再利用

## Components and Interfaces

### 主要コンポーネント

#### 1. PararealManager
Parareal計算全体を統括するマネージャークラス

```julia
struct PararealManager{T <: AbstractFloat}
    mpi_comm::MPI.Comm
    rank::Int
    size::Int
    time_windows::Vector{TimeWindow{T}}
    coarse_params::CoarseParameters{T}
    fine_params::FineParameters{T}
    convergence_monitor::ConvergenceMonitor{T}
end
```

**主要メソッド:**
- `initialize!(manager, total_time, n_windows)`
- `run_parareal!(manager, initial_condition)`
- `finalize!(manager)`

#### 2. TimeWindow
各MPIプロセスが担当する時間区間を表現

```julia
struct TimeWindow{T <: AbstractFloat}
    start_time::T
    end_time::T
    dt_coarse::T
    dt_fine::T
    n_coarse_steps::Int
    n_fine_steps::Int
    process_rank::Int
end
```

#### 3. MPICommunicator
MPIプロセス間の温度場データ交換を管理

```julia
struct MPICommunicator{T <: AbstractFloat}
    comm::MPI.Comm
    send_buffers::Vector{Array{T,3}}
    recv_buffers::Vector{Array{T,3}}
    requests::Vector{MPI.Request}
end
```

**主要メソッド:**
- `exchange_temperature_fields!(comm, temperature_data)`
- `broadcast_convergence_status!(comm, is_converged)`
- `gather_performance_metrics!(comm, local_metrics)`

#### 4. CoarseSolver / FineSolver
粗解法と精密解法の実装

```julia
abstract type PararealSolver{T <: AbstractFloat} end

struct CoarseSolver{T} <: PararealSolver{T}
    dt::T
    spatial_resolution_factor::T  # 空間解像度削減係数
    simplified_physics::Bool      # 物理モデル簡略化フラグ
    solver_type::Symbol          # :pbicgstab, :cg, :sor
end

struct FineSolver{T} <: PararealSolver{T}
    dt::T
    solver_type::Symbol
    smoother::Symbol
    tolerance::T
end
```

#### 5. ParameterOptimizer
時間ステップ比率の最適化を実行

```julia
struct ParameterOptimizer{T <: AbstractFloat}
    problem_characteristics::ProblemCharacteristics{T}
    optimization_history::Vector{OptimizationResult{T}}
    literature_guidelines::LiteratureGuidelines{T}
end
```

#### 6. ValidationManager
計算精度の検証と逐次実行との比較を管理

```julia
struct ValidationManager{T <: AbstractFloat}
    reference_solver::SequentialSolver{T}
    accuracy_metrics::AccuracyMetrics{T}
    validation_history::Vector{ValidationResult{T}}
    tolerance_settings::ToleranceSettings{T}
end

struct AccuracyMetrics{T}
    l2_norm_error::T
    max_pointwise_error::T
    relative_error::T
    convergence_rate::T
    error_distribution::Array{T,3}
end

struct ValidationResult{T}
    timestamp::DateTime
    problem_id::String
    parareal_config::PararealConfig{T}
    accuracy_metrics::AccuracyMetrics{T}
    is_within_tolerance::Bool
    recommendations::Vector{String}
end
```

**主要メソッド:**
- `validate_against_sequential!(manager, parareal_result, problem)`
- `compute_accuracy_metrics(parareal_data, sequential_data)`
- `generate_error_analysis_report(validation_result)`
- `check_numerical_stability(convergence_history)`

### インターフェース設計

#### Heat3ds統合インターフェース
既存のHeat3dsコードとの統合点

```julia
# 既存のq3d関数を拡張
function q3d(NX::Int, NY::Int, NZ::Int,
             solver::String="sor", smoother::String="";
             epsilon::Float64=1.0e-6, 
             par::String="thread", 
             is_steady::Bool=false,
             parareal::Bool=false,           # 新規追加
             parareal_config::Union{Nothing, PararealConfig}=nothing)  # 新規追加
```

#### MPI初期化インターフェース
```julia
function initialize_mpi_parareal(config::PararealConfig)
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    return PararealManager(comm, rank, size, config)
end
```

## Data Models

### 温度場データ構造
```julia
struct TemperatureField{T <: AbstractFloat}
    data::Array{T,3}           # 温度分布 [NX+2, NY+2, NZ+2]
    time::T                    # 時刻
    metadata::FieldMetadata{T} # メタデータ
end

struct FieldMetadata{T}
    grid_size::NTuple{3,Int}
    spatial_resolution::NTuple{3,T}
    boundary_conditions::BoundaryConditionSet
    checksum::UInt64           # データ整合性チェック用
end
```

### Parareal設定データ
```julia
struct PararealConfig{T <: AbstractFloat}
    # 時間分割設定
    total_time::T
    n_time_windows::Int
    
    # 時間ステップ設定
    dt_coarse::T
    dt_fine::T
    time_step_ratio::T         # dt_coarse / dt_fine
    
    # 収束設定
    max_iterations::Int
    convergence_tolerance::T
    
    # 性能設定
    n_mpi_processes::Int
    n_threads_per_process::Int
    
    # 最適化設定
    auto_optimize_parameters::Bool
    parameter_exploration_mode::Bool
end
```

### 性能監視データ
```julia
struct PerformanceMetrics{T <: AbstractFloat}
    # 計算時間
    coarse_solver_time::Vector{T}
    fine_solver_time::Vector{T}
    communication_time::Vector{T}
    
    # 並列化効率
    mpi_efficiency::T
    threading_efficiency::T
    overall_speedup::T
    
    # 収束特性
    parareal_iterations::Int
    convergence_history::Vector{T}
    
    # リソース使用量
    memory_usage::T
    load_balance_factor::T
end
```

## Error Handling

### エラー分類と対応策

#### 1. MPI通信エラー
```julia
struct MPIError <: Exception
    rank::Int
    error_code::Int
    message::String
end

function handle_mpi_error(error::MPIError)
    @error "MPI Error on rank $(error.rank): $(error.message)"
    # タイムアウト機能付きリトライ
    # 失敗時は逐次計算にフォールバック
end
```

#### 2. 収束失敗エラー
```julia
struct ConvergenceError <: Exception
    iteration::Int
    residual::Float64
    tolerance::Float64
end

function handle_convergence_failure(error::ConvergenceError)
    @warn "Parareal failed to converge after $(error.iteration) iterations"
    # パラメータ自動調整
    # 最終的に逐次計算にフォールバック
end
```

#### 3. メモリ不足エラー
```julia
function handle_memory_error()
    # 空間解像度の動的削減
    # 時間窓数の削減
    # ガベージコレクション強制実行
end
```

## Testing Strategy

### 単体テスト (Unit Tests)

#### MPICommunicatorテスト
```julia
@testset "MPI Communication Tests" begin
    @test test_temperature_field_exchange()
    @test test_convergence_broadcast()
    @test test_error_recovery()
end
```

#### ParameterOptimizerテスト
```julia
@testset "Parameter Optimization Tests" begin
    @test test_literature_guidelines()
    @test test_automatic_tuning()
    @test test_performance_prediction()
end
```

### 統合テスト (Integration Tests)

#### Heat3ds統合テスト
```julia
@testset "Heat3ds Integration Tests" begin
    @test test_parareal_vs_sequential_accuracy()
    @test test_boundary_condition_compatibility()
    @test test_solver_compatibility()
end
```

#### 計算精度検証テスト
```julia
@testset "Computational Accuracy Tests" begin
    @test test_sequential_consistency()
    @test test_numerical_precision_preservation()
    @test test_convergence_tolerance_verification()
    @test test_error_accumulation_bounds()
    @test test_machine_precision_comparison()
end

function test_sequential_consistency()
    # 同一問題設定でparareal vs sequential比較
    problem = create_test_problem()
    
    # Sequential実行
    seq_result = run_sequential_heat3ds(problem)
    
    # Parareal実行
    par_result = run_parareal_heat3ds(problem)
    
    # 結果比較（相対誤差 < 1e-6）
    relative_error = norm(par_result.temperature - seq_result.temperature) / norm(seq_result.temperature)
    @test relative_error < 1e-6
    
    # 最大点別誤差確認
    max_pointwise_error = maximum(abs.(par_result.temperature - seq_result.temperature))
    @test max_pointwise_error < 1e-10  # 機械精度レベル
end

function test_numerical_precision_preservation()
    # 異なる時間ステップ比率での精度確認
    ratios = [10, 50, 100]
    problem = create_test_problem()
    
    for ratio in ratios
        config = PararealConfig(time_step_ratio=ratio)
        result = run_parareal_heat3ds(problem, config)
        
        # Fine solverの理論誤差と比較
        theoretical_error = estimate_fine_solver_error(problem, config.dt_fine)
        actual_error = compute_solution_error(result, analytical_solution(problem))
        
        @test actual_error <= 2.0 * theoretical_error  # 並列化による誤差増大は2倍以内
    end
end
```

### 性能テスト (Performance Tests)

#### スケーラビリティテスト
```julia
@testset "Scalability Tests" begin
    @test test_strong_scaling(process_counts=[1,2,4,8,16])
    @test test_weak_scaling(problem_sizes=[small, medium, large])
    @test test_hybrid_efficiency()
end
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: MPI Environment Initialization Consistency
*For any* valid time domain specification and MPI configuration, initializing parareal mode should successfully create the correct number of time windows and assign them to available processes
**Validates: Requirements 1.1, 1.2**

### Property 2: Hybrid Parallelization Activation
*For any* valid parareal configuration, when computation begins, both MPI processes (for time parallelization) and thread pools (for spatial parallelization) should be active simultaneously
**Validates: Requirements 1.3, 1.4, 1.5**

### Property 3: Parareal Convergence Accuracy
*For any* heat conduction problem, when parareal algorithm converges, the final temperature distribution should match the sequential fine solver result within numerical tolerance (typically 1e-6 relative error)
**Validates: Requirements 1.6**

### Property 11: Sequential Consistency Verification
*For any* identical problem setup, parareal and sequential computations should produce temperature fields that differ by no more than machine precision when using identical solvers and time steps
**Validates: Requirements 5.1, 5.2, 5.3**

### Property 12: Numerical Precision Preservation
*For any* parareal computation, the accumulated numerical error should not exceed the error bounds of the underlying fine solver, ensuring that parallelization does not degrade solution accuracy
**Validates: Requirements 5.4, 5.5**

### Property 4: Parameter Validation Completeness
*For any* valid parareal parameter (time steps, window count, iterations, tolerance), the system should accept and correctly store the configuration
**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

### Property 5: Backward Compatibility Preservation
*For any* existing Heat3ds configuration (boundary conditions, solvers, output formats), enabling parareal mode should maintain identical behavior and results when parareal is disabled
**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

### Property 6: Graceful Degradation
*For any* parareal computation that fails to converge, the system should automatically fall back to sequential computation and produce valid results with appropriate warnings
**Validates: Requirements 3.5**

### Property 7: Time Step Ratio Optimization
*For any* thermal diffusivity and grid spacing combination, the system should estimate time step ratios that fall within physically reasonable bounds and improve performance over naive settings
**Validates: Requirements 11.2, 11.3, 11.4, 11.5**

### Property 8: Parameter Space Exploration Completeness
*For any* parameter exploration configuration, the system should systematically test the specified parameter combinations and generate comprehensive performance metrics
**Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5**

### Property 9: MPI Communication Reliability
*For any* temperature field exchange between MPI processes, the data should be transmitted without corruption and within reasonable time bounds
**Validates: Requirements 9.1, 9.2, 9.3, 9.4**

### Property 10: Performance Monitoring Accuracy
*For any* parareal computation, the performance metrics (timing, efficiency, speedup) should accurately reflect the actual resource usage and computational behavior
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 10.1, 10.2, 10.3, 10.4, 10.5**
