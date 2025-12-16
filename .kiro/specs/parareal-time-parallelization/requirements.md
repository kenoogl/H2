# Requirements Document

## Introduction

Heat3dsシミュレーションコードにおいて、非定常熱伝導解析の計算時間を短縮するため、時間方向の並列化手法であるparareal法を導入する。現在のHeat3dsは空間方向の並列化（ThreadsX）のみ対応しており、時間方向の並列化は未実装である。本機能では、MPI（Message Passing Interface）を用いた時間方向の並列化と、各MPIプロセス内でのThreadsによる空間方向の並列化を組み合わせたハイブリッド並列化を実装する。これにより、複数の時間ステップを並列に計算し、大規模な非定常解析の高速化を実現する。

## Glossary

- **Parareal法**: 時間方向の並列化手法。粗い時間ステップでの予測と細かい時間ステップでの修正を組み合わせて並列計算を行う
- **Heat3ds**: 3次元非一様格子熱伝導解析シミュレーションコード
- **MPI_Process**: 時間方向の並列化を担当するMPIプロセス。各プロセスが異なる時間窓を処理
- **Thread_Pool**: 各MPIプロセス内で空間方向の並列化を担当するスレッドプール
- **Hybrid_Parallelization**: MPI（時間方向）とThreads（空間方向）を組み合わせた並列化手法
- **Coarse_Solver**: 粗い時間ステップ（大きなΔt）で高速に計算する粗解法
- **Fine_Solver**: 細かい時間ステップ（小さなΔt）で高精度に計算する精密解法
- **Time_Window**: 並列計算する時間区間の単位。各MPIプロセスに割り当てられる
- **Predictor**: 粗解法による初期予測値
- **Corrector**: 精密解法による修正値
- **Convergence_Tolerance**: Parareal反復の収束判定基準
- **MPI_Communication**: MPIプロセス間での温度場データの通信
- **Spatial_Threading**: 各MPIプロセス内での空間方向ThreadsX並列化
- **Time_Step_Ratio**: 粗解法と精密解法の時間積分幅の比率（Δt_coarse / Δt_fine）
- **Optimal_Parameters**: 文献ベースのガイドラインに基づく最適パラメータ設定
- **Parameter_Tuning**: 自動的な最適パラメータ探索機能

## Requirements

### Requirement 1

**User Story:** As a thermal analysis engineer, I want to perform large-scale transient heat conduction simulations with reduced computation time using hybrid MPI+Threads parallelization, so that I can analyze complex IC thermal behavior efficiently.

#### Acceptance Criteria

1. WHEN a user specifies parareal mode for transient analysis, THE Heat3ds_System SHALL initialize MPI environment and divide the time domain into parallel time windows
2. WHEN time windows are created, THE Heat3ds_System SHALL assign each time window to available MPI processes
3. WHEN each MPI process begins computation, THE Heat3ds_System SHALL initialize thread pools for spatial parallelization within each process
4. WHEN parareal iteration begins, THE Heat3ds_System SHALL compute coarse predictions for all time windows simultaneously using MPI processes
5. WHEN coarse predictions are complete, THE Heat3ds_System SHALL compute fine corrections for each time window in parallel using both MPI and spatial threading
6. WHEN parareal convergence is achieved, THE Heat3ds_System SHALL provide the final temperature distribution with accuracy equivalent to sequential fine solver

### Requirement 2

**User Story:** As a computational scientist, I want to configure parareal algorithm parameters, so that I can optimize the balance between accuracy and computational efficiency.

#### Acceptance Criteria

1. WHEN configuring parareal parameters, THE Heat3ds_System SHALL accept coarse time step size specification
2. WHEN configuring parareal parameters, THE Heat3ds_System SHALL accept fine time step size specification  
3. WHEN configuring parareal parameters, THE Heat3ds_System SHALL accept number of time windows specification
4. WHEN configuring parareal parameters, THE Heat3ds_System SHALL accept maximum parareal iterations specification
5. WHEN configuring parareal parameters, THE Heat3ds_System SHALL accept parareal convergence tolerance specification

### Requirement 3

**User Story:** As a system administrator, I want parareal computation to integrate seamlessly with existing Heat3ds infrastructure, so that current workflows remain functional.

#### Acceptance Criteria

1. WHEN parareal mode is disabled, THE Heat3ds_System SHALL execute standard sequential time stepping
2. WHEN parareal mode is enabled, THE Heat3ds_System SHALL maintain compatibility with existing boundary conditions
3. WHEN parareal computation runs, THE Heat3ds_System SHALL preserve all existing solver options (PBiCGSTAB, CG, SOR)
4. WHEN parareal computation completes, THE Heat3ds_System SHALL generate output in the same format as sequential computation
5. WHEN parareal fails to converge, THE Heat3ds_System SHALL fall back to sequential computation with appropriate warnings

### Requirement 4

**User Story:** As a performance analyst, I want to monitor parareal algorithm efficiency, so that I can evaluate the effectiveness of time parallelization.

#### Acceptance Criteria

1. WHEN parareal computation executes, THE Heat3ds_System SHALL record coarse solver computation time for each iteration
2. WHEN parareal computation executes, THE Heat3ds_System SHALL record fine solver computation time for each iteration  
3. WHEN parareal computation executes, THE Heat3ds_System SHALL record communication overhead between time windows
4. WHEN parareal computation completes, THE Heat3ds_System SHALL report total speedup compared to sequential execution
5. WHEN parareal convergence monitoring is enabled, THE Heat3ds_System SHALL output residual norms for each parareal iteration

### Requirement 5

**User Story:** As a validation engineer, I want to verify parareal results against sequential computation, so that I can ensure numerical accuracy is maintained.

#### Acceptance Criteria

1. WHEN parareal validation mode is enabled, THE Heat3ds_System SHALL compute both parareal and sequential solutions
2. WHEN comparing solutions, THE Heat3ds_System SHALL calculate L2 norm differences between parareal and sequential results
3. WHEN comparing solutions, THE Heat3ds_System SHALL calculate maximum pointwise differences between parareal and sequential results
4. WHEN validation completes, THE Heat3ds_System SHALL report whether differences are within acceptable tolerance
5. WHEN validation fails, THE Heat3ds_System SHALL provide detailed error analysis including spatial distribution of differences

### Requirement 6

**User Story:** As a software developer, I want parareal implementation to follow Heat3ds coding standards and utilize MPI.jl for time parallelization, so that the code remains maintainable and extensible.

#### Acceptance Criteria

1. WHEN implementing parareal functions, THE Heat3ds_System SHALL use type parameterization for AbstractFloat compatibility
2. WHEN implementing parareal functions, THE Heat3ds_System SHALL use MPI.jl for inter-process communication and existing ThreadsX for spatial parallelization
3. WHEN implementing parareal data structures, THE Heat3ds_System SHALL integrate with existing WorkBuffers architecture and support MPI data serialization
4. WHEN implementing parareal communication, THE Heat3ds_System SHALL use MPI collective operations for efficient temperature field exchange
5. WHEN implementing parareal algorithms, THE Heat3ds_System SHALL maintain consistent error handling with existing Heat3ds modules across all MPI processes

### Requirement 7

**User Story:** As a researcher, I want to experiment with different parareal configurations, so that I can optimize performance for specific problem types.

#### Acceptance Criteria

1. WHEN selecting coarse solver, THE Heat3ds_System SHALL support reduced spatial resolution for coarse predictions
2. WHEN selecting coarse solver, THE Heat3ds_System SHALL support simplified physics models for coarse predictions
3. WHEN selecting fine solver, THE Heat3ds_System SHALL use full spatial resolution and complete physics models
4. WHEN configuring time windows, THE Heat3ds_System SHALL support adaptive time window sizing based on solution characteristics
5. WHEN analyzing convergence, THE Heat3ds_System SHALL provide diagnostics for optimal coarse/fine solver ratio selection

### Requirement 8

**User Story:** As a HPC system administrator, I want to configure MPI and threading resources efficiently, so that I can maximize computational throughput on available hardware.

#### Acceptance Criteria

1. WHEN launching parareal computation, THE Heat3ds_System SHALL accept MPI process count specification for time parallelization
2. WHEN launching parareal computation, THE Heat3ds_System SHALL accept thread count per MPI process specification for spatial parallelization
3. WHEN initializing MPI processes, THE Heat3ds_System SHALL distribute time windows evenly across available MPI processes
4. WHEN initializing thread pools, THE Heat3ds_System SHALL configure ThreadsX backend within each MPI process according to specified thread count
5. WHEN resource allocation is complete, THE Heat3ds_System SHALL report the hybrid parallelization configuration to all processes

### Requirement 9

**User Story:** As a computational physicist, I want MPI communication to be efficient and robust, so that parareal algorithm can scale to large processor counts.

#### Acceptance Criteria

1. WHEN exchanging temperature fields between time windows, THE Heat3ds_System SHALL use MPI non-blocking communication to overlap computation and communication
2. WHEN synchronizing parareal iterations, THE Heat3ds_System SHALL use MPI collective operations for global convergence checking
3. WHEN handling MPI communication errors, THE Heat3ds_System SHALL implement timeout mechanisms and graceful error recovery
4. WHEN serializing temperature field data, THE Heat3ds_System SHALL use efficient binary serialization compatible with Julia's MPI.jl
5. WHEN load balancing is required, THE Heat3ds_System SHALL support dynamic redistribution of time windows across MPI processes

### Requirement 10

**User Story:** As a performance engineer, I want to monitor both MPI and threading performance separately, so that I can identify bottlenecks in the hybrid parallelization.

#### Acceptance Criteria

1. WHEN parareal computation executes, THE Heat3ds_System SHALL measure MPI communication time separately from computation time
2. WHEN parareal computation executes, THE Heat3ds_System SHALL measure threading efficiency within each MPI process
3. WHEN parareal computation executes, THE Heat3ds_System SHALL track load balancing across MPI processes
4. WHEN performance analysis is enabled, THE Heat3ds_System SHALL output detailed timing breakdown for MPI and threading components
5. WHEN scalability testing is performed, THE Heat3ds_System SHALL report strong and weak scaling metrics for the hybrid parallelization

### Requirement 11

**User Story:** As a computational scientist, I want to optimize coarse/fine time step ratios based on established guidelines, so that I can achieve maximum parareal efficiency for heat conduction problems.

#### Acceptance Criteria

1. WHEN configuring time step parameters, THE Heat3ds_System SHALL provide literature-based default ratios for heat conduction problems (typically Δt_coarse/Δt_fine = 10-100)
2. WHEN analyzing problem characteristics, THE Heat3ds_System SHALL estimate optimal time step ratios based on thermal diffusivity and grid spacing
3. WHEN automatic parameter tuning is enabled, THE Heat3ds_System SHALL perform preliminary runs to determine optimal coarse/fine ratios
4. WHEN parameter optimization completes, THE Heat3ds_System SHALL report the selected time step configuration and expected performance gain
5. WHEN suboptimal parameters are detected, THE Heat3ds_System SHALL provide recommendations for parameter adjustment based on convergence behavior

### Requirement 12

**User Story:** As a research engineer, I want automated parameter exploration capabilities, so that I can systematically find optimal parareal configurations for different problem types.

#### Acceptance Criteria

1. WHEN parameter exploration mode is enabled, THE Heat3ds_System SHALL automatically test multiple coarse/fine time step ratio combinations
2. WHEN exploring parameter space, THE Heat3ds_System SHALL evaluate parareal efficiency metrics including speedup and convergence rate
3. WHEN parameter sweep is performed, THE Heat3ds_System SHALL test different numbers of time windows and MPI process configurations
4. WHEN exploration completes, THE Heat3ds_System SHALL generate performance maps showing optimal parameter regions
5. WHEN optimal parameters are identified, THE Heat3ds_System SHALL save configuration files for future use with similar problems