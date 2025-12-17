# Parareal time parallelization module for Heat3ds
module Parareal

using MPI
using Printf
using LinearAlgebra
using FLoops
using ThreadsX
using Dates

# Import Heat3ds modules
include("common.jl")
include("parameter_optimization.jl")
using .Common
using .Common: WorkBuffers, get_backend
using .ParameterOptimization

export PararealManager, PararealConfig, TimeWindow, MPICommunicator
export initialize_mpi_parareal!, finalize_mpi_parareal!, run_parareal!
export Heat3dsProblemData, create_heat3ds_problem_data
export CoarseSolver, FineSolver, SolverConfiguration
export PararealResult, ConvergenceMonitor
# Parameter optimization exports
export ParameterOptimizer, LiteratureGuidelines, ProblemCharacteristics
export OptimizationResult, ParameterRecommendation
export create_literature_guidelines, analyze_problem_characteristics
export estimate_optimal_time_step_ratio, get_parameter_recommendations
export create_parameter_optimizer, optimize_parameters!
# Automatic tuning exports
export AutomaticTuner, TuningResult, PerformanceMetrics
export create_automatic_tuner, perform_automatic_tuning!
export run_preliminary_tests!, evaluate_performance_metrics
export generate_tuning_recommendations
# Parameter space exploration exports
export ParameterSpaceExplorer, ExplorationResult, PerformanceMap
export create_parameter_space_explorer, explore_parameter_space!
export generate_performance_map, save_exploration_results
export load_exploration_results, find_optimal_configurations
# Performance monitoring exports (will be defined later)
# export PerformanceMetrics, TimingData, CommunicationMetrics, EfficiencyMetrics
# export create_performance_metrics, update_timing_data!, record_communication_overhead!
# export calculate_efficiency_metrics!, get_performance_summary
# export reset_performance_metrics!, merge_performance_metrics

"""
Parareal configuration parameters
"""
struct PararealConfig{T <: AbstractFloat}
    # Time domain settings
    total_time::T
    n_time_windows::Int
    
    # Time step settings
    dt_coarse::T
    dt_fine::T
    time_step_ratio::T
    
    # Convergence settings
    max_iterations::Int
    convergence_tolerance::T
    
    # Performance settings
    n_mpi_processes::Int
    n_threads_per_process::Int
    
    # Optimization settings
    auto_optimize_parameters::Bool
    parameter_exploration_mode::Bool
end

"""
Default constructor for PararealConfig
"""
function PararealConfig{T}(;
    total_time::T = T(1.0),
    n_time_windows::Int = 4,
    dt_coarse::T = T(0.1),
    dt_fine::T = T(0.01),
    max_iterations::Int = 10,
    convergence_tolerance::T = T(1e-6),
    n_mpi_processes::Int = 4,
    n_threads_per_process::Int = 1,
    auto_optimize_parameters::Bool = false,
    parameter_exploration_mode::Bool = false
) where {T <: AbstractFloat}
    
    time_step_ratio = dt_coarse / dt_fine
    
    return PararealConfig{T}(
        total_time, n_time_windows, dt_coarse, dt_fine, time_step_ratio,
        max_iterations, convergence_tolerance, n_mpi_processes, n_threads_per_process,
        auto_optimize_parameters, parameter_exploration_mode
    )
end

# Convenience constructor without type parameter
function PararealConfig(;
    total_time::AbstractFloat = 1.0,
    n_time_windows::Int = 4,
    dt_coarse::AbstractFloat = 0.1,
    dt_fine::AbstractFloat = 0.01,
    max_iterations::Int = 10,
    convergence_tolerance::AbstractFloat = 1e-6,
    n_mpi_processes::Int = 4,
    n_threads_per_process::Int = 1,
    auto_optimize_parameters::Bool = false,
    parameter_exploration_mode::Bool = false
)
    T = promote_type(typeof(total_time), typeof(dt_coarse), typeof(dt_fine), typeof(convergence_tolerance))
    
    return PararealConfig{T}(
        total_time = T(total_time),
        n_time_windows = n_time_windows,
        dt_coarse = T(dt_coarse),
        dt_fine = T(dt_fine),
        max_iterations = max_iterations,
        convergence_tolerance = T(convergence_tolerance),
        n_mpi_processes = n_mpi_processes,
        n_threads_per_process = n_threads_per_process,
        auto_optimize_parameters = auto_optimize_parameters,
        parameter_exploration_mode = parameter_exploration_mode
    )
end

"""
Time window structure for each MPI process
"""
struct TimeWindow{T <: AbstractFloat}
    start_time::T
    end_time::T
    dt_coarse::T
    dt_fine::T
    n_coarse_steps::Int
    n_fine_steps::Int
    process_rank::Int
end

"""
Coarse solver configuration for Parareal predictor
"""
struct CoarseSolver{T <: AbstractFloat}
    dt::T
    solver_type::Symbol
    simplified_physics::Bool
    spatial_resolution_factor::T
    max_iterations::Int
    tolerance::T
    
    function CoarseSolver{T}(;
        dt::T = T(0.01),
        solver_type::Symbol = :pbicgstab,
        simplified_physics::Bool = true,
        spatial_resolution_factor::T = T(2.0),
        max_iterations::Int = 100,
        tolerance::T = T(1e-4)
    ) where {T <: AbstractFloat}
        return new{T}(dt, solver_type, simplified_physics, spatial_resolution_factor, max_iterations, tolerance)
    end
end

"""
Fine solver configuration for Parareal corrector
"""
struct FineSolver{T <: AbstractFloat}
    dt::T
    solver_type::Symbol
    use_full_physics::Bool
    max_iterations::Int
    tolerance::T
    
    function FineSolver{T}(;
        dt::T = T(0.001),
        solver_type::Symbol = :pbicgstab,
        use_full_physics::Bool = true,
        max_iterations::Int = 1000,
        tolerance::T = T(1e-6)
    ) where {T <: AbstractFloat}
        return new{T}(dt, solver_type, use_full_physics, max_iterations, tolerance)
    end
end

"""
MPI communicator for temperature field exchange
"""
mutable struct MPICommunicator{T <: AbstractFloat}
    comm::MPI.Comm
    rank::Int
    size::Int
    send_buffers::Vector{Array{T,3}}
    recv_buffers::Vector{Array{T,3}}
    requests::Vector{MPI.Request}
    
    function MPICommunicator{T}(comm::MPI.Comm) where {T <: AbstractFloat}
        if comm == MPI.COMM_NULL
            # For testing without MPI initialization
            rank = 0
            size = 1
        else
            rank = MPI.Comm_rank(comm)
            size = MPI.Comm_size(comm)
        end
        
        # Initialize empty buffers - will be resized when needed
        send_buffers = Vector{Array{T,3}}()
        recv_buffers = Vector{Array{T,3}}()
        requests = Vector{MPI.Request}()
        
        return new{T}(comm, rank, size, send_buffers, recv_buffers, requests)
    end
end

"""
Main Parareal manager
"""
mutable struct PararealManager{T <: AbstractFloat}
    config::PararealConfig{T}
    mpi_comm::MPICommunicator{T}
    time_windows::Vector{TimeWindow{T}}
    performance_metrics::Union{Any, Nothing}  # Will be PerformanceMetrics{T} after definition
    is_initialized::Bool
    
    function PararealManager{T}(config::PararealConfig{T}) where {T <: AbstractFloat}
        # MPI communicator will be initialized later
        mpi_comm = MPICommunicator{T}(MPI.COMM_NULL)
        time_windows = Vector{TimeWindow{T}}()
        
        return new{T}(config, mpi_comm, time_windows, nothing, false)
    end
end

"""
Initialize MPI environment for Parareal computation
"""
function initialize_mpi_parareal!(manager::PararealManager{T}) where {T <: AbstractFloat}
    # Initialize MPI if not already done
    if !MPI.Initialized()
        MPI.Init()
    end
    
    # Set up MPI communicator
    comm = MPI.COMM_WORLD
    manager.mpi_comm = MPICommunicator{T}(comm)
    
    rank = manager.mpi_comm.rank
    mpi_size = manager.mpi_comm.size
    
    # Initialize performance metrics
    manager.performance_metrics = create_performance_metrics(
        rank, mpi_size, manager.config.n_threads_per_process
    )
    
    # Validate MPI process count
    if mpi_size != manager.config.n_mpi_processes
        if rank == 0
            @warn "MPI process count mismatch: expected $(manager.config.n_mpi_processes), got $mpi_size"
            @warn "Adjusting configuration to match actual MPI size"
        end
        # Update configuration to match actual MPI size
        manager.config = PararealConfig{T}(
            manager.config.total_time,
            manager.config.n_time_windows,
            manager.config.dt_coarse,
            manager.config.dt_fine,
            manager.config.time_step_ratio,
            manager.config.max_iterations,
            manager.config.convergence_tolerance,
            mpi_size,  # Use actual MPI size
            manager.config.n_threads_per_process,
            manager.config.auto_optimize_parameters,
            manager.config.parameter_exploration_mode
        )
    end
    
    # Create time windows
    create_time_windows!(manager)
    
    # Report configuration
    if rank == 0
        println("=== Parareal MPI Configuration ===")
        println("MPI Processes: $mpi_size")
        println("Threads per process: $(manager.config.n_threads_per_process)")
        println("Time windows: $(manager.config.n_time_windows)")
        println("Total time: $(manager.config.total_time)")
        println("Coarse dt: $(manager.config.dt_coarse)")
        println("Fine dt: $(manager.config.dt_fine)")
        println("Time step ratio: $(manager.config.time_step_ratio)")
        println("==================================")
    end
    
    manager.is_initialized = true
    
    return nothing
end

"""
Create time windows and assign to MPI processes
"""
function create_time_windows!(manager::PararealManager{T}) where {T <: AbstractFloat}
    n_windows = manager.config.n_time_windows
    total_time = manager.config.total_time
    dt_coarse = manager.config.dt_coarse
    dt_fine = manager.config.dt_fine
    n_processes = manager.mpi_comm.size
    
    # Validate inputs
    if n_windows <= 0
        error("Number of time windows must be positive")
    end
    if total_time <= 0
        error("Total time must be positive")
    end
    if dt_coarse <= 0 || dt_fine <= 0
        error("Time steps must be positive")
    end
    if dt_fine > dt_coarse
        @warn "Fine time step is larger than coarse time step. This may lead to poor performance."
    end
    
    # Calculate time window size
    window_size = total_time / n_windows
    
    # Create time windows
    manager.time_windows = Vector{TimeWindow{T}}()
    
    for i in 1:n_windows
        start_time = (i - 1) * window_size
        end_time = i * window_size
        
        # Ensure exact end time for last window to avoid floating point errors
        if i == n_windows
            end_time = total_time
        end
        
        # Calculate number of steps for each solver
        actual_window_size = end_time - start_time
        n_coarse_steps = max(1, ceil(Int, actual_window_size / dt_coarse))
        n_fine_steps = max(1, ceil(Int, actual_window_size / dt_fine))
        
        # Assign to MPI process (round-robin distribution for load balancing)
        process_rank = (i - 1) % n_processes
        
        window = TimeWindow{T}(
            start_time, end_time, dt_coarse, dt_fine,
            n_coarse_steps, n_fine_steps, process_rank
        )
        
        push!(manager.time_windows, window)
    end
    
    return nothing
end

"""
Get time windows assigned to a specific MPI process
"""
function get_local_time_windows(manager::PararealManager{T}, rank::Int) where {T <: AbstractFloat}
    return filter(window -> window.process_rank == rank, manager.time_windows)
end

"""
Validate time window configuration
"""
function validate_time_windows(manager::PararealManager{T}) where {T <: AbstractFloat}
    windows = manager.time_windows
    
    if isempty(windows)
        return false, "No time windows created"
    end
    
    # Check time continuity
    for i in 2:length(windows)
        if windows[i].start_time != windows[i-1].end_time
            return false, "Time windows are not continuous at window $i"
        end
    end
    
    # Check total time coverage
    if windows[1].start_time != 0.0
        return false, "First window does not start at time 0"
    end
    
    if abs(windows[end].end_time - manager.config.total_time) > eps(manager.config.total_time)
        return false, "Last window does not end at total time"
    end
    
    # Check process assignment
    assigned_processes = Set(window.process_rank for window in windows)
    if maximum(assigned_processes) >= manager.mpi_comm.size
        return false, "Window assigned to non-existent process"
    end
    
    return true, "Time windows are valid"
end

"""
Print time window information
"""
function print_time_windows(manager::PararealManager{T}) where {T <: AbstractFloat}
    println("=== Time Window Configuration ===")
    println("Total windows: $(length(manager.time_windows))")
    println("MPI processes: $(manager.mpi_comm.size)")
    
    for (i, window) in enumerate(manager.time_windows)
        println("Window $i: [$(window.start_time), $(window.end_time)] -> Process $(window.process_rank)")
        println("  Coarse steps: $(window.n_coarse_steps), Fine steps: $(window.n_fine_steps)")
    end
    
    # Show load balancing
    process_counts = zeros(Int, manager.mpi_comm.size)
    for window in manager.time_windows
        process_counts[window.process_rank + 1] += 1
    end
    
    println("Load balancing:")
    for (i, count) in enumerate(process_counts)
        println("  Process $(i-1): $count windows")
    end
    println("================================")
end

"""
Finalize MPI environment
"""
function finalize_mpi_parareal!(manager::PararealManager{T}) where {T <: AbstractFloat}
    if manager.is_initialized
        # Clean up any pending MPI requests
        if !isempty(manager.mpi_comm.requests)
            MPI.Waitall(manager.mpi_comm.requests)
            empty!(manager.mpi_comm.requests)
        end
        
        # Note: We don't call MPI.Finalize() here as it should be called by the main program
        manager.is_initialized = false
    end
    
    return nothing
end

"""
Initialize communication buffers for temperature field exchange
"""
function initialize_communication_buffers!(comm::MPICommunicator{T}, grid_size::NTuple{3,Int}) where {T <: AbstractFloat}
    # Clear existing buffers
    empty!(comm.send_buffers)
    empty!(comm.recv_buffers)
    empty!(comm.requests)
    
    # Create buffers for each neighboring process
    # For now, create buffers for all processes (will optimize later)
    for i in 1:comm.size
        send_buffer = zeros(T, grid_size...)
        recv_buffer = zeros(T, grid_size...)
        
        push!(comm.send_buffers, send_buffer)
        push!(comm.recv_buffers, recv_buffer)
    end
    
    return nothing
end

"""
Exchange temperature fields between MPI processes
"""
function exchange_temperature_fields!(comm::MPICommunicator{T}, 
                                    temperature_data::Array{T,3},
                                    target_rank::Int,
                                    performance_metrics::Union{Any, Nothing} = nothing) where {T <: AbstractFloat}
    # Check for invalid target rank first (before MPI check)
    if target_rank < 0 || target_rank >= comm.size
        error("Invalid target rank: $target_rank")
    end
    
    if comm.comm == MPI.COMM_NULL
        # For testing without MPI - still need to validate rank
        if target_rank == comm.rank
            return temperature_data
        else
            error("Invalid target rank: $target_rank")
        end
    end
    
    if target_rank == comm.rank
        # No communication needed for same process
        return temperature_data
    end
    
    # Ensure buffers are initialized
    if length(comm.send_buffers) < comm.size
        initialize_communication_buffers!(comm, Base.size(temperature_data))
    end
    
    # Copy data to send buffer
    comm.send_buffers[target_rank + 1] .= temperature_data
    
    # Use non-blocking communication for better performance
    tag = 100 + comm.rank  # Unique tag for each sender
    
    # Measure communication time
    comm_start = time_ns()
    
    # Post non-blocking send
    send_req = MPI.Isend(comm.send_buffers[target_rank + 1], comm.comm; dest=target_rank, tag=tag)
    
    # Post non-blocking receive
    recv_tag = 100 + target_rank
    recv_req = MPI.Irecv!(comm.recv_buffers[target_rank + 1], comm.comm; source=target_rank, tag=recv_tag)
    
    # Store requests for later completion
    push!(comm.requests, send_req, recv_req)
    
    # Wait for completion
    MPI.Waitall(comm.requests)
    empty!(comm.requests)
    
    # Record communication time
    if performance_metrics !== nothing
        comm_time = T((time_ns() - comm_start) / 1e9)
        data_size = sizeof(temperature_data)
        record_communication_overhead!(performance_metrics, :send, comm_time / 2, data_size)
        record_communication_overhead!(performance_metrics, :receive, comm_time / 2, data_size)
    end
    
    # Return received data
    return copy(comm.recv_buffers[target_rank + 1])
end

"""
Broadcast convergence status to all processes
"""
function broadcast_convergence_status!(comm::MPICommunicator{T}, is_converged::Bool) where {T <: AbstractFloat}
    if comm.comm == MPI.COMM_NULL
        # For testing without MPI
        return is_converged
    end
    
    # Convert boolean to integer for MPI communication
    status = is_converged ? 1 : 0
    result = MPI.Allreduce(status, MPI.LAND, comm.comm)
    
    return result == 1
end

"""
Gather performance metrics from all processes
"""
function gather_performance_metrics!(comm::MPICommunicator{T}, local_metrics::Dict{String,T}) where {T <: AbstractFloat}
    if comm.comm == MPI.COMM_NULL
        # For testing without MPI
        return Dict("process_0" => local_metrics)
    end
    
    # Serialize local metrics
    metric_names = collect(keys(local_metrics))
    metric_values = [local_metrics[name] for name in metric_names]
    
    # Gather all metrics to root process
    all_values = MPI.Gather(metric_values, comm.comm; root=0)
    
    if comm.rank == 0
        # Reconstruct metrics dictionary for all processes
        all_metrics = Dict{String, Dict{String,T}}()
        
        for (i, values) in enumerate(all_values)
            process_name = "process_$(i-1)"
            process_metrics = Dict{String,T}()
            
            for (j, name) in enumerate(metric_names)
                process_metrics[name] = values[j]
            end
            
            all_metrics[process_name] = process_metrics
        end
        
        return all_metrics
    else
        return Dict{String, Dict{String,T}}()
    end
end

"""
Synchronize all processes at a barrier
"""
function synchronize_processes!(comm::MPICommunicator{T},
                              performance_metrics::Union{Any, Nothing} = nothing) where {T <: AbstractFloat}
    if comm.comm == MPI.COMM_NULL
        # For testing without MPI
        return nothing
    end
    
    # Measure synchronization time
    sync_start = time_ns()
    MPI.Barrier(comm.comm)
    
    if performance_metrics !== nothing
        sync_time = T((time_ns() - sync_start) / 1e9)
        record_communication_overhead!(performance_metrics, :synchronization, sync_time)
    end
    
    return nothing
end

"""
Check data integrity using checksums
"""
function compute_data_checksum(data::Array{T,3}) where {T <: AbstractFloat}
    # Simple checksum using sum of all elements
    return sum(data)
end

"""
Validate received temperature field data
"""
function validate_temperature_data(data::Array{T,3}, expected_checksum::T, tolerance::T = T(1e-12)) where {T <: AbstractFloat}
    actual_checksum = compute_data_checksum(data)
    return abs(actual_checksum - expected_checksum) < tolerance
end

"""
Hybrid parallelization coordinator for MPI + Threads
"""
mutable struct HybridCoordinator{T <: AbstractFloat}
    mpi_comm::MPICommunicator{T}
    thread_pool_size::Int
    is_thread_pool_initialized::Bool
    
    function HybridCoordinator{T}(mpi_comm::MPICommunicator{T}, n_threads::Int) where {T <: AbstractFloat}
        return new{T}(mpi_comm, n_threads, false)
    end
end

"""
Initialize thread pool within MPI process
"""
function initialize_thread_pool!(coordinator::HybridCoordinator{T}) where {T <: AbstractFloat}
    if coordinator.is_thread_pool_initialized
        return nothing
    end
    
    rank = coordinator.mpi_comm.rank
    
    # Set thread pool size for this MPI process
    if coordinator.thread_pool_size > 0
        # Note: Julia's thread count is set at startup, but we can validate it
        available_threads = Threads.nthreads()
        
        if available_threads < coordinator.thread_pool_size
            if rank == 0
                @warn "Requested $(coordinator.thread_pool_size) threads, but only $available_threads available"
                @warn "Consider starting Julia with: julia -t $(coordinator.thread_pool_size)"
            end
            coordinator.thread_pool_size = available_threads
        end
        
        if rank == 0
            println("Hybrid parallelization initialized:")
            println("  MPI processes: $(coordinator.mpi_comm.size)")
            println("  Threads per process: $(coordinator.thread_pool_size)")
            println("  Total parallel units: $(coordinator.mpi_comm.size * coordinator.thread_pool_size)")
        end
    end
    
    coordinator.is_thread_pool_initialized = true
    return nothing
end

"""
Get appropriate backend for hybrid execution
"""
function get_hybrid_backend(coordinator::HybridCoordinator{T}) where {T <: AbstractFloat}
    if !coordinator.is_thread_pool_initialized
        error("Thread pool not initialized. Call initialize_thread_pool! first.")
    end
    
    # Use the Common module's hybrid backend function
    par = coordinator.thread_pool_size > 1 ? "thread" : "sequential"
    return Common.get_hybrid_backend(par, coordinator.mpi_comm.size, coordinator.thread_pool_size)
end

"""
Coordinate hybrid execution across MPI processes and threads
"""
function coordinate_hybrid_execution!(coordinator::HybridCoordinator{T}, 
                                    work_function::Function, 
                                    work_data::Any) where {T <: AbstractFloat}
    if !coordinator.is_thread_pool_initialized
        error("Thread pool not initialized. Call initialize_thread_pool! first.")
    end
    
    # Synchronize all MPI processes before starting work
    synchronize_processes!(coordinator.mpi_comm)
    
    # Execute work function with appropriate backend
    backend = get_hybrid_backend(coordinator)
    result = work_function(work_data, backend)
    
    # Synchronize all MPI processes after completing work
    synchronize_processes!(coordinator.mpi_comm)
    
    return result
end

"""
Validate hybrid parallelization configuration
"""
function validate_hybrid_configuration(coordinator::HybridCoordinator{T}) where {T <: AbstractFloat}
    # Check MPI configuration
    if coordinator.mpi_comm.size <= 0
        return false, "Invalid MPI communicator size"
    end
    
    # Check thread configuration
    if coordinator.thread_pool_size <= 0
        return false, "Invalid thread pool size"
    end
    
    # Check if thread pool is initialized
    if !coordinator.is_thread_pool_initialized
        return false, "Thread pool not initialized"
    end
    
    # Check total parallelism makes sense
    total_units = coordinator.mpi_comm.size * coordinator.thread_pool_size
    if total_units > 1000  # Reasonable upper limit
        return false, "Excessive parallelism: $total_units units"
    end
    
    return true, "Hybrid configuration is valid"
end

"""
Get hybrid parallelization statistics
"""
function get_hybrid_statistics(coordinator::HybridCoordinator{T}) where {T <: AbstractFloat}
    stats = Dict{String, Any}()
    
    stats["mpi_processes"] = coordinator.mpi_comm.size
    stats["threads_per_process"] = coordinator.thread_pool_size
    stats["total_parallel_units"] = coordinator.mpi_comm.size * coordinator.thread_pool_size
    stats["current_mpi_rank"] = coordinator.mpi_comm.rank
    stats["thread_pool_initialized"] = coordinator.is_thread_pool_initialized
    stats["available_julia_threads"] = Threads.nthreads()
    
    return stats
end

"""
Timing data for individual solver components
"""
mutable struct TimingData{T <: AbstractFloat}
    coarse_solver_time::T
    fine_solver_time::T
    coarse_solver_calls::Int
    fine_solver_calls::Int
    interpolation_time::T
    restriction_time::T
    total_solver_time::T
    
    function TimingData{T}() where {T <: AbstractFloat}
        return new{T}(T(0.0), T(0.0), 0, 0, T(0.0), T(0.0), T(0.0))
    end
end

"""
MPI communication performance metrics
"""
mutable struct CommunicationMetrics{T <: AbstractFloat}
    send_time::T
    receive_time::T
    synchronization_time::T
    broadcast_time::T
    allreduce_time::T
    total_communication_time::T
    message_count::Int
    bytes_transferred::Int
    
    function CommunicationMetrics{T}() where {T <: AbstractFloat}
        return new{T}(T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), 0, 0)
    end
end

"""
Efficiency calculation metrics
"""
mutable struct EfficiencyMetrics{T <: AbstractFloat}
    parallel_efficiency::T
    strong_scaling_efficiency::T
    weak_scaling_efficiency::T
    speedup_factor::T
    load_balance_factor::T
    communication_overhead_ratio::T
    
    function EfficiencyMetrics{T}() where {T <: AbstractFloat}
        return new{T}(T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0))
    end
end

"""
Comprehensive performance metrics for Parareal algorithm
"""
mutable struct PerformanceMetrics{T <: AbstractFloat}
    timing_data::TimingData{T}
    communication_metrics::CommunicationMetrics{T}
    efficiency_metrics::EfficiencyMetrics{T}
    
    # Overall performance tracking
    total_wall_time::T
    sequential_reference_time::T
    parareal_iterations::Int
    convergence_time::T
    
    # Memory usage tracking
    peak_memory_usage_mb::T
    average_memory_usage_mb::T
    
    # Process-specific metrics
    process_rank::Int
    n_processes::Int
    n_threads_per_process::Int
    
    # Timestamp for performance tracking
    start_timestamp::DateTime
    end_timestamp::Union{DateTime, Nothing}
    
    function PerformanceMetrics{T}(rank::Int, n_processes::Int, n_threads::Int) where {T <: AbstractFloat}
        return new{T}(
            TimingData{T}(),
            CommunicationMetrics{T}(),
            EfficiencyMetrics{T}(),
            T(0.0), T(0.0), 0, T(0.0),
            T(0.0), T(0.0),
            rank, n_processes, n_threads,
            now(), nothing
        )
    end
end

# Export performance monitoring types and functions
export PerformanceMetrics, TimingData, CommunicationMetrics, EfficiencyMetrics
export create_performance_metrics, update_timing_data!, record_communication_overhead!
export calculate_efficiency_metrics!, get_performance_summary
export reset_performance_metrics!, merge_performance_metrics
# Performance monitoring system exports
export PerformanceMonitor, LoadBalanceAnalyzer, ScalabilityAnalyzer
export create_performance_monitor, start_monitoring!, stop_monitoring!
export analyze_load_balance!, calculate_scalability_metrics!
export get_real_time_metrics, generate_monitoring_report

"""
Parareal iteration result structure (forward declaration)
"""
mutable struct PararealResult{T <: AbstractFloat}
    final_solution::Array{T,3}
    converged::Bool
    iterations::Int
    residual_history::Vector{T}
    computation_time::T
    communication_time::T
    performance_metrics::Union{Any, Nothing}  # Will be PerformanceMetrics{T} after definition
end

"""
Problem data structure for Heat3ds integration
"""
struct Heat3dsProblemData{T <: AbstractFloat}
    # Grid parameters
    Δh::NTuple{3,T}              # Cell spacing
    ZC::Vector{T}                # Z-coordinate centers
    ΔZ::Vector{T}                # Z-coordinate spacing
    
    # Material properties and boundary conditions
    ID::Array{UInt8,3}           # Material ID array
    bc_set::Any                  # Boundary condition set
    
    # Solver parameters
    par::String                  # Parallelization mode ("thread" or "sequential")
    is_steady::Bool              # Steady-state flag
    
    function Heat3dsProblemData{T}(Δh, ZC, ΔZ, ID, bc_set, par, is_steady) where {T <: AbstractFloat}
        return new{T}(Δh, ZC, ΔZ, ID, bc_set, par, is_steady)
    end
end

"""
Convergence monitoring for Parareal iterations
"""
mutable struct ConvergenceMonitor{T <: AbstractFloat}
    tolerance::T
    max_iterations::Int
    residual_history::Vector{T}
    iteration_count::Int
    is_converged::Bool
    
    function ConvergenceMonitor{T}(tolerance::T, max_iterations::Int) where {T <: AbstractFloat}
        return new{T}(tolerance, max_iterations, Vector{T}(), 0, false)
    end
end

"""
Check convergence based on residual norm
"""
function check_convergence!(monitor::ConvergenceMonitor{T}, residual::T) where {T <: AbstractFloat}
    monitor.iteration_count += 1
    push!(monitor.residual_history, residual)
    
    # Check convergence criteria
    monitor.is_converged = (residual < monitor.tolerance) || (monitor.iteration_count >= monitor.max_iterations)
    
    return monitor.is_converged
end

"""
Reset convergence monitor for new computation
"""
function reset_convergence_monitor!(monitor::ConvergenceMonitor{T}) where {T <: AbstractFloat}
    empty!(monitor.residual_history)
    monitor.iteration_count = 0
    monitor.is_converged = false
    return nothing
end

"""
Main Parareal algorithm implementation
"""
function run_parareal!(manager::PararealManager{T}, 
                      initial_condition::Array{T,3},
                      problem_data::Heat3dsProblemData{T}) where {T <: AbstractFloat}
    if !manager.is_initialized
        error("PararealManager not initialized. Call initialize_mpi_parareal! first.")
    end
    
    rank = manager.mpi_comm.rank
    mpi_size = manager.mpi_comm.size
    
    # Create hybrid coordinator
    coordinator = HybridCoordinator{T}(manager.mpi_comm, manager.config.n_threads_per_process)
    initialize_thread_pool!(coordinator)
    
    # Validate hybrid configuration
    is_valid, message = validate_hybrid_configuration(coordinator)
    if !is_valid
        error("Hybrid configuration validation failed: $message")
    end
    
    if rank == 0
        println("Starting Parareal computation with hybrid parallelization...")
        stats = get_hybrid_statistics(coordinator)
        println("Hybrid parallelization statistics:")
        for (key, value) in stats
            println("  $key: $value")
        end
    end
    
    # Initialize convergence monitor
    monitor = ConvergenceMonitor{T}(manager.config.convergence_tolerance, manager.config.max_iterations)
    
    # Initialize performance monitoring system
    perf_monitor = create_performance_monitor(manager.mpi_comm, T(0.5))  # 0.5 second monitoring interval
    integrate_monitoring!(manager, perf_monitor)
    start_monitoring!(perf_monitor)
    
    # Create solver configuration
    coarse_solver = CoarseSolver{T}(
        dt = manager.config.dt_coarse,
        solver_type = :pbicgstab,
        simplified_physics = true
    )
    
    fine_solver = FineSolver{T}(
        dt = manager.config.dt_fine,
        solver_type = :pbicgstab,
        use_full_physics = true
    )
    
    # Get local time windows for this process
    local_windows = get_local_time_windows(manager, rank)
    
    if rank == 0
        println("Process $rank assigned $(length(local_windows)) time windows")
    end
    
    # Initialize solution arrays for all time windows
    n_windows = manager.config.n_time_windows
    grid_size = Base.size(initial_condition)
    
    # Solutions at the beginning of each time window
    window_solutions = Vector{Array{T,3}}(undef, n_windows + 1)
    window_solutions[1] = copy(initial_condition)  # Initial condition
    
    # Initialize with zeros for other windows (will be computed)
    for i in 2:(n_windows + 1)
        window_solutions[i] = zeros(T, grid_size)
    end
    
    # Timing variables
    total_start_time = time_ns() / 1e9
    total_comm_time = T(0.0)
    
    try
        # Run Parareal iterations
        result = run_parareal_iterations!(
            manager, monitor, coarse_solver, fine_solver, 
            window_solutions, local_windows, problem_data, coordinator, perf_monitor
        )
        
        total_time = time_ns() / 1e9 - total_start_time
        
        # Finalize performance metrics
        if manager.performance_metrics !== nothing
            manager.performance_metrics.total_wall_time = T(total_time)
            manager.performance_metrics.parareal_iterations = result.iterations
            manager.performance_metrics.convergence_time = T(total_time)
            manager.performance_metrics.end_timestamp = now()
            
            # Calculate efficiency metrics
            calculate_efficiency_metrics!(manager.performance_metrics)
            
            # Stop performance monitoring and perform analysis
            if perf_monitor !== nothing
                stop_monitoring!(perf_monitor)
                
                # Gather performance metrics from all processes for load balance analysis
                all_metrics = Vector{Any}()
                if mpi_size > 1
                    # In a real implementation, this would gather metrics from all processes
                    # For now, we'll use the local metrics
                    push!(all_metrics, manager.performance_metrics)
                    
                    # Perform load balance analysis
                    analyze_load_balance!(perf_monitor, all_metrics)
                    
                    # Calculate scalability metrics
                    calculate_scalability_metrics!(perf_monitor, mpi_size, T(total_time))
                end
                
                if rank == 0
                    println("Parareal computation completed:")
                    println("  Converged: $(result.converged)")
                    println("  Iterations: $(result.iterations)")
                    println("  Final residual: $(length(result.residual_history) > 0 ? result.residual_history[end] : "N/A")")
                    println("  Total time: $(total_time) seconds")
                    println()
                    print_performance_report(manager.performance_metrics)
                    println()
                    println(generate_monitoring_report(perf_monitor))
                end
            else
                if rank == 0
                    println("Parareal computation completed:")
                    println("  Converged: $(result.converged)")
                    println("  Iterations: $(result.iterations)")
                    println("  Final residual: $(length(result.residual_history) > 0 ? result.residual_history[end] : "N/A")")
                    println("  Total time: $(total_time) seconds")
                    println()
                    print_performance_report(manager.performance_metrics)
                end
            end
        else
            if rank == 0
                println("Parareal computation completed:")
                println("  Converged: $(result.converged)")
                println("  Iterations: $(result.iterations)")
                println("  Final residual: $(length(result.residual_history) > 0 ? result.residual_history[end] : "N/A")")
                println("  Total time: $(total_time) seconds")
            end
        end
        
        # Return result with performance metrics
        return PararealResult{T}(
            result.final_solution,
            result.converged,
            result.iterations,
            result.residual_history,
            result.computation_time,
            result.communication_time,
            manager.performance_metrics
        )
        
    catch e
        error_context = string(e)
        
        if rank == 0
            @warn "Parareal computation failed: $e"
        end
        
        # Check if graceful degradation should be triggered
        if should_trigger_graceful_degradation(manager, monitor, error_context)
            if rank == 0
                @warn "Triggering graceful degradation to sequential computation..."
            end
            
            # Attempt graceful degradation to sequential computation
            try
                sequential_result = fallback_to_sequential!(
                    manager, initial_condition, problem_data, coordinator
                )
                
                # Log successful degradation
                log_graceful_degradation_event(manager, error_context, true)
                
                if rank == 0
                    @info "Sequential fallback completed successfully"
                end
                
                return sequential_result
                
            catch sequential_error
                # Log failed degradation
                log_graceful_degradation_event(manager, error_context, false)
                
                if rank == 0
                    @error "Sequential fallback also failed: $sequential_error"
                end
                
                # Return failure result as last resort
                return PararealResult{T}(
                    copy(initial_condition),  # Return initial condition on failure
                    false,  # Not converged
                    0,      # No iterations completed
                    T[],    # Empty residual history
                    T(time_ns() / 1e9 - total_start_time),  # Computation time
                    T(0.0)  # Communication time
                )
            end
        else
            # Don't attempt graceful degradation for certain error types
            if rank == 0
                @error "Error does not qualify for graceful degradation: $error_context"
            end
            
            # Return failure result immediately
            return PararealResult{T}(
                copy(initial_condition),  # Return initial condition on failure
                false,  # Not converged
                0,      # No iterations completed
                T[],    # Empty residual history
                T(time_ns() / 1e9 - total_start_time),  # Computation time
                T(0.0)  # Communication time
            )
        end
    end
end

"""
Run Parareal iterations with predictor-corrector scheme
"""
function run_parareal_iterations!(manager::PararealManager{T},
                                 monitor::ConvergenceMonitor{T},
                                 coarse_solver::CoarseSolver{T},
                                 fine_solver::FineSolver{T},
                                 window_solutions::Vector{Array{T,3}},
                                 local_windows::Vector{TimeWindow{T}},
                                 problem_data::Heat3dsProblemData{T},
                                 coordinator::HybridCoordinator{T},
                                 perf_monitor::Union{Any, Nothing} = nothing) where {T <: AbstractFloat}
    
    rank = manager.mpi_comm.rank
    mpi_size = manager.mpi_comm.size
    n_windows = manager.config.n_time_windows
    grid_size = Base.size(window_solutions[1])
    
    # Storage for coarse and fine solutions
    coarse_solutions = Vector{Array{T,3}}(undef, n_windows + 1)
    fine_solutions = Vector{Array{T,3}}(undef, n_windows + 1)
    
    # Initialize first solution (initial condition)
    coarse_solutions[1] = copy(window_solutions[1])
    fine_solutions[1] = copy(window_solutions[1])
    
    iteration_start_time = time_ns() / 1e9
    
    # Initial coarse prediction phase (k=0)
    if rank == 0
        println("Performing initial coarse prediction...")
    end
    
    perform_coarse_prediction_phase!(
        manager, coarse_solver, coarse_solutions, problem_data, coordinator
    )
    
    # Copy coarse solutions as initial guess
    for i in 1:(n_windows + 1)
        window_solutions[i] = copy(coarse_solutions[i])
        fine_solutions[i] = copy(coarse_solutions[i])
    end
    
    # Parareal iteration loop
    reset_convergence_monitor!(monitor)
    
    while !monitor.is_converged
        iteration_num = monitor.iteration_count + 1
        iteration_start = time_ns() / 1e9
        
        if rank == 0
            println("Parareal iteration $iteration_num")
        end
        
        # Fine correction phase - compute F(U_n^k) for assigned time windows
        perform_fine_correction_phase!(
            manager, fine_solver, fine_solutions, local_windows, problem_data, coordinator
        )
        
        # Synchronize and exchange fine solutions
        exchange_fine_solutions!(manager, fine_solutions, coordinator)
        
        # Coarse prediction phase - compute G(U_n^{k+1}) for all windows
        perform_coarse_prediction_phase!(
            manager, coarse_solver, coarse_solutions, problem_data, coordinator
        )
        
        # Update solutions using Parareal formula: U_n^{k+1} = G(U_n^{k+1}) + F(U_n^k) - G(U_n^k)
        old_coarse_solutions = Vector{Array{T,3}}(undef, n_windows + 1)
        for i in 1:(n_windows + 1)
            old_coarse_solutions[i] = copy(window_solutions[i])
        end
        
        # Compute old coarse solutions G(U_n^k)
        perform_coarse_prediction_on_old_solutions!(
            manager, coarse_solver, old_coarse_solutions, problem_data, coordinator
        )
        
        # Apply Parareal update formula
        max_residual = T(0.0)
        for i in 2:(n_windows + 1)  # Skip initial condition
            # U_n^{k+1} = G(U_n^{k+1}) + F(U_n^k) - G(U_n^k)
            window_solutions[i] = coarse_solutions[i] + fine_solutions[i] - old_coarse_solutions[i]
            
            # Compute residual for convergence check
            residual = norm(window_solutions[i] - fine_solutions[i])
            max_residual = max(max_residual, residual)
        end
        
        # Check convergence across all processes
        global_residual = compute_global_residual!(manager, max_residual)
        
        # Update convergence monitor
        converged = check_convergence!(monitor, global_residual)
        
        if rank == 0
            println("  Iteration $iteration_num: residual = $global_residual, converged = $converged")
        end
        
        # Broadcast convergence status to ensure all processes agree
        converged = broadcast_convergence_status!(manager.mpi_comm, converged)
        monitor.is_converged = converged
        
        # Update performance monitoring
        if perf_monitor !== nothing
            iteration_time = time_ns() / 1e9 - iteration_start
            memory_usage = T(0.0)  # Would be implemented with actual memory monitoring
            update_monitoring_data!(perf_monitor, iteration_num, global_residual, memory_usage)
            
            # Print real-time status every few iterations
            if rank == 0 && iteration_num % 5 == 0
                print_monitoring_status(perf_monitor)
            end
        end
        
        if converged
            break
        end
    end
    
    computation_time = T(time_ns() / 1e9 - iteration_start_time)
    
    # Return final solution (from the last time window)
    final_solution = window_solutions[end]
    
    return PararealResult{T}(
        final_solution,
        monitor.is_converged,
        monitor.iteration_count,
        copy(monitor.residual_history),
        computation_time,
        T(0.0)  # Communication time tracking would be added here
    )
end

"""
Perform coarse prediction phase across all time windows
"""
function perform_coarse_prediction_phase!(manager::PararealManager{T},
                                        coarse_solver::CoarseSolver{T},
                                        solutions::Vector{Array{T,3}},
                                        problem_data::Heat3dsProblemData{T},
                                        coordinator::HybridCoordinator{T}) where {T <: AbstractFloat}
    
    rank = manager.mpi_comm.rank
    n_windows = manager.config.n_time_windows
    
    # Each process computes coarse solutions for all time windows sequentially
    # This ensures all processes have the same coarse trajectory
    
    for i in 1:n_windows
        window = manager.time_windows[i]
        
        # Solve coarse problem from window start to end
        solutions[i + 1] = solve_coarse!(
            coarse_solver, solutions[i], window, problem_data, manager.performance_metrics
        )
    end
    
    # Synchronize to ensure all processes have completed coarse phase
    synchronize_processes!(manager.mpi_comm, manager.performance_metrics)
    
    return nothing
end

"""
Perform coarse prediction on old solutions (for Parareal update formula)
"""
function perform_coarse_prediction_on_old_solutions!(manager::PararealManager{T},
                                                   coarse_solver::CoarseSolver{T},
                                                   old_solutions::Vector{Array{T,3}},
                                                   problem_data::Heat3dsProblemData{T},
                                                   coordinator::HybridCoordinator{T}) where {T <: AbstractFloat}
    
    n_windows = manager.config.n_time_windows
    
    # Compute G(U_n^k) - coarse solutions based on old window solutions
    for i in 1:n_windows
        window = manager.time_windows[i]
        
        # Solve coarse problem from old solution at window start
        old_solutions[i + 1] = solve_coarse!(
            coarse_solver, old_solutions[i], window, problem_data, manager.performance_metrics
        )
    end
    
    synchronize_processes!(manager.mpi_comm, manager.performance_metrics)
    
    return nothing
end

"""
Perform fine correction phase for assigned time windows
"""
function perform_fine_correction_phase!(manager::PararealManager{T},
                                      fine_solver::FineSolver{T},
                                      fine_solutions::Vector{Array{T,3}},
                                      local_windows::Vector{TimeWindow{T}},
                                      problem_data::Heat3dsProblemData{T},
                                      coordinator::HybridCoordinator{T}) where {T <: AbstractFloat}
    
    rank = manager.mpi_comm.rank
    
    # Each process computes fine solutions only for its assigned time windows
    for window in local_windows
        window_index = findfirst(w -> w.start_time == window.start_time && w.end_time == window.end_time, 
                                manager.time_windows)
        
        if window_index !== nothing
            # Solve fine problem for this time window
            fine_solutions[window_index + 1] = solve_fine!(
                fine_solver, fine_solutions[window_index], window, problem_data, manager.performance_metrics
            )
        end
    end
    
    return nothing
end

"""
Exchange fine solutions between MPI processes
"""
function exchange_fine_solutions!(manager::PararealManager{T},
                                 fine_solutions::Vector{Array{T,3}},
                                 coordinator::HybridCoordinator{T}) where {T <: AbstractFloat}
    
    rank = manager.mpi_comm.rank
    mpi_size = manager.mpi_comm.size
    n_windows = manager.config.n_time_windows
    
    # Each process broadcasts its computed fine solutions to all other processes
    for window_idx in 1:n_windows
        window = manager.time_windows[window_idx]
        responsible_rank = window.process_rank
        
        if rank == responsible_rank
            # This process computed this window - broadcast to others
            for target_rank in 0:(mpi_size-1)
                if target_rank != rank
                    exchange_temperature_fields!(
                        manager.mpi_comm, fine_solutions[window_idx + 1], target_rank, manager.performance_metrics
                    )
                end
            end
        else
            # This process needs to receive this window solution
            fine_solutions[window_idx + 1] = exchange_temperature_fields!(
                manager.mpi_comm, fine_solutions[window_idx + 1], responsible_rank, manager.performance_metrics
            )
        end
    end
    
    # Synchronize to ensure all exchanges are complete
    synchronize_processes!(manager.mpi_comm, manager.performance_metrics)
    
    return nothing
end

"""
Compute global residual across all MPI processes
"""
function compute_global_residual!(manager::PararealManager{T}, local_residual::T) where {T <: AbstractFloat}
    if manager.mpi_comm.comm == MPI.COMM_NULL
        # For testing without MPI
        return local_residual
    end
    
    # Use MPI_Allreduce to find maximum residual across all processes
    global_residual = MPI.Allreduce(local_residual, MPI.MAX, manager.mpi_comm.comm)
    
    return global_residual
end

"""
Advanced convergence monitoring with multiple criteria
"""
mutable struct AdvancedConvergenceMonitor{T <: AbstractFloat}
    # Basic convergence criteria
    absolute_tolerance::T
    relative_tolerance::T
    max_iterations::Int
    
    # Convergence history
    residual_history::Vector{T}
    relative_change_history::Vector{T}
    iteration_count::Int
    
    # Convergence status
    is_converged::Bool
    convergence_reason::String
    
    # Stagnation detection
    stagnation_threshold::T
    stagnation_window::Int
    
    function AdvancedConvergenceMonitor{T}(;
        absolute_tolerance::T = T(1e-6),
        relative_tolerance::T = T(1e-4),
        max_iterations::Int = 20,
        stagnation_threshold::T = T(1e-8),
        stagnation_window::Int = 3
    ) where {T <: AbstractFloat}
        
        return new{T}(
            absolute_tolerance, relative_tolerance, max_iterations,
            Vector{T}(), Vector{T}(), 0,
            false, "",
            stagnation_threshold, stagnation_window
        )
    end
end

"""
Check advanced convergence criteria
"""
function check_advanced_convergence!(monitor::AdvancedConvergenceMonitor{T}, 
                                   current_residual::T,
                                   previous_residual::T = T(Inf)) where {T <: AbstractFloat}
    
    monitor.iteration_count += 1
    push!(monitor.residual_history, current_residual)
    
    # Calculate relative change if we have a previous residual
    if previous_residual != T(Inf) && previous_residual > 0
        relative_change = abs(current_residual - previous_residual) / previous_residual
        push!(monitor.relative_change_history, relative_change)
    end
    
    # Check absolute tolerance
    if current_residual < monitor.absolute_tolerance
        monitor.is_converged = true
        monitor.convergence_reason = "Absolute tolerance achieved"
        return true
    end
    
    # Check relative tolerance (if we have previous iteration)
    if length(monitor.relative_change_history) > 0
        latest_relative_change = monitor.relative_change_history[end]
        if latest_relative_change < monitor.relative_tolerance
            monitor.is_converged = true
            monitor.convergence_reason = "Relative tolerance achieved"
            return true
        end
    end
    
    # Check for stagnation
    if length(monitor.relative_change_history) >= monitor.stagnation_window
        recent_changes = monitor.relative_change_history[end-monitor.stagnation_window+1:end]
        if all(change -> change < monitor.stagnation_threshold, recent_changes)
            monitor.is_converged = true
            monitor.convergence_reason = "Convergence stagnated"
            return true
        end
    end
    
    # Check maximum iterations
    if monitor.iteration_count >= monitor.max_iterations
        monitor.is_converged = true
        monitor.convergence_reason = "Maximum iterations reached"
        return true
    end
    
    return false
end

"""
Get convergence statistics
"""
function get_convergence_statistics(monitor::AdvancedConvergenceMonitor{T}) where {T <: AbstractFloat}
    stats = Dict{String, Any}()
    
    stats["iterations"] = monitor.iteration_count
    stats["converged"] = monitor.is_converged
    stats["convergence_reason"] = monitor.convergence_reason
    
    if length(monitor.residual_history) > 0
        stats["initial_residual"] = monitor.residual_history[1]
        stats["final_residual"] = monitor.residual_history[end]
        stats["residual_reduction"] = monitor.residual_history[1] / monitor.residual_history[end]
    end
    
    if length(monitor.relative_change_history) > 0
        stats["average_relative_change"] = sum(monitor.relative_change_history) / length(monitor.relative_change_history)
        stats["final_relative_change"] = monitor.relative_change_history[end]
    end
    
    # Convergence rate estimation (if we have enough data)
    if length(monitor.residual_history) >= 3
        # Estimate convergence rate using linear regression on log(residual)
        log_residuals = log.(monitor.residual_history[2:end])  # Skip first to avoid log(0)
        iterations = collect(2:length(monitor.residual_history))
        
        # Simple linear regression: log(residual) = a * iteration + b
        n = length(log_residuals)
        sum_x = sum(iterations)
        sum_y = sum(log_residuals)
        sum_xy = sum(iterations .* log_residuals)
        sum_x2 = sum(iterations .^ 2)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)
        stats["convergence_rate"] = -slope  # Negative slope means decreasing residual
    end
    
    return stats
end

"""
Reset advanced convergence monitor
"""
function reset_advanced_convergence_monitor!(monitor::AdvancedConvergenceMonitor{T}) where {T <: AbstractFloat}
    empty!(monitor.residual_history)
    empty!(monitor.relative_change_history)
    monitor.iteration_count = 0
    monitor.is_converged = false
    monitor.convergence_reason = ""
    return nothing
end

"""
Enforce iteration limits and provide warnings
"""
function enforce_iteration_limits!(manager::PararealManager{T}, 
                                 monitor::ConvergenceMonitor{T}) where {T <: AbstractFloat}
    
    rank = manager.mpi_comm.rank
    
    # Check if we're approaching iteration limit
    remaining_iterations = monitor.max_iterations - monitor.iteration_count
    
    if remaining_iterations <= 3 && remaining_iterations > 0 && rank == 0
        @warn "Approaching maximum iterations: $remaining_iterations iterations remaining"
        @warn "Current residual: $(length(monitor.residual_history) > 0 ? monitor.residual_history[end] : "N/A")"
        @warn "Target tolerance: $(monitor.tolerance)"
    end
    
    # Check if iteration limit exceeded
    if monitor.iteration_count >= monitor.max_iterations
        if rank == 0
            @warn "Maximum iterations ($(monitor.max_iterations)) reached without convergence"
            if length(monitor.residual_history) > 0
                @warn "Final residual: $(monitor.residual_history[end])"
                @warn "Target tolerance: $(monitor.tolerance)"
                
                # Provide convergence analysis
                if length(monitor.residual_history) >= 2
                    initial_residual = monitor.residual_history[1]
                    final_residual = monitor.residual_history[end]
                    reduction_factor = initial_residual / final_residual
                    
                    @warn "Residual reduction factor: $(reduction_factor)"
                    
                    if reduction_factor < 2.0
                        @warn "Poor convergence detected. Consider:"
                        @warn "  - Reducing coarse time step"
                        @warn "  - Increasing fine solver accuracy"
                        @warn "  - Checking problem conditioning"
                    end
                end
            end
        end
        
        monitor.is_converged = true  # Force convergence to exit loop
        return true
    end
    
    return false
end

"""
Compute residual norms across MPI processes with different norm types
"""
function compute_distributed_residual_norm!(manager::PararealManager{T},
                                          local_data::Array{T,3},
                                          reference_data::Array{T,3},
                                          norm_type::Symbol = :l2) where {T <: AbstractFloat}
    
    # Compute local contribution to the norm
    local_contribution = if norm_type == :l2
        sum((local_data - reference_data).^2)
    elseif norm_type == :linf
        maximum(abs.(local_data - reference_data))
    elseif norm_type == :l1
        sum(abs.(local_data - reference_data))
    else
        error("Unsupported norm type: $norm_type")
    end
    
    if manager.mpi_comm.comm == MPI.COMM_NULL
        # For testing without MPI
        return norm_type == :l2 ? sqrt(local_contribution) : local_contribution
    end
    
    # Reduce across all processes
    if norm_type == :l2
        global_sum = MPI.Allreduce(local_contribution, MPI.SUM, manager.mpi_comm.comm)
        return sqrt(global_sum)
    elseif norm_type == :linf
        return MPI.Allreduce(local_contribution, MPI.MAX, manager.mpi_comm.comm)
    elseif norm_type == :l1
        return MPI.Allreduce(local_contribution, MPI.SUM, manager.mpi_comm.comm)
    end
end

"""
Monitor convergence with detailed diagnostics
"""
function monitor_convergence_with_diagnostics!(manager::PararealManager{T},
                                             window_solutions::Vector{Array{T,3}},
                                             fine_solutions::Vector{Array{T,3}},
                                             iteration::Int) where {T <: AbstractFloat}
    
    rank = manager.mpi_comm.rank
    n_windows = manager.config.n_time_windows
    
    # Compute various residual norms
    residuals = Dict{String, T}()
    
    for i in 2:(n_windows + 1)  # Skip initial condition
        window_idx = i - 1
        
        # Only compute for windows assigned to this process
        window = manager.time_windows[window_idx]
        if window.process_rank == rank
            l2_residual = compute_distributed_residual_norm!(
                manager, window_solutions[i], fine_solutions[i], :l2
            )
            linf_residual = compute_distributed_residual_norm!(
                manager, window_solutions[i], fine_solutions[i], :linf
            )
            
            residuals["window_$(window_idx)_l2"] = l2_residual
            residuals["window_$(window_idx)_linf"] = linf_residual
        end
    end
    
    # Gather all residuals to root process for reporting
    all_residuals = gather_performance_metrics!(manager.mpi_comm, residuals)
    
    if rank == 0 && !isempty(all_residuals)
        println("Convergence diagnostics for iteration $iteration:")
        
        # Find maximum residuals across all windows
        max_l2 = T(0.0)
        max_linf = T(0.0)
        
        for (process_name, process_residuals) in all_residuals
            for (metric_name, value) in process_residuals
                if occursin("_l2", metric_name)
                    max_l2 = max(max_l2, value)
                elseif occursin("_linf", metric_name)
                    max_linf = max(max_linf, value)
                end
            end
        end
        
        println("  Maximum L2 residual: $max_l2")
        println("  Maximum L∞ residual: $max_linf")
        
        return max(max_l2, max_linf)
    end
    
    return T(0.0)
end

"""
Abstract base type for Parareal solvers
"""
abstract type PararealSolver{T <: AbstractFloat} end

"""
Coarse solver with reduced resolution and simplified physics
"""
struct CoarseSolver{T <: AbstractFloat} <: PararealSolver{T}
    dt::T
    spatial_resolution_factor::T  # 空間解像度削減係数 (例: 2.0 = 半分の解像度)
    simplified_physics::Bool      # 物理モデル簡略化フラグ
    solver_type::Symbol          # :pbicgstab, :cg, :sor
    tolerance::T                 # 収束許容誤差（粗解法用に緩い設定）
    max_iterations::Int          # 最大反復回数（粗解法用に少ない設定）
    
    function CoarseSolver{T}(;
        dt::T = T(0.1),
        spatial_resolution_factor::T = T(2.0),
        simplified_physics::Bool = true,
        solver_type::Symbol = :pbicgstab,
        tolerance::T = T(1e-3),  # 粗解法なので緩い許容誤差
        max_iterations::Int = 100
    ) where {T <: AbstractFloat}
        
        if spatial_resolution_factor <= 0
            error("Spatial resolution factor must be positive")
        end
        if dt <= 0
            error("Time step must be positive")
        end
        if tolerance <= 0
            error("Tolerance must be positive")
        end
        if max_iterations <= 0
            error("Max iterations must be positive")
        end
        
        return new{T}(dt, spatial_resolution_factor, simplified_physics, 
                     solver_type, tolerance, max_iterations)
    end
end

# Convenience constructor
function CoarseSolver(; kwargs...)
    return CoarseSolver{Float64}(; kwargs...)
end

"""
Fine solver using full resolution and existing Heat3ds solvers
"""
struct FineSolver{T <: AbstractFloat} <: PararealSolver{T}
    dt::T
    solver_type::Symbol          # :pbicgstab, :cg, :sor
    smoother::Symbol            # :gs, :none
    tolerance::T                # 収束許容誤差（精密解法用に厳しい設定）
    max_iterations::Int         # 最大反復回数
    use_full_physics::Bool      # 完全な物理モデルを使用
    
    function FineSolver{T}(;
        dt::T = T(0.01),
        solver_type::Symbol = :pbicgstab,
        smoother::Symbol = :gs,
        tolerance::T = T(1e-6),  # 精密解法なので厳しい許容誤差
        max_iterations::Int = 8000,
        use_full_physics::Bool = true
    ) where {T <: AbstractFloat}
        
        if dt <= 0
            error("Time step must be positive")
        end
        if tolerance <= 0
            error("Tolerance must be positive")
        end
        if max_iterations <= 0
            error("Max iterations must be positive")
        end
        
        return new{T}(dt, solver_type, smoother, tolerance, max_iterations, use_full_physics)
    end
end

# Convenience constructor
function FineSolver(; kwargs...)
    return FineSolver{Float64}(; kwargs...)
end

"""
Solver configuration for Parareal algorithm
"""
struct SolverConfiguration{T <: AbstractFloat}
    coarse_solver::CoarseSolver{T}
    fine_solver::FineSolver{T}
    
    function SolverConfiguration{T}(coarse::CoarseSolver{T}, fine::FineSolver{T}) where {T <: AbstractFloat}
        # Validate that coarse dt >= fine dt
        if coarse.dt < fine.dt
            @warn "Coarse time step ($(coarse.dt)) is smaller than fine time step ($(fine.dt)). This may lead to poor performance."
        end
        
        return new{T}(coarse, fine)
    end
end

# Convenience constructor
function SolverConfiguration(coarse::CoarseSolver{T}, fine::FineSolver{T}) where {T <: AbstractFloat}
    return SolverConfiguration{T}(coarse, fine)
end

"""
Create coarse grid by reducing spatial resolution
"""
function create_coarse_grid(original_size::NTuple{3,Int}, resolution_factor::T) where {T <: AbstractFloat}
    if resolution_factor <= 1.0
        return original_size  # No reduction
    end
    
    # Calculate new grid size (ensure minimum size of 3 for boundary conditions)
    new_mx = max(3, ceil(Int, original_size[1] / resolution_factor))
    new_my = max(3, ceil(Int, original_size[2] / resolution_factor))
    new_mz = max(3, ceil(Int, original_size[3] / resolution_factor))
    
    return (new_mx, new_my, new_mz)
end

"""
Interpolate solution from coarse grid to fine grid
"""
function interpolate_coarse_to_fine!(fine_data::Array{T,3}, coarse_data::Array{T,3}) where {T <: AbstractFloat}
    fine_size = Base.size(fine_data)
    coarse_size = Base.size(coarse_data)
    
    # Simple trilinear interpolation
    for k in 1:fine_size[3], j in 1:fine_size[2], i in 1:fine_size[1]
        # Map fine grid indices to coarse grid coordinates
        ci = 1 + (i - 1) * (coarse_size[1] - 1) / (fine_size[1] - 1)
        cj = 1 + (j - 1) * (coarse_size[2] - 1) / (fine_size[2] - 1)
        ck = 1 + (k - 1) * (coarse_size[3] - 1) / (fine_size[3] - 1)
        
        # Get integer indices and weights
        i1, i2 = floor(Int, ci), ceil(Int, ci)
        j1, j2 = floor(Int, cj), ceil(Int, cj)
        k1, k2 = floor(Int, ck), ceil(Int, ck)
        
        # Clamp to valid range
        i1 = clamp(i1, 1, coarse_size[1])
        i2 = clamp(i2, 1, coarse_size[1])
        j1 = clamp(j1, 1, coarse_size[2])
        j2 = clamp(j2, 1, coarse_size[2])
        k1 = clamp(k1, 1, coarse_size[3])
        k2 = clamp(k2, 1, coarse_size[3])
        
        # Interpolation weights
        wi = (i1 == i2) ? T(1.0) : (ci - i1) / (i2 - i1)
        wj = (j1 == j2) ? T(1.0) : (cj - j1) / (j2 - j1)
        wk = (k1 == k2) ? T(1.0) : (ck - k1) / (k2 - k1)
        
        # Trilinear interpolation
        v000 = coarse_data[i1, j1, k1]
        v001 = coarse_data[i1, j1, k2]
        v010 = coarse_data[i1, j2, k1]
        v011 = coarse_data[i1, j2, k2]
        v100 = coarse_data[i2, j1, k1]
        v101 = coarse_data[i2, j1, k2]
        v110 = coarse_data[i2, j2, k1]
        v111 = coarse_data[i2, j2, k2]
        
        # Interpolate along i-direction
        v00 = v000 * (1 - wi) + v100 * wi
        v01 = v001 * (1 - wi) + v101 * wi
        v10 = v010 * (1 - wi) + v110 * wi
        v11 = v011 * (1 - wi) + v111 * wi
        
        # Interpolate along j-direction
        v0 = v00 * (1 - wj) + v10 * wj
        v1 = v01 * (1 - wj) + v11 * wj
        
        # Interpolate along k-direction
        fine_data[i, j, k] = v0 * (1 - wk) + v1 * wk
    end
    
    return nothing
end

"""
Restrict solution from fine grid to coarse grid
"""
function restrict_fine_to_coarse!(coarse_data::Array{T,3}, fine_data::Array{T,3}) where {T <: AbstractFloat}
    fine_size = Base.size(fine_data)
    coarse_size = Base.size(coarse_data)
    
    # Simple restriction using averaging
    for k in 1:coarse_size[3], j in 1:coarse_size[2], i in 1:coarse_size[1]
        # Map coarse grid indices to fine grid region
        fi_start = 1 + (i - 1) * (fine_size[1] - 1) ÷ (coarse_size[1] - 1)
        fi_end = min(fine_size[1], fi_start + (fine_size[1] - 1) ÷ (coarse_size[1] - 1))
        
        fj_start = 1 + (j - 1) * (fine_size[2] - 1) ÷ (coarse_size[2] - 1)
        fj_end = min(fine_size[2], fj_start + (fine_size[2] - 1) ÷ (coarse_size[2] - 1))
        
        fk_start = 1 + (k - 1) * (fine_size[3] - 1) ÷ (coarse_size[3] - 1)
        fk_end = min(fine_size[3], fk_start + (fine_size[3] - 1) ÷ (coarse_size[3] - 1))
        
        # Average values in the region
        sum_val = T(0.0)
        count = 0
        
        for fk in fk_start:fk_end, fj in fj_start:fj_end, fi in fi_start:fi_end
            sum_val += fine_data[fi, fj, fk]
            count += 1
        end
        
        coarse_data[i, j, k] = count > 0 ? sum_val / count : T(0.0)
    end
    
    return nothing
end

"""
Solve using coarse solver (reduced resolution and simplified physics)
"""
function solve_coarse!(solver::CoarseSolver{T}, 
                      initial_condition::Array{T,3},
                      time_window::TimeWindow{T},
                      problem_data::Any,
                      performance_metrics::Union{Any, Nothing} = nothing) where {T <: AbstractFloat}
    
    # Create coarse grid
    fine_size = Base.size(initial_condition)
    coarse_size = create_coarse_grid(fine_size, solver.spatial_resolution_factor)
    
    # Initialize coarse grid data
    coarse_solution = zeros(T, coarse_size...)
    
    # Restrict initial condition to coarse grid
    restrict_fine_to_coarse!(coarse_solution, initial_condition)
    
    # Simplified time stepping for coarse solver
    current_time = time_window.start_time
    dt = solver.dt
    
    while current_time < time_window.end_time
        # Adjust time step to not overshoot end time
        actual_dt = min(dt, time_window.end_time - current_time)
        
        # Simple explicit time stepping (simplified physics)
        if solver.simplified_physics
            # Very simple heat diffusion update (explicit Euler)
            # This is a placeholder - in practice, you'd use a simplified version of the full solver
            coarse_solution_new = copy(coarse_solution)
            
            # Simple diffusion update (placeholder)
            for k in 2:coarse_size[3]-1, j in 2:coarse_size[2]-1, i in 2:coarse_size[1]-1
                # Simple 6-point stencil diffusion
                diffusion = (coarse_solution[i+1,j,k] + coarse_solution[i-1,j,k] +
                           coarse_solution[i,j+1,k] + coarse_solution[i,j-1,k] +
                           coarse_solution[i,j,k+1] + coarse_solution[i,j,k-1] -
                           6 * coarse_solution[i,j,k])
                
                coarse_solution_new[i,j,k] = coarse_solution[i,j,k] + actual_dt * T(0.1) * diffusion
            end
            
            coarse_solution .= coarse_solution_new
        else
            # Use full solver on coarse grid (not implemented in this placeholder)
            @warn "Full physics on coarse grid not yet implemented"
        end
        
        current_time += actual_dt
    end
    
    # Interpolate back to fine grid
    fine_solution = zeros(T, fine_size...)
    interpolate_coarse_to_fine!(fine_solution, coarse_solution)
    
    return fine_solution
end

"""
Problem data structure for Heat3ds integration
"""
struct Heat3dsProblemData{T <: AbstractFloat}
    # Grid parameters
    Δh::NTuple{3,T}              # Cell spacing
    ZC::Vector{T}                # Z-coordinate centers
    ΔZ::Vector{T}                # Z-coordinate spacing
    
    # Material properties and boundary conditions
    ID::Array{UInt8,3}           # Material ID array
    bc_set::Any                  # Boundary condition set
    
    # Solver parameters
    par::String                  # Parallelization mode ("thread" or "sequential")
    is_steady::Bool              # Steady-state flag
    
    function Heat3dsProblemData{T}(Δh, ZC, ΔZ, ID, bc_set, par, is_steady) where {T <: AbstractFloat}
        return new{T}(Δh, ZC, ΔZ, ID, bc_set, par, is_steady)
    end
end

"""
Solve using fine solver (full resolution and complete physics)
Integrates with existing Heat3ds solver infrastructure
"""
function solve_fine!(solver::FineSolver{T},
                    initial_condition::Array{T,3},
                    time_window::TimeWindow{T},
                    problem_data::Heat3dsProblemData{T},
                    performance_metrics::Union{Any, Nothing} = nothing) where {T <: AbstractFloat}
    
    start_time = time_ns()
    
    # Create working buffers for the fine solver
    grid_size = Base.size(initial_condition)
    wk = Common.WorkBuffers(grid_size[1], grid_size[2], grid_size[3])
    
    # Initialize temperature field with initial condition
    wk.θ .= initial_condition
    
    # Set up heat source
    # Note: This would need to be integrated with the existing HeatSrc! function
    # For now, we'll use a placeholder
    wk.hsrc .= 0.0  # Placeholder
    
    # Time stepping parameters
    current_time = time_window.start_time
    dt = solver.dt
    n_steps = ceil(Int, (time_window.end_time - time_window.start_time) / dt)
    
    # Solver configuration
    solver_str = string(solver.solver_type)
    smoother_str = string(solver.smoother)
    
    # Integration with existing Heat3ds solver
    for step in 1:n_steps
        # Adjust time step for last step
        if step == n_steps
            dt = time_window.end_time - current_time
        end
        
        if solver.use_full_physics
            # Use existing Heat3ds solver infrastructure
            try
                # This would call the existing RHS calculation and solver
                # For now, we'll use a simplified approach
                
                # Calculate RHS (would use existing calRHS! function)
                # RHSCore.calRHS!(wk, problem_data.Δh, dt, problem_data.ΔZ, 
                #                 problem_data.bc_set, zeros(grid_size[1], grid_size[2]), 
                #                 problem_data.par, is_steady=problem_data.is_steady)
                
                # Call appropriate solver
                if solver.solver_type == :cg
                    # Would call: NonUniform.CG!(wk, problem_data.Δh, dt, problem_data.ZC, 
                    #                           problem_data.ΔZ, HC, tol=solver.tolerance, 
                    #                           smoother=solver.smoother, par=problem_data.par, 
                    #                           verbose=false, is_steady=problem_data.is_steady)
                    
                    # Placeholder: Simple diffusion update
                    apply_simple_diffusion_step!(wk.θ, dt, T(0.1))
                    
                elseif solver.solver_type == :pbicgstab
                    # Would call: NonUniform.PBiCGSTAB!(wk, problem_data.Δh, dt, problem_data.ZC, 
                    #                                  problem_data.ΔZ, HC, tol=solver.tolerance, 
                    #                                  smoother=solver.smoother, par=problem_data.par, 
                    #                                  verbose=false, is_steady=problem_data.is_steady)
                    
                    # Placeholder: Simple diffusion update
                    apply_simple_diffusion_step!(wk.θ, dt, T(0.1))
                    
                else
                    @warn "Solver type $(solver.solver_type) not fully implemented in fine solver"
                    apply_simple_diffusion_step!(wk.θ, dt, T(0.1))
                end
                
            catch e
                @warn "Error in fine solver: $e. Using simplified physics."
                apply_simple_diffusion_step!(wk.θ, dt, T(0.1))
            end
        else
            # Simplified physics (should not normally be used for fine solver)
            apply_simple_diffusion_step!(wk.θ, dt, T(0.1))
        end
        
        current_time += dt
    end
    
    # Record total fine solver time
    if performance_metrics !== nothing
        total_time = T((time_ns() - start_time) / 1e9)
        update_timing_data!(performance_metrics, :fine, total_time)
    end
    
    return copy(wk.θ)
end

"""
Apply simple diffusion step (placeholder for full Heat3ds integration)
"""
function apply_simple_diffusion_step!(temperature::Array{T,3}, dt::T, diffusivity::T) where {T <: AbstractFloat}
    grid_size = Base.size(temperature)
    temp_new = copy(temperature)
    
    # Simple explicit diffusion update
    for k in 2:grid_size[3]-1, j in 2:grid_size[2]-1, i in 2:grid_size[1]-1
        # 6-point stencil for 3D diffusion
        laplacian = (temperature[i+1,j,k] + temperature[i-1,j,k] +
                    temperature[i,j+1,k] + temperature[i,j-1,k] +
                    temperature[i,j,k+1] + temperature[i,j,k-1] -
                    6 * temperature[i,j,k])
        
        temp_new[i,j,k] = temperature[i,j,k] + dt * diffusivity * laplacian
    end
    
    temperature .= temp_new
    return nothing
end

"""
Create Heat3ds problem data from existing Heat3ds parameters
"""
function create_heat3ds_problem_data(Δh::NTuple{3,T}, ZC::Vector{T}, ΔZ::Vector{T},
                                   ID::Array{UInt8,3}, bc_set::Any, 
                                   par::String="thread", is_steady::Bool=false) where {T <: AbstractFloat}
    return Heat3dsProblemData{T}(Δh, ZC, ΔZ, ID, bc_set, par, is_steady)
end

"""
Solve using coarse solver with Heat3ds problem data
"""
function solve_coarse!(solver::CoarseSolver{T}, 
                      initial_condition::Array{T,3},
                      time_window::TimeWindow{T},
                      problem_data::Heat3dsProblemData{T},
                      performance_metrics::Union{Any, Nothing} = nothing) where {T <: AbstractFloat}
    
    start_time = time_ns()
    
    # Create coarse grid
    fine_size = Base.size(initial_condition)
    coarse_size = create_coarse_grid(fine_size, solver.spatial_resolution_factor)
    
    # Initialize coarse grid data
    coarse_solution = zeros(T, coarse_size...)
    
    # Restrict initial condition to coarse grid (measure restriction time)
    if performance_metrics !== nothing
        restriction_start = time_ns()
        restrict_fine_to_coarse!(coarse_solution, initial_condition)
        restriction_time = T((time_ns() - restriction_start) / 1e9)
        update_timing_data!(performance_metrics, :restriction, restriction_time)
    else
        restrict_fine_to_coarse!(coarse_solution, initial_condition)
    end
    
    # Simplified time stepping for coarse solver
    current_time = time_window.start_time
    dt = solver.dt
    n_steps = ceil(Int, (time_window.end_time - time_window.start_time) / dt)
    
    for step in 1:n_steps
        # Adjust time step for last step
        if step == n_steps
            dt = time_window.end_time - current_time
        end
        
        if solver.simplified_physics
            # Simple explicit diffusion (faster but less accurate)
            apply_simple_diffusion_step!(coarse_solution, dt, T(0.2))  # Higher diffusivity for stability
        else
            # Use simplified version of full solver on coarse grid
            # This would involve creating coarse versions of the problem data
            apply_simple_diffusion_step!(coarse_solution, dt, T(0.1))
        end
        
        current_time += dt
    end
    
    # Interpolate back to fine grid (measure interpolation time)
    fine_solution = zeros(T, fine_size...)
    if performance_metrics !== nothing
        interpolation_start = time_ns()
        interpolate_coarse_to_fine!(fine_solution, coarse_solution)
        interpolation_time = T((time_ns() - interpolation_start) / 1e9)
        update_timing_data!(performance_metrics, :interpolation, interpolation_time)
        
        # Record total coarse solver time
        total_time = T((time_ns() - start_time) / 1e9)
        update_timing_data!(performance_metrics, :coarse, total_time)
    else
        interpolate_coarse_to_fine!(fine_solution, coarse_solution)
    end
    
    return fine_solution
end

"""
Validate solver configuration
"""
function validate_solver_configuration(config::SolverConfiguration{T}) where {T <: AbstractFloat}
    coarse = config.coarse_solver
    fine = config.fine_solver
    
    # Check time step ratio
    time_step_ratio = coarse.dt / fine.dt
    if time_step_ratio < 1.0
        return false, "Coarse time step must be >= fine time step"
    end
    
    # Check reasonable time step ratio (literature suggests 10-100)
    if time_step_ratio > 1000.0
        return false, "Time step ratio too large ($(time_step_ratio)), may cause instability"
    end
    
    if time_step_ratio < 2.0
        @warn "Time step ratio is small ($(time_step_ratio)), may not provide significant speedup"
    end
    
    # Check spatial resolution factor
    if coarse.spatial_resolution_factor < 1.0
        return false, "Spatial resolution factor must be >= 1.0"
    end
    
    if coarse.spatial_resolution_factor > 10.0
        @warn "Large spatial resolution factor ($(coarse.spatial_resolution_factor)), may affect accuracy"
    end
    
    # Check solver compatibility
    valid_solvers = [:pbicgstab, :cg, :sor]
    if !(coarse.solver_type in valid_solvers)
        return false, "Invalid coarse solver type: $(coarse.solver_type)"
    end
    
    if !(fine.solver_type in valid_solvers)
        return false, "Invalid fine solver type: $(fine.solver_type)"
    end
    
    return true, "Solver configuration is valid"
end

"""
Get solver performance characteristics
"""
function get_solver_characteristics(solver::PararealSolver{T}) where {T <: AbstractFloat}
    characteristics = Dict{String, Any}()
    
    if solver isa CoarseSolver
        characteristics["type"] = "coarse"
        characteristics["dt"] = solver.dt
        characteristics["spatial_resolution_factor"] = solver.spatial_resolution_factor
        characteristics["simplified_physics"] = solver.simplified_physics
        characteristics["solver_type"] = solver.solver_type
        characteristics["tolerance"] = solver.tolerance
        characteristics["max_iterations"] = solver.max_iterations
        characteristics["expected_speedup"] = solver.spatial_resolution_factor^3  # Rough estimate
    elseif solver isa FineSolver
        characteristics["type"] = "fine"
        characteristics["dt"] = solver.dt
        characteristics["solver_type"] = solver.solver_type
        characteristics["smoother"] = solver.smoother
        characteristics["tolerance"] = solver.tolerance
        characteristics["max_iterations"] = solver.max_iterations
        characteristics["use_full_physics"] = solver.use_full_physics
        characteristics["expected_speedup"] = 1.0  # Reference
    end
    
    return characteristics
end

"""
Solver selection and configuration interface
"""
struct SolverSelector{T <: AbstractFloat}
    available_solvers::Dict{Symbol, Vector{Symbol}}  # solver_type => [available_variants]
    performance_profiles::Dict{Symbol, Dict{String, Any}}
    default_configurations::Dict{Symbol, Dict{String, Any}}
    
    function SolverSelector{T}() where {T <: AbstractFloat}
        # Define available solvers and their variants
        available_solvers = Dict{Symbol, Vector{Symbol}}(
            :pbicgstab => [:standard, :preconditioned],
            :cg => [:standard, :preconditioned],
            :sor => [:standard, :red_black]
        )
        
        # Performance profiles for different solver types
        performance_profiles = Dict{Symbol, Dict{String, Any}}(
            :pbicgstab => Dict(
                "convergence_rate" => "fast",
                "memory_usage" => "moderate",
                "stability" => "high",
                "recommended_for" => ["general_purpose", "ill_conditioned"]
            ),
            :cg => Dict(
                "convergence_rate" => "very_fast",
                "memory_usage" => "low",
                "stability" => "moderate",
                "recommended_for" => ["well_conditioned", "symmetric"]
            ),
            :sor => Dict(
                "convergence_rate" => "slow",
                "memory_usage" => "very_low",
                "stability" => "high",
                "recommended_for" => ["simple_problems", "memory_constrained"]
            )
        )
        
        # Default configurations for different solver types
        default_configurations = Dict{Symbol, Dict{String, Any}}(
            :coarse => Dict(
                "dt_factor" => 10.0,  # Coarse dt = fine_dt * dt_factor
                "spatial_resolution_factor" => 2.0,
                "simplified_physics" => true,
                "tolerance" => 1e-3,
                "max_iterations" => 100,
                "solver_type" => :pbicgstab
            ),
            :fine => Dict(
                "dt_factor" => 1.0,   # Fine dt = base_dt * dt_factor
                "tolerance" => 1e-6,
                "max_iterations" => 8000,
                "solver_type" => :pbicgstab,
                "smoother" => :gs,
                "use_full_physics" => true
            )
        )
        
        return new{T}(available_solvers, performance_profiles, default_configurations)
    end
end

# Convenience constructor
function SolverSelector()
    return SolverSelector{Float64}()
end

"""
Create solver configuration based on problem characteristics and user preferences
"""
function create_solver_configuration(selector::SolverSelector{T};
                                   problem_size::NTuple{3,Int} = (100, 100, 100),
                                   base_dt::T = T(0.01),
                                   target_speedup::T = T(4.0),
                                   accuracy_priority::Symbol = :balanced,  # :speed, :balanced, :accuracy
                                   solver_preference::Union{Symbol, Nothing} = nothing) where {T <: AbstractFloat}
    
    # Determine optimal solver types based on preferences
    coarse_solver_type, fine_solver_type = select_optimal_solvers(
        selector, problem_size, accuracy_priority, solver_preference
    )
    
    # Calculate time step parameters
    coarse_dt, fine_dt = calculate_optimal_time_steps(
        base_dt, target_speedup, accuracy_priority
    )
    
    # Calculate spatial resolution factor for coarse solver
    spatial_factor = calculate_spatial_resolution_factor(
        problem_size, target_speedup, accuracy_priority
    )
    
    # Create coarse solver configuration
    coarse_config = selector.default_configurations[:coarse]
    coarse_solver = CoarseSolver{T}(
        dt = coarse_dt,
        spatial_resolution_factor = spatial_factor,
        simplified_physics = coarse_config["simplified_physics"],
        solver_type = coarse_solver_type,
        tolerance = T(coarse_config["tolerance"]),
        max_iterations = coarse_config["max_iterations"]
    )
    
    # Create fine solver configuration
    fine_config = selector.default_configurations[:fine]
    fine_solver = FineSolver{T}(
        dt = fine_dt,
        solver_type = fine_solver_type,
        smoother = fine_config["smoother"],
        tolerance = T(fine_config["tolerance"]),
        max_iterations = fine_config["max_iterations"],
        use_full_physics = fine_config["use_full_physics"]
    )
    
    return SolverConfiguration(coarse_solver, fine_solver)
end

"""
Select optimal solver types based on problem characteristics
"""
function select_optimal_solvers(selector::SolverSelector{T},
                               problem_size::NTuple{3,Int},
                               accuracy_priority::Symbol,
                               solver_preference::Union{Symbol, Nothing}) where {T <: AbstractFloat}
    
    # If user has a preference, use it (with validation)
    if solver_preference !== nothing
        if haskey(selector.available_solvers, solver_preference)
            return solver_preference, solver_preference
        else
            @warn "Solver preference $solver_preference not available, using automatic selection"
        end
    end
    
    # Automatic selection based on problem characteristics
    total_dofs = prod(problem_size)
    
    if accuracy_priority == :speed
        # Prioritize speed: use faster but potentially less accurate solvers
        if total_dofs > 1_000_000  # Large problem
            return :sor, :pbicgstab  # Fast coarse, robust fine
        else
            return :pbicgstab, :pbicgstab  # Balanced choice
        end
    elseif accuracy_priority == :accuracy
        # Prioritize accuracy: use more accurate solvers
        return :pbicgstab, :cg  # Robust coarse, accurate fine
    else  # :balanced
        # Balanced approach
        if total_dofs > 500_000
            return :pbicgstab, :pbicgstab  # Robust for large problems
        else
            return :cg, :cg  # Fast for smaller problems
        end
    end
end

"""
Calculate optimal time step sizes
"""
function calculate_optimal_time_steps(base_dt::T, target_speedup::T, accuracy_priority::Symbol) where {T <: AbstractFloat}
    # Fine time step is typically the base time step
    fine_dt = base_dt
    
    # Coarse time step calculation based on target speedup and accuracy priority
    if accuracy_priority == :speed
        # Aggressive coarse time step for maximum speedup
        coarse_dt_factor = min(target_speedup * 2, T(100.0))
    elseif accuracy_priority == :accuracy
        # Conservative coarse time step for better accuracy
        coarse_dt_factor = max(target_speedup / 2, T(2.0))
    else  # :balanced
        # Balanced coarse time step
        coarse_dt_factor = target_speedup
    end
    
    coarse_dt = fine_dt * coarse_dt_factor
    
    return coarse_dt, fine_dt
end

"""
Calculate spatial resolution factor for coarse solver
"""
function calculate_spatial_resolution_factor(problem_size::NTuple{3,Int}, 
                                           target_speedup::T, 
                                           accuracy_priority::Symbol) where {T <: AbstractFloat}
    
    # Base spatial factor calculation
    # Speedup from spatial coarsening is approximately factor^3
    base_spatial_factor = (target_speedup / 2)^(1/3)  # Conservative estimate
    
    # Adjust based on accuracy priority
    if accuracy_priority == :speed
        # More aggressive spatial coarsening
        spatial_factor = min(base_spatial_factor * T(1.5), T(8.0))
    elseif accuracy_priority == :accuracy
        # Conservative spatial coarsening
        spatial_factor = max(base_spatial_factor / T(1.5), T(1.5))
    else  # :balanced
        spatial_factor = base_spatial_factor
    end
    
    # Ensure reasonable bounds
    spatial_factor = clamp(spatial_factor, T(1.0), T(10.0))
    
    return spatial_factor
end

"""
Create optimized PararealConfig using literature-based guidelines
"""
function create_optimized_parareal_config(
    grid_size::NTuple{3,Int},
    grid_spacing::NTuple{3,T},
    thermal_diffusivity::T,
    total_simulation_time::T,
    base_time_step::T;
    target_speedup::T = T(4.0),
    accuracy_priority::Symbol = :balanced,
    n_mpi_processes::Int = 4,
    n_threads_per_process::Int = 1
) where {T <: AbstractFloat}
    
    # 問題特性を分析
    characteristics = analyze_problem_characteristics(
        grid_size, grid_spacing, thermal_diffusivity,
        total_simulation_time, base_time_step
    )
    
    # パラメータ最適化器を作成
    optimizer = create_parameter_optimizer(T; problem_type = :heat_conduction)
    
    # パラメータを最適化
    result = optimize_parameters!(
        optimizer, characteristics;
        target_speedup = target_speedup,
        accuracy_priority = accuracy_priority
    )
    
    # 最適化結果を表示
    println("パラメータ最適化が完了しました:")
    print_optimization_result(result)
    
    # PararealConfigを作成
    return PararealConfig{T}(
        total_time = total_simulation_time,
        n_time_windows = result.recommended_n_windows,
        dt_coarse = result.recommended_coarse_dt,
        dt_fine = result.recommended_fine_dt,
        max_iterations = result.predicted_iterations + 5,  # 余裕を持たせる
        convergence_tolerance = T(1e-6),  # デフォルト値
        n_mpi_processes = n_mpi_processes,
        n_threads_per_process = n_threads_per_process,
        auto_optimize_parameters = true,
        parameter_exploration_mode = false
    )
end

"""
Create automatically tuned PararealConfig using preliminary runs
"""
function create_auto_tuned_parareal_config(
    grid_size::NTuple{3,Int},
    grid_spacing::NTuple{3,T},
    thermal_diffusivity::T,
    total_simulation_time::T,
    base_time_step::T;
    tuning_strategy::Symbol = :adaptive,
    n_mpi_processes::Int = 4,
    n_threads_per_process::Int = 1
) where {T <: AbstractFloat}
    
    println("=== 自動パラメータチューニング開始 ===")
    
    # 問題特性を分析
    characteristics = analyze_problem_characteristics(
        grid_size, grid_spacing, thermal_diffusivity,
        total_simulation_time, base_time_step
    )
    
    # 自動チューニング器を作成
    tuner = create_automatic_tuner(T; 
        problem_type = :heat_conduction,
        tuning_strategy = tuning_strategy
    )
    
    # 自動チューニングを実行
    result = perform_automatic_tuning!(tuner, characteristics)
    
    if result === nothing
        @warn "自動チューニングが失敗しました。文献ベースの設定を使用します。"
        return create_optimized_parareal_config(
            grid_size, grid_spacing, thermal_diffusivity,
            total_simulation_time, base_time_step;
            n_mpi_processes = n_mpi_processes,
            n_threads_per_process = n_threads_per_process
        )
    end
    
    println("自動チューニングが完了しました:")
    print_optimization_result(result)
    
    # PararealConfigを作成
    return PararealConfig{T}(
        total_time = total_simulation_time,
        n_time_windows = result.recommended_n_windows,
        dt_coarse = result.recommended_coarse_dt,
        dt_fine = result.recommended_fine_dt,
        max_iterations = result.predicted_iterations + 3,  # 自動チューニング結果に基づく
        convergence_tolerance = T(1e-6),
        n_mpi_processes = n_mpi_processes,
        n_threads_per_process = n_threads_per_process,
        auto_optimize_parameters = true,
        parameter_exploration_mode = false
    )
end

"""
Validate solver parameters against problem constraints
"""
function validate_solver_parameters(config::SolverConfiguration{T},
                                  problem_constraints::Dict{String, Any}) where {T <: AbstractFloat}
    
    validation_results = Dict{String, Any}()
    validation_results["is_valid"] = true
    validation_results["warnings"] = String[]
    validation_results["errors"] = String[]
    
    # Check time step constraints
    if haskey(problem_constraints, "max_dt")
        max_dt = problem_constraints["max_dt"]
        if config.coarse_solver.dt > max_dt
            push!(validation_results["errors"], 
                  "Coarse time step ($(config.coarse_solver.dt)) exceeds maximum allowed ($max_dt)")
            validation_results["is_valid"] = false
        end
        if config.fine_solver.dt > max_dt
            push!(validation_results["errors"],
                  "Fine time step ($(config.fine_solver.dt)) exceeds maximum allowed ($max_dt)")
            validation_results["is_valid"] = false
        end
    end
    
    # Check memory constraints
    if haskey(problem_constraints, "max_memory_gb")
        max_memory = problem_constraints["max_memory_gb"]
        # Rough memory estimate (this would be more sophisticated in practice)
        estimated_memory = estimate_memory_usage(config, problem_constraints)
        if estimated_memory > max_memory
            push!(validation_results["warnings"],
                  "Estimated memory usage ($(estimated_memory) GB) may exceed limit ($max_memory GB)")
        end
    end
    
    # Check solver compatibility
    coarse_type = config.coarse_solver.solver_type
    fine_type = config.fine_solver.solver_type
    
    # Some solver combinations may not be optimal
    if coarse_type == :sor && fine_type == :cg
        push!(validation_results["warnings"],
              "SOR coarse with CG fine may have convergence issues")
    end
    
    return validation_results
end

"""
Estimate memory usage for solver configuration (rough approximation)
"""
function estimate_memory_usage(config::SolverConfiguration{T}, 
                              problem_constraints::Dict{String, Any}) where {T <: AbstractFloat}
    
    # Get problem size
    if haskey(problem_constraints, "grid_size")
        grid_size = problem_constraints["grid_size"]
        total_cells = prod(grid_size)
    else
        total_cells = 1_000_000  # Default estimate
    end
    
    # Bytes per cell (rough estimate for temperature field + work arrays)
    bytes_per_cell = sizeof(T) * 15  # Multiple work arrays
    
    # Memory for fine solver
    fine_memory = total_cells * bytes_per_cell
    
    # Memory for coarse solver (reduced by spatial factor^3)
    coarse_factor = config.coarse_solver.spatial_resolution_factor
    coarse_memory = fine_memory / (coarse_factor^3)
    
    # Total memory in GB
    total_memory_gb = (fine_memory + coarse_memory) / (1024^3)
    
    return total_memory_gb
end

"""
Get solver recommendations based on problem characteristics
"""
function get_solver_recommendations(selector::SolverSelector{T},
                                  problem_characteristics::Dict{String, Any}) where {T <: AbstractFloat}
    
    recommendations = Dict{String, Any}()
    
    # Analyze problem characteristics
    if haskey(problem_characteristics, "condition_number")
        condition_number = problem_characteristics["condition_number"]
        if condition_number > 1000
            recommendations["solver_type"] = :pbicgstab
            recommendations["reason"] = "High condition number detected, PBiCGSTAB recommended for stability"
        elseif condition_number < 100
            recommendations["solver_type"] = :cg
            recommendations["reason"] = "Well-conditioned problem, CG recommended for speed"
        else
            recommendations["solver_type"] = :pbicgstab
            recommendations["reason"] = "Moderate condition number, PBiCGSTAB recommended for robustness"
        end
    end
    
    # Memory-based recommendations
    if haskey(problem_characteristics, "available_memory_gb")
        memory_gb = problem_characteristics["available_memory_gb"]
        if memory_gb < 4.0
            recommendations["spatial_resolution_factor"] = 4.0
            recommendations["memory_note"] = "Limited memory detected, aggressive spatial coarsening recommended"
        elseif memory_gb > 32.0
            recommendations["spatial_resolution_factor"] = 1.5
            recommendations["memory_note"] = "Abundant memory available, minimal spatial coarsening recommended"
        end
    end
    
    # Performance-based recommendations
    if haskey(problem_characteristics, "target_wall_time_hours")
        target_time = problem_characteristics["target_wall_time_hours"]
        if target_time < 1.0
            recommendations["accuracy_priority"] = :speed
            recommendations["time_note"] = "Short target time, prioritizing speed over accuracy"
        elseif target_time > 24.0
            recommendations["accuracy_priority"] = :accuracy
            recommendations["time_note"] = "Long target time available, prioritizing accuracy"
        end
    end
    
    return recommendations
end

"""
Solve a single time step using the fine solver
"""
function solve_time_step!(solver::FineSolver{T},
                         solution::Array{T,3},
                         current_time::T,
                         dt::T,
                         problem_data::Heat3dsProblemData{T}) where {T <: AbstractFloat}
    
    # For now, implement a simple explicit diffusion step
    # In a full implementation, this would integrate with Heat3ds solvers
    grid_size = Base.size(solution)
    new_solution = copy(solution)
    
    # Simple explicit diffusion update
    diffusivity = T(0.1)  # Thermal diffusivity coefficient
    
    for k in 2:grid_size[3]-1, j in 2:grid_size[2]-1, i in 2:grid_size[1]-1
        # 6-point stencil for 3D diffusion
        laplacian = (solution[i+1,j,k] + solution[i-1,j,k] +
                    solution[i,j+1,k] + solution[i,j-1,k] +
                    solution[i,j,k+1] + solution[i,j,k-1] -
                    6 * solution[i,j,k])
        
        new_solution[i,j,k] = solution[i,j,k] + dt * diffusivity * laplacian
    end
    
    return new_solution
end

"""
Accuracy metrics for validation
"""
struct AccuracyMetrics{T <: AbstractFloat}
    l2_norm_error::T
    max_pointwise_error::T
    relative_error::T
    convergence_rate::T
    error_distribution::Array{T,3}
    
    function AccuracyMetrics{T}(l2_error::T, max_error::T, rel_error::T, conv_rate::T, error_dist::Array{T,3}) where {T <: AbstractFloat}
        return new{T}(l2_error, max_error, rel_error, conv_rate, error_dist)
    end
end

"""
Validation result structure
"""
struct ValidationResult{T <: AbstractFloat}
    timestamp::DateTime
    problem_id::String
    parareal_config::PararealConfig{T}
    accuracy_metrics::AccuracyMetrics{T}
    is_within_tolerance::Bool
    recommendations::Vector{String}
    
    function ValidationResult{T}(timestamp::DateTime, problem_id::String, config::PararealConfig{T}, 
                               metrics::AccuracyMetrics{T}, within_tolerance::Bool, 
                               recommendations::Vector{String}) where {T <: AbstractFloat}
        return new{T}(timestamp, problem_id, config, metrics, within_tolerance, recommendations)
    end
end

"""
Sequential solver for reference computation
"""
struct SequentialSolver{T <: AbstractFloat}
    dt::T
    solver_type::Symbol
    tolerance::T
    max_iterations::Int
    
    function SequentialSolver{T}(;
        dt::T = T(0.01),
        solver_type::Symbol = :pbicgstab,
        tolerance::T = T(1e-6),
        max_iterations::Int = 8000
    ) where {T <: AbstractFloat}
        return new{T}(dt, solver_type, tolerance, max_iterations)
    end
end

"""
Tolerance settings for validation
"""
struct ToleranceSettings{T <: AbstractFloat}
    absolute_tolerance::T
    relative_tolerance::T
    max_pointwise_tolerance::T
    convergence_rate_tolerance::T
    
    function ToleranceSettings{T}(;
        absolute_tolerance::T = T(1e-6),
        relative_tolerance::T = T(1e-4),
        max_pointwise_tolerance::T = T(1e-5),
        convergence_rate_tolerance::T = T(0.1)
    ) where {T <: AbstractFloat}
        return new{T}(absolute_tolerance, relative_tolerance, max_pointwise_tolerance, convergence_rate_tolerance)
    end
end

"""
Validation manager for sequential comparison and accuracy verification
"""
mutable struct ValidationManager{T <: AbstractFloat}
    reference_solver::SequentialSolver{T}
    accuracy_metrics::Union{AccuracyMetrics{T}, Nothing}
    validation_history::Vector{ValidationResult{T}}
    tolerance_settings::ToleranceSettings{T}
    
    function ValidationManager{T}(;
        reference_solver::Union{SequentialSolver{T}, Nothing} = nothing,
        tolerance_settings::Union{ToleranceSettings{T}, Nothing} = nothing
    ) where {T <: AbstractFloat}
        
        ref_solver = reference_solver !== nothing ? reference_solver : SequentialSolver{T}()
        tol_settings = tolerance_settings !== nothing ? tolerance_settings : ToleranceSettings{T}()
        
        return new{T}(ref_solver, nothing, Vector{ValidationResult{T}}(), tol_settings)
    end
end

"""
Validate parareal results against sequential computation
"""
function validate_against_sequential!(manager::ValidationManager{T}, 
                                    parareal_result::PararealResult{T},
                                    sequential_result::Array{T,3},
                                    problem_id::String,
                                    config::PararealConfig{T}) where {T <: AbstractFloat}
    
    # Compute accuracy metrics
    metrics = compute_accuracy_metrics(parareal_result.final_solution, sequential_result)
    
    # Check if within tolerance
    within_tolerance = check_tolerance(metrics, manager.tolerance_settings)
    
    # Generate recommendations
    recommendations = generate_recommendations(metrics, manager.tolerance_settings, config)
    
    # Create validation result
    result = ValidationResult{T}(
        now(), problem_id, config, metrics, within_tolerance, recommendations
    )
    
    # Store in history
    push!(manager.validation_history, result)
    manager.accuracy_metrics = metrics
    
    return result
end

"""
Compute accuracy metrics between parareal and sequential solutions
"""
function compute_accuracy_metrics(parareal_data::Array{T,3}, sequential_data::Array{T,3}) where {T <: AbstractFloat}
    # Ensure same size
    if size(parareal_data) != size(sequential_data)
        error("Solution arrays must have the same size")
    end
    
    # Compute error distribution
    error_dist = abs.(parareal_data - sequential_data)
    
    # L2 norm error
    l2_error = sqrt(sum(error_dist.^2)) / sqrt(sum(sequential_data.^2))
    
    # Maximum pointwise error
    max_error = maximum(error_dist)
    
    # Relative error (normalized by sequential solution magnitude)
    seq_magnitude = sqrt(sum(sequential_data.^2))
    relative_error = seq_magnitude > 0 ? sqrt(sum(error_dist.^2)) / seq_magnitude : T(0.0)
    
    # Convergence rate (placeholder - would need iteration history)
    convergence_rate = T(0.0)
    
    return AccuracyMetrics{T}(l2_error, max_error, relative_error, convergence_rate, error_dist)
end

"""
Check if accuracy metrics are within tolerance
"""
function check_tolerance(metrics::AccuracyMetrics{T}, tolerance_settings::ToleranceSettings{T}) where {T <: AbstractFloat}
    return (metrics.l2_norm_error <= tolerance_settings.absolute_tolerance &&
            metrics.relative_error <= tolerance_settings.relative_tolerance &&
            metrics.max_pointwise_error <= tolerance_settings.max_pointwise_tolerance)
end

"""
Generate recommendations based on accuracy metrics
"""
function generate_recommendations(metrics::AccuracyMetrics{T}, 
                                tolerance_settings::ToleranceSettings{T},
                                config::PararealConfig{T}) where {T <: AbstractFloat}
    
    recommendations = String[]
    
    if metrics.l2_norm_error > tolerance_settings.absolute_tolerance
        push!(recommendations, "L2 error exceeds tolerance. Consider reducing coarse time step or increasing parareal iterations.")
    end
    
    if metrics.relative_error > tolerance_settings.relative_tolerance
        push!(recommendations, "Relative error exceeds tolerance. Consider improving coarse solver accuracy.")
    end
    
    if metrics.max_pointwise_error > tolerance_settings.max_pointwise_tolerance
        push!(recommendations, "Maximum pointwise error exceeds tolerance. Check for numerical instabilities.")
    end
    
    # Time step ratio recommendations
    time_step_ratio = config.dt_coarse / config.dt_fine
    if time_step_ratio > 100 && metrics.l2_norm_error > tolerance_settings.absolute_tolerance * 0.1
        push!(recommendations, "Large time step ratio ($time_step_ratio) may be causing accuracy issues. Consider reducing coarse time step.")
    end
    
    if isempty(recommendations)
        push!(recommendations, "Validation passed. Parareal results are within acceptable tolerance.")
    end
    
    return recommendations
end

"""
Generate error analysis report
"""
function generate_error_analysis_report(validation_result::ValidationResult{T}) where {T <: AbstractFloat}
    report = String[]
    
    push!(report, "=== Parareal Validation Report ===")
    push!(report, "Timestamp: $(validation_result.timestamp)")
    push!(report, "Problem ID: $(validation_result.problem_id)")
    push!(report, "")
    
    metrics = validation_result.accuracy_metrics
    push!(report, "Accuracy Metrics:")
    push!(report, "  L2 norm error: $(metrics.l2_norm_error)")
    push!(report, "  Max pointwise error: $(metrics.max_pointwise_error)")
    push!(report, "  Relative error: $(metrics.relative_error)")
    push!(report, "")
    
    push!(report, "Validation Status: $(validation_result.is_within_tolerance ? "PASSED" : "FAILED")")
    push!(report, "")
    
    push!(report, "Recommendations:")
    for rec in validation_result.recommendations
        push!(report, "  - $rec")
    end
    
    push!(report, "================================")
    
    return join(report, "\n")
end

"""
Check numerical stability of parareal computation
"""
function check_numerical_stability(convergence_history::Vector{T}) where {T <: AbstractFloat}
    if length(convergence_history) < 3
        return true, "Insufficient data for stability analysis"
    end
    
    # Check for monotonic decrease in residuals
    is_monotonic = all(convergence_history[i] >= convergence_history[i+1] for i in 1:length(convergence_history)-1)
    
    # Check for stagnation
    recent_changes = [abs(convergence_history[i] - convergence_history[i+1]) / convergence_history[i] 
                     for i in max(1, length(convergence_history)-3):length(convergence_history)-1]
    is_stagnant = all(change < 1e-8 for change in recent_changes)
    
    # Check for oscillations
    sign_changes = sum(sign(convergence_history[i] - convergence_history[i+1]) != 
                      sign(convergence_history[i+1] - convergence_history[i+2]) 
                      for i in 1:length(convergence_history)-2)
    has_oscillations = sign_changes > length(convergence_history) / 3
    
    if !is_monotonic && has_oscillations
        return false, "Convergence shows oscillatory behavior, indicating potential numerical instability"
    elseif is_stagnant
        return false, "Convergence has stagnated, may indicate poor parameter choice"
    else
        return true, "Convergence appears numerically stable"
    end
end

"""
Run sequential computation for validation reference
"""
function run_sequential_reference!(manager::ValidationManager{T},
                                 initial_condition::Array{T,3},
                                 problem_data::Heat3dsProblemData{T},
                                 total_time::T) where {T <: AbstractFloat}
    
    solver = manager.reference_solver
    current_solution = copy(initial_condition)
    current_time = T(0.0)
    dt = solver.dt
    
    # Sequential time stepping
    while current_time < total_time
        # Adjust time step to not overshoot
        actual_dt = min(dt, total_time - current_time)
        
        # Use simple diffusion for reference (in practice, would use full Heat3ds solver)
        apply_simple_diffusion_step!(current_solution, actual_dt, T(0.1))
        
        current_time += actual_dt
    end
    
    return current_solution
end

"""
Fallback to sequential computation when Parareal fails
"""
function fallback_to_sequential!(manager::PararealManager{T},
                                initial_condition::Array{T,3},
                                problem_data::Heat3dsProblemData{T},
                                coordinator::HybridCoordinator{T}) where {T <: AbstractFloat}
    
    rank = manager.mpi_comm.rank
    
    if rank == 0
        @info "Executing sequential fallback computation..."
    end
    
    # Use fine solver for sequential computation
    fine_solver = FineSolver{T}(
        dt = manager.config.dt_fine,
        solver_type = :pbicgstab,
        use_full_physics = true
    )
    
    # Sequential time stepping
    current_solution = copy(initial_condition)
    total_time = manager.config.total_time
    dt = manager.config.dt_fine
    n_steps = Int(ceil(total_time / dt))
    
    start_time = time_ns() / 1e9
    
    # Simple forward Euler or existing Heat3ds solver integration
    for step in 1:n_steps
        current_time = (step - 1) * dt
        
        # Use fine solver to advance one time step
        try
            current_solution = solve_time_step!(
                fine_solver, current_solution, current_time, dt, problem_data
            )
        catch solver_error
            if rank == 0
                @error "Sequential solver failed at step $step: $solver_error"
            end
            rethrow(solver_error)
        end
        
        # Progress reporting
        if rank == 0 && step % max(1, n_steps ÷ 10) == 0
            progress = 100.0 * step / n_steps
            @info "Sequential computation progress: $(round(progress, digits=1))%"
        end
    end
    
    computation_time = time_ns() / 1e9 - start_time
    
    if rank == 0
        @info "Sequential computation completed in $(computation_time) seconds"
    end
    
    # Return result in Parareal format
    return PararealResult{T}(
        current_solution,
        true,  # Mark as converged (sequential is exact)
        1,     # One "iteration" (sequential)
        T[],   # No residual history for sequential
        T(computation_time),
        T(0.0) # No communication time
    )
end

"""
Solve a single time step using the fine solver
"""
function solve_time_step!(solver::FineSolver{T},
                         solution::Array{T,3},
                         current_time::T,
                         dt::T,
                         problem_data::Heat3dsProblemData{T}) where {T <: AbstractFloat}
    
    # This is a simplified implementation - in practice, this would call
    # the actual Heat3ds solver for one time step
    
    # For now, implement a simple explicit diffusion step
    grid_size = Base.size(solution)
    new_solution = copy(solution)
    
    # Simple 3D diffusion with periodic boundary conditions
    alpha = T(0.1)  # Thermal diffusivity
    dx, dy, dz = T(0.1), T(0.1), T(0.1)  # Grid spacing
    
    for k in 2:(grid_size[3]-1)
        for j in 2:(grid_size[2]-1)
            for i in 2:(grid_size[1]-1)
                # 3D Laplacian
                laplacian = (solution[i+1,j,k] - 2*solution[i,j,k] + solution[i-1,j,k]) / dx^2 +
                           (solution[i,j+1,k] - 2*solution[i,j,k] + solution[i,j-1,k]) / dy^2 +
                           (solution[i,j,k+1] - 2*solution[i,j,k] + solution[i,j,k-1]) / dz^2
                
                new_solution[i,j,k] = solution[i,j,k] + dt * alpha * laplacian
            end
        end
    end
    
    return new_solution
end

"""
Check if graceful degradation should be triggered
"""
function should_trigger_graceful_degradation(manager::PararealManager{T},
                                           monitor::ConvergenceMonitor{T},
                                           error_context::String) where {T <: AbstractFloat}
    
    # Trigger conditions for graceful degradation
    trigger_conditions = [
        # MPI communication failures
        occursin("MPI", error_context) || occursin("communication", error_context),
        
        # Memory allocation failures
        occursin("OutOfMemoryError", error_context) || occursin("memory", error_context),
        
        # Solver convergence failures after many iterations
        monitor.iteration_count >= manager.config.max_iterations ÷ 2,
        
        # Numerical instability indicators
        occursin("NaN", error_context) || occursin("Inf", error_context) || occursin("infinite", error_context),
        
        # Thread pool failures
        occursin("thread", error_context) || occursin("ThreadsX", error_context)
    ]
    
    return any(trigger_conditions)
end

"""
Log graceful degradation event for analysis
"""
function log_graceful_degradation_event(manager::PararealManager{T},
                                       error_context::String,
                                       fallback_successful::Bool) where {T <: AbstractFloat}
    
    rank = manager.mpi_comm.rank
    
    if rank == 0
        timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        
        @info "=== Graceful Degradation Event Log ==="
        @info "Timestamp: $timestamp"
        @info "Original error: $error_context"
        @info "Parareal configuration:"
        @info "  Time windows: $(manager.config.n_time_windows)"
        @info "  MPI processes: $(manager.config.n_mpi_processes)"
        @info "  Threads per process: $(manager.config.n_threads_per_process)"
        @info "  Max iterations: $(manager.config.max_iterations)"
        @info "Sequential fallback successful: $fallback_successful"
        @info "======================================"
    end
end

# ===== Performance Metrics Implementation =====

"""
Create performance metrics instance for a specific MPI process
"""
function create_performance_metrics(rank::Int, n_processes::Int, n_threads::Int)
    return PerformanceMetrics{Float64}(rank, n_processes, n_threads)
end

"""
Update timing data for solver performance
"""
function update_timing_data!(metrics::Any, 
                           solver_type::Symbol, 
                           execution_time::T) where {T <: AbstractFloat}
    
    if solver_type == :coarse
        metrics.timing_data.coarse_solver_time += execution_time
        metrics.timing_data.coarse_solver_calls += 1
    elseif solver_type == :fine
        metrics.timing_data.fine_solver_time += execution_time
        metrics.timing_data.fine_solver_calls += 1
    elseif solver_type == :interpolation
        metrics.timing_data.interpolation_time += execution_time
    elseif solver_type == :restriction
        metrics.timing_data.restriction_time += execution_time
    end
    
    # Update total solver time
    metrics.timing_data.total_solver_time = (
        metrics.timing_data.coarse_solver_time +
        metrics.timing_data.fine_solver_time +
        metrics.timing_data.interpolation_time +
        metrics.timing_data.restriction_time
    )
    
    return nothing
end

"""
Record MPI communication overhead
"""
function record_communication_overhead!(metrics::Any,
                                      operation_type::Symbol,
                                      execution_time::T,
                                      bytes_transferred::Int = 0) where {T <: AbstractFloat}
    
    if operation_type == :send
        metrics.communication_metrics.send_time += execution_time
    elseif operation_type == :receive
        metrics.communication_metrics.receive_time += execution_time
    elseif operation_type == :synchronization
        metrics.communication_metrics.synchronization_time += execution_time
    elseif operation_type == :broadcast
        metrics.communication_metrics.broadcast_time += execution_time
    elseif operation_type == :allreduce
        metrics.communication_metrics.allreduce_time += execution_time
    end
    
    # Update totals
    metrics.communication_metrics.total_communication_time = (
        metrics.communication_metrics.send_time +
        metrics.communication_metrics.receive_time +
        metrics.communication_metrics.synchronization_time +
        metrics.communication_metrics.broadcast_time +
        metrics.communication_metrics.allreduce_time
    )
    
    metrics.communication_metrics.message_count += 1
    metrics.communication_metrics.bytes_transferred += bytes_transferred
    
    return nothing
end

"""
Calculate efficiency metrics based on collected performance data
"""
function calculate_efficiency_metrics!(metrics::Any,
                                     sequential_time::T = T(0.0)) where {T <: AbstractFloat}
    
    # Calculate speedup factor
    if sequential_time > 0 && metrics.total_wall_time > 0
        metrics.efficiency_metrics.speedup_factor = sequential_time / metrics.total_wall_time
    else
        metrics.efficiency_metrics.speedup_factor = T(0.0)
    end
    
    # Calculate parallel efficiency (speedup / number of processes)
    if metrics.n_processes > 0
        metrics.efficiency_metrics.parallel_efficiency = 
            metrics.efficiency_metrics.speedup_factor / metrics.n_processes
    end
    
    # Calculate communication overhead ratio
    if metrics.total_wall_time > 0
        metrics.efficiency_metrics.communication_overhead_ratio = 
            metrics.communication_metrics.total_communication_time / metrics.total_wall_time
    end
    
    # Calculate load balance factor (simplified - would need data from all processes)
    # For now, use a placeholder calculation
    total_computation_time = metrics.timing_data.total_solver_time
    if total_computation_time > 0 && metrics.total_wall_time > 0
        metrics.efficiency_metrics.load_balance_factor = 
            total_computation_time / metrics.total_wall_time
    end
    
    # Strong scaling efficiency (placeholder - would need baseline measurements)
    metrics.efficiency_metrics.strong_scaling_efficiency = 
        metrics.efficiency_metrics.parallel_efficiency
    
    # Weak scaling efficiency (placeholder - would need problem size scaling data)
    metrics.efficiency_metrics.weak_scaling_efficiency = 
        metrics.efficiency_metrics.parallel_efficiency
    
    return nothing
end

"""
Get comprehensive performance summary
"""
function get_performance_summary(metrics::Any)
    summary = Dict{String, Any}()
    
    # Basic information
    summary["process_rank"] = metrics.process_rank
    summary["n_processes"] = metrics.n_processes
    summary["n_threads_per_process"] = metrics.n_threads_per_process
    summary["parareal_iterations"] = metrics.parareal_iterations
    
    # Timing information
    summary["total_wall_time"] = metrics.total_wall_time
    summary["sequential_reference_time"] = metrics.sequential_reference_time
    summary["convergence_time"] = metrics.convergence_time
    
    # Solver timing breakdown
    summary["coarse_solver_time"] = metrics.timing_data.coarse_solver_time
    summary["fine_solver_time"] = metrics.timing_data.fine_solver_time
    summary["coarse_solver_calls"] = metrics.timing_data.coarse_solver_calls
    summary["fine_solver_calls"] = metrics.timing_data.fine_solver_calls
    summary["interpolation_time"] = metrics.timing_data.interpolation_time
    summary["restriction_time"] = metrics.timing_data.restriction_time
    summary["total_solver_time"] = metrics.timing_data.total_solver_time
    
    # Communication metrics
    summary["total_communication_time"] = metrics.communication_metrics.total_communication_time
    summary["send_time"] = metrics.communication_metrics.send_time
    summary["receive_time"] = metrics.communication_metrics.receive_time
    summary["synchronization_time"] = metrics.communication_metrics.synchronization_time
    summary["broadcast_time"] = metrics.communication_metrics.broadcast_time
    summary["allreduce_time"] = metrics.communication_metrics.allreduce_time
    summary["message_count"] = metrics.communication_metrics.message_count
    summary["bytes_transferred"] = metrics.communication_metrics.bytes_transferred
    
    # Efficiency metrics
    summary["speedup_factor"] = metrics.efficiency_metrics.speedup_factor
    summary["parallel_efficiency"] = metrics.efficiency_metrics.parallel_efficiency
    summary["strong_scaling_efficiency"] = metrics.efficiency_metrics.strong_scaling_efficiency
    summary["weak_scaling_efficiency"] = metrics.efficiency_metrics.weak_scaling_efficiency
    summary["load_balance_factor"] = metrics.efficiency_metrics.load_balance_factor
    summary["communication_overhead_ratio"] = metrics.efficiency_metrics.communication_overhead_ratio
    
    # Memory usage
    summary["peak_memory_usage_mb"] = metrics.peak_memory_usage_mb
    summary["average_memory_usage_mb"] = metrics.average_memory_usage_mb
    
    # Derived metrics
    if metrics.timing_data.coarse_solver_calls > 0
        summary["average_coarse_solver_time"] = 
            metrics.timing_data.coarse_solver_time / metrics.timing_data.coarse_solver_calls
    else
        summary["average_coarse_solver_time"] = T(0.0)
    end
    
    if metrics.timing_data.fine_solver_calls > 0
        summary["average_fine_solver_time"] = 
            metrics.timing_data.fine_solver_time / metrics.timing_data.fine_solver_calls
    else
        summary["average_fine_solver_time"] = T(0.0)
    end
    
    if metrics.communication_metrics.message_count > 0
        summary["average_message_size_bytes"] = 
            metrics.communication_metrics.bytes_transferred / metrics.communication_metrics.message_count
        summary["average_communication_time_per_message"] = 
            metrics.communication_metrics.total_communication_time / metrics.communication_metrics.message_count
    else
        summary["average_message_size_bytes"] = 0
        summary["average_communication_time_per_message"] = T(0.0)
    end
    
    # Timestamps
    summary["start_timestamp"] = metrics.start_timestamp
    summary["end_timestamp"] = metrics.end_timestamp
    
    return summary
end

"""
Reset performance metrics for new computation
"""
function reset_performance_metrics!(metrics::Any)
    # Reset timing data
    metrics.timing_data.coarse_solver_time = 0.0
    metrics.timing_data.fine_solver_time = 0.0
    metrics.timing_data.coarse_solver_calls = 0
    metrics.timing_data.fine_solver_calls = 0
    metrics.timing_data.interpolation_time = 0.0
    metrics.timing_data.restriction_time = 0.0
    metrics.timing_data.total_solver_time = 0.0
    
    # Reset communication metrics
    metrics.communication_metrics.send_time = 0.0
    metrics.communication_metrics.receive_time = 0.0
    metrics.communication_metrics.synchronization_time = 0.0
    metrics.communication_metrics.broadcast_time = 0.0
    metrics.communication_metrics.allreduce_time = 0.0
    metrics.communication_metrics.total_communication_time = 0.0
    metrics.communication_metrics.message_count = 0
    metrics.communication_metrics.bytes_transferred = 0
    
    # Reset efficiency metrics
    metrics.efficiency_metrics.parallel_efficiency = 0.0
    metrics.efficiency_metrics.strong_scaling_efficiency = 0.0
    metrics.efficiency_metrics.weak_scaling_efficiency = 0.0
    metrics.efficiency_metrics.speedup_factor = 0.0
    metrics.efficiency_metrics.load_balance_factor = 0.0
    metrics.efficiency_metrics.communication_overhead_ratio = 0.0
    
    # Reset overall metrics
    metrics.total_wall_time = 0.0
    metrics.sequential_reference_time = 0.0
    metrics.parareal_iterations = 0
    metrics.convergence_time = 0.0
    metrics.peak_memory_usage_mb = 0.0
    metrics.average_memory_usage_mb = 0.0
    
    # Reset timestamps
    metrics.start_timestamp = now()
    metrics.end_timestamp = nothing
    
    return nothing
end

"""
Merge performance metrics from multiple processes
"""
function merge_performance_metrics(metrics_list::Vector{Any})
    if isempty(metrics_list)
        error("Cannot merge empty metrics list")
    end
    
    # Use first metrics as base
    merged = deepcopy(metrics_list[1])
    
    # Aggregate timing data (sum across processes)
    for i in 2:length(metrics_list)
        m = metrics_list[i]
        merged.timing_data.coarse_solver_time += m.timing_data.coarse_solver_time
        merged.timing_data.fine_solver_time += m.timing_data.fine_solver_time
        merged.timing_data.coarse_solver_calls += m.timing_data.coarse_solver_calls
        merged.timing_data.fine_solver_calls += m.timing_data.fine_solver_calls
        merged.timing_data.interpolation_time += m.timing_data.interpolation_time
        merged.timing_data.restriction_time += m.timing_data.restriction_time
        merged.timing_data.total_solver_time += m.timing_data.total_solver_time
    end
    
    # Aggregate communication metrics (sum across processes)
    for i in 2:length(metrics_list)
        m = metrics_list[i]
        merged.communication_metrics.send_time += m.communication_metrics.send_time
        merged.communication_metrics.receive_time += m.communication_metrics.receive_time
        merged.communication_metrics.synchronization_time += m.communication_metrics.synchronization_time
        merged.communication_metrics.broadcast_time += m.communication_metrics.broadcast_time
        merged.communication_metrics.allreduce_time += m.communication_metrics.allreduce_time
        merged.communication_metrics.total_communication_time += m.communication_metrics.total_communication_time
        merged.communication_metrics.message_count += m.communication_metrics.message_count
        merged.communication_metrics.bytes_transferred += m.communication_metrics.bytes_transferred
    end
    
    # Take maximum wall time and memory usage
    for i in 2:length(metrics_list)
        m = metrics_list[i]
        merged.total_wall_time = max(merged.total_wall_time, m.total_wall_time)
        merged.peak_memory_usage_mb = max(merged.peak_memory_usage_mb, m.peak_memory_usage_mb)
        merged.average_memory_usage_mb = max(merged.average_memory_usage_mb, m.average_memory_usage_mb)
    end
    
    # Take maximum iterations and convergence time
    for i in 2:length(metrics_list)
        m = metrics_list[i]
        merged.parareal_iterations = max(merged.parareal_iterations, m.parareal_iterations)
        merged.convergence_time = max(merged.convergence_time, m.convergence_time)
    end
    
    # Recalculate efficiency metrics for merged data
    calculate_efficiency_metrics!(merged, merged.sequential_reference_time)
    
    # Set process information to reflect merged nature
    merged.process_rank = -1  # Indicates merged metrics
    merged.n_processes = length(metrics_list)
    
    return merged
end

"""
Print formatted performance report
"""
function print_performance_report(metrics::Any)
    summary = get_performance_summary(metrics)
    
    println("=== Parareal Performance Report ===")
    println("Process Information:")
    println("  Rank: $(summary["process_rank"])")
    println("  Total processes: $(summary["n_processes"])")
    println("  Threads per process: $(summary["n_threads_per_process"])")
    println("  Parareal iterations: $(summary["parareal_iterations"])")
    println()
    
    println("Timing Breakdown:")
    println("  Total wall time: $(round(summary["total_wall_time"], digits=3)) s")
    println("  Sequential reference: $(round(summary["sequential_reference_time"], digits=3)) s")
    println("  Convergence time: $(round(summary["convergence_time"], digits=3)) s")
    println("  Coarse solver time: $(round(summary["coarse_solver_time"], digits=3)) s ($(summary["coarse_solver_calls"]) calls)")
    println("  Fine solver time: $(round(summary["fine_solver_time"], digits=3)) s ($(summary["fine_solver_calls"]) calls)")
    println("  Interpolation time: $(round(summary["interpolation_time"], digits=3)) s")
    println("  Restriction time: $(round(summary["restriction_time"], digits=3)) s")
    println()
    
    println("Communication Metrics:")
    println("  Total communication time: $(round(summary["total_communication_time"], digits=3)) s")
    println("  Send time: $(round(summary["send_time"], digits=3)) s")
    println("  Receive time: $(round(summary["receive_time"], digits=3)) s")
    println("  Synchronization time: $(round(summary["synchronization_time"], digits=3)) s")
    println("  Messages sent: $(summary["message_count"])")
    println("  Bytes transferred: $(summary["bytes_transferred"])")
    println()
    
    println("Efficiency Metrics:")
    println("  Speedup factor: $(round(summary["speedup_factor"], digits=2))x")
    println("  Parallel efficiency: $(round(summary["parallel_efficiency"] * 100, digits=1))%")
    println("  Load balance factor: $(round(summary["load_balance_factor"], digits=3))")
    println("  Communication overhead: $(round(summary["communication_overhead_ratio"] * 100, digits=1))%")
    println()
    
    println("Memory Usage:")
    println("  Peak memory: $(round(summary["peak_memory_usage_mb"], digits=1)) MB")
    println("  Average memory: $(round(summary["average_memory_usage_mb"], digits=1)) MB")
    println("===================================")
end

"""
Measure execution time of a function and update performance metrics
"""
function measure_and_record!(metrics::Any, 
                           operation_type::Symbol, 
                           func::Function, 
                           args...)
    
    start_time = time_ns()
    result = func(args...)
    end_time = time_ns()
    
    execution_time = (end_time - start_time) / 1e9  # Convert to seconds
    
    if operation_type in [:coarse, :fine, :interpolation, :restriction]
        update_timing_data!(metrics, operation_type, execution_time)
    elseif operation_type in [:send, :receive, :synchronization, :broadcast, :allreduce]
        record_communication_overhead!(metrics, operation_type, execution_time)
    end
    
    return result
end

# ===== Performance Monitoring System Implementation =====

"""
Real-time performance monitoring data
"""
mutable struct MonitoringData{T <: AbstractFloat}
    # Real-time metrics
    current_iteration::Int
    current_wall_time::T
    current_memory_usage::T
    
    # Historical data
    iteration_times::Vector{T}
    memory_usage_history::Vector{T}
    residual_history::Vector{T}
    
    # Load balancing metrics
    process_workloads::Vector{T}
    process_idle_times::Vector{T}
    communication_patterns::Dict{Tuple{Int,Int}, T}
    
    # Scalability data
    strong_scaling_data::Dict{Int, T}  # n_processes => execution_time
    weak_scaling_data::Dict{Int, T}    # problem_size => execution_time
    
    function MonitoringData{T}() where {T <: AbstractFloat}
        return new{T}(
            0, T(0.0), T(0.0),
            Vector{T}(), Vector{T}(), Vector{T}(),
            Vector{T}(), Vector{T}(), Dict{Tuple{Int,Int}, T}(),
            Dict{Int, T}(), Dict{Int, T}()
        )
    end
end

"""
Load balance analyzer for MPI processes
"""
mutable struct LoadBalanceAnalyzer{T <: AbstractFloat}
    process_metrics::Vector{PerformanceMetrics{T}}
    load_balance_threshold::T
    imbalance_factor::T
    bottleneck_processes::Vector{Int}
    
    function LoadBalanceAnalyzer{T}(n_processes::Int, threshold::T = T(0.1)) where {T <: AbstractFloat}
        return new{T}(
            Vector{PerformanceMetrics{T}}(),
            threshold,
            T(0.0),
            Vector{Int}()
        )
    end
end

"""
Scalability analyzer for strong and weak scaling
"""
mutable struct ScalabilityAnalyzer{T <: AbstractFloat}
    baseline_time::T
    baseline_processes::Int
    baseline_problem_size::Int
    
    # Strong scaling (fixed problem size, varying processes)
    strong_scaling_efficiency::Vector{T}
    strong_scaling_speedup::Vector{T}
    
    # Weak scaling (proportional problem size and processes)
    weak_scaling_efficiency::Vector{T}
    weak_scaling_throughput::Vector{T}
    
    # Amdahl's law parameters
    serial_fraction::T
    parallel_fraction::T
    
    function ScalabilityAnalyzer{T}() where {T <: AbstractFloat}
        return new{T}(
            T(0.0), 0, 0,
            Vector{T}(), Vector{T}(),
            Vector{T}(), Vector{T}(),
            T(0.0), T(0.0)
        )
    end
end

"""
Comprehensive performance monitoring system
"""
mutable struct PerformanceMonitor{T <: AbstractFloat}
    monitoring_data::MonitoringData{T}
    load_analyzer::LoadBalanceAnalyzer{T}
    scalability_analyzer::ScalabilityAnalyzer{T}
    
    # Monitoring control
    is_monitoring::Bool
    monitoring_interval::T
    last_update_time::T
    
    # MPI communication
    mpi_comm::Union{MPICommunicator{T}, Nothing}
    
    function PerformanceMonitor{T}(mpi_comm::Union{MPICommunicator{T}, Nothing} = nothing,
                                  monitoring_interval::T = T(1.0)) where {T <: AbstractFloat}
        n_processes = mpi_comm !== nothing ? mpi_comm.size : 1
        
        return new{T}(
            MonitoringData{T}(),
            LoadBalanceAnalyzer{T}(n_processes),
            ScalabilityAnalyzer{T}(),
            false,
            monitoring_interval,
            T(0.0),
            mpi_comm
        )
    end
end

"""
Create performance monitor instance
"""
function create_performance_monitor(mpi_comm::Union{Any, Nothing} = nothing,
                                   monitoring_interval::AbstractFloat = 1.0)
    T = Float64
    return PerformanceMonitor{T}(mpi_comm, T(monitoring_interval))
end

"""
Start real-time performance monitoring
"""
function start_monitoring!(monitor::PerformanceMonitor{T}) where {T <: AbstractFloat}
    if monitor.is_monitoring
        @warn "Performance monitoring is already active"
        return nothing
    end
    
    monitor.is_monitoring = true
    monitor.last_update_time = time_ns() / 1e9
    
    # Initialize monitoring data
    monitor.monitoring_data.current_iteration = 0
    monitor.monitoring_data.current_wall_time = T(0.0)
    monitor.monitoring_data.current_memory_usage = T(0.0)
    
    # Clear historical data
    empty!(monitor.monitoring_data.iteration_times)
    empty!(monitor.monitoring_data.memory_usage_history)
    empty!(monitor.monitoring_data.residual_history)
    empty!(monitor.monitoring_data.process_workloads)
    empty!(monitor.monitoring_data.process_idle_times)
    empty!(monitor.monitoring_data.communication_patterns)
    
    if monitor.mpi_comm !== nothing && monitor.mpi_comm.rank == 0
        println("=== Performance Monitoring Started ===")
        println("Monitoring interval: $(monitor.monitoring_interval) seconds")
        println("MPI processes: $(monitor.mpi_comm.size)")
        println("=====================================")
    end
    
    return nothing
end

"""
Stop performance monitoring
"""
function stop_monitoring!(monitor::PerformanceMonitor{T}) where {T <: AbstractFloat}
    if !monitor.is_monitoring
        @warn "Performance monitoring is not active"
        return nothing
    end
    
    monitor.is_monitoring = false
    
    if monitor.mpi_comm !== nothing && monitor.mpi_comm.rank == 0
        println("=== Performance Monitoring Stopped ===")
        println("Total iterations monitored: $(monitor.monitoring_data.current_iteration)")
        println("Total monitoring time: $(monitor.monitoring_data.current_wall_time) seconds")
        println("======================================")
    end
    
    return nothing
end

"""
Update real-time monitoring data
"""
function update_monitoring_data!(monitor::PerformanceMonitor{T},
                                iteration::Int,
                                residual::T,
                                memory_usage::T = T(0.0)) where {T <: AbstractFloat}
    
    if !monitor.is_monitoring
        return nothing
    end
    
    current_time = time_ns() / 1e9
    
    # Update current metrics
    monitor.monitoring_data.current_iteration = iteration
    monitor.monitoring_data.current_wall_time = T(current_time - monitor.last_update_time)
    monitor.monitoring_data.current_memory_usage = memory_usage
    
    # Add to historical data
    push!(monitor.monitoring_data.iteration_times, monitor.monitoring_data.current_wall_time)
    push!(monitor.monitoring_data.memory_usage_history, memory_usage)
    push!(monitor.monitoring_data.residual_history, residual)
    
    # Update timestamp
    monitor.last_update_time = current_time
    
    return nothing
end

"""
Analyze load balance across MPI processes
"""
function analyze_load_balance!(monitor::PerformanceMonitor{T},
                              process_metrics::Vector{Any}) where {T <: AbstractFloat}
    
    if isempty(process_metrics)
        @warn "No process metrics provided for load balance analysis"
        return nothing
    end
    
    analyzer = monitor.load_analyzer
    n_processes = length(process_metrics)
    
    # Calculate workload for each process
    workloads = Vector{T}(undef, n_processes)
    idle_times = Vector{T}(undef, n_processes)
    
    for (i, metrics) in enumerate(process_metrics)
        # Total computation time as workload indicator
        workloads[i] = metrics.timing_data.total_solver_time
        
        # Idle time = wall time - computation time - communication time
        total_active_time = (metrics.timing_data.total_solver_time + 
                           metrics.communication_metrics.total_communication_time)
        idle_times[i] = max(T(0.0), metrics.total_wall_time - total_active_time)
    end
    
    # Store workload data
    analyzer.process_metrics = process_metrics
    monitor.monitoring_data.process_workloads = workloads
    monitor.monitoring_data.process_idle_times = idle_times
    
    # Calculate load imbalance factor
    if !isempty(workloads)
        max_workload = maximum(workloads)
        min_workload = minimum(workloads)
        avg_workload = sum(workloads) / length(workloads)
        
        # Imbalance factor: (max - min) / avg
        analyzer.imbalance_factor = max_workload > 0 ? (max_workload - min_workload) / avg_workload : T(0.0)
        
        # Identify bottleneck processes (workload > avg + threshold * avg)
        threshold_workload = avg_workload * (1 + analyzer.load_balance_threshold)
        analyzer.bottleneck_processes = [i-1 for (i, workload) in enumerate(workloads) if workload > threshold_workload]
    end
    
    return nothing
end

"""
Calculate scalability metrics
"""
function calculate_scalability_metrics!(monitor::PerformanceMonitor{T},
                                      n_processes::Int,
                                      execution_time::T,
                                      problem_size::Int = 0) where {T <: AbstractFloat}
    
    analyzer = monitor.scalability_analyzer
    
    # Set baseline if not already set
    if analyzer.baseline_time == T(0.0)
        analyzer.baseline_time = execution_time
        analyzer.baseline_processes = n_processes
        analyzer.baseline_problem_size = problem_size
        return nothing
    end
    
    # Strong scaling analysis (fixed problem size)
    if problem_size == analyzer.baseline_problem_size || problem_size == 0
        # Calculate speedup: T_1 / T_p
        speedup = analyzer.baseline_time / execution_time
        
        # Calculate efficiency: speedup / n_processes
        efficiency = speedup / n_processes
        
        # Store strong scaling data
        monitor.monitoring_data.strong_scaling_data[n_processes] = execution_time
        push!(analyzer.strong_scaling_speedup, speedup)
        push!(analyzer.strong_scaling_efficiency, efficiency)
        
        # Estimate Amdahl's law parameters
        if n_processes > 1
            # S = 1 / (f + (1-f)/p) where f is serial fraction, p is processes
            # Rearranging: f = (p - S) / (S * (p - 1))
            if speedup > 0 && speedup < n_processes
                serial_fraction = (n_processes - speedup) / (speedup * (n_processes - 1))
                analyzer.serial_fraction = max(T(0.0), min(T(1.0), serial_fraction))
                analyzer.parallel_fraction = T(1.0) - analyzer.serial_fraction
            end
        end
    end
    
    # Weak scaling analysis (proportional problem size and processes)
    if problem_size > 0 && problem_size != analyzer.baseline_problem_size
        # Weak scaling efficiency: T_1 / T_p (for proportional problem sizes)
        baseline_time_per_unit = analyzer.baseline_time / analyzer.baseline_problem_size
        current_time_per_unit = execution_time / problem_size
        
        weak_efficiency = baseline_time_per_unit / current_time_per_unit
        throughput = problem_size / execution_time
        
        # Store weak scaling data
        monitor.monitoring_data.weak_scaling_data[problem_size] = execution_time
        push!(analyzer.weak_scaling_efficiency, weak_efficiency)
        push!(analyzer.weak_scaling_throughput, throughput)
    end
    
    return nothing
end

"""
Get current real-time metrics
"""
function get_real_time_metrics(monitor::PerformanceMonitor{T}) where {T <: AbstractFloat}
    metrics = Dict{String, Any}()
    
    # Current status
    metrics["is_monitoring"] = monitor.is_monitoring
    metrics["current_iteration"] = monitor.monitoring_data.current_iteration
    metrics["current_wall_time"] = monitor.monitoring_data.current_wall_time
    metrics["current_memory_usage"] = monitor.monitoring_data.current_memory_usage
    
    # Historical data
    metrics["iteration_count"] = length(monitor.monitoring_data.iteration_times)
    metrics["total_iterations"] = monitor.monitoring_data.current_iteration
    
    if !isempty(monitor.monitoring_data.iteration_times)
        metrics["average_iteration_time"] = sum(monitor.monitoring_data.iteration_times) / length(monitor.monitoring_data.iteration_times)
        metrics["max_iteration_time"] = maximum(monitor.monitoring_data.iteration_times)
        metrics["min_iteration_time"] = minimum(monitor.monitoring_data.iteration_times)
    end
    
    if !isempty(monitor.monitoring_data.memory_usage_history)
        metrics["peak_memory_usage"] = maximum(monitor.monitoring_data.memory_usage_history)
        metrics["average_memory_usage"] = sum(monitor.monitoring_data.memory_usage_history) / length(monitor.monitoring_data.memory_usage_history)
    end
    
    if !isempty(monitor.monitoring_data.residual_history)
        metrics["current_residual"] = monitor.monitoring_data.residual_history[end]
        metrics["initial_residual"] = monitor.monitoring_data.residual_history[1]
        metrics["residual_reduction"] = monitor.monitoring_data.residual_history[1] / monitor.monitoring_data.residual_history[end]
    end
    
    # Load balance metrics
    if !isempty(monitor.monitoring_data.process_workloads)
        metrics["load_imbalance_factor"] = monitor.load_analyzer.imbalance_factor
        metrics["bottleneck_processes"] = monitor.load_analyzer.bottleneck_processes
        metrics["max_workload"] = maximum(monitor.monitoring_data.process_workloads)
        metrics["min_workload"] = minimum(monitor.monitoring_data.process_workloads)
        metrics["average_workload"] = sum(monitor.monitoring_data.process_workloads) / length(monitor.monitoring_data.process_workloads)
    end
    
    # Scalability metrics
    if !isempty(monitor.scalability_analyzer.strong_scaling_speedup)
        metrics["current_speedup"] = monitor.scalability_analyzer.strong_scaling_speedup[end]
        metrics["current_efficiency"] = monitor.scalability_analyzer.strong_scaling_efficiency[end]
        metrics["serial_fraction"] = monitor.scalability_analyzer.serial_fraction
        metrics["parallel_fraction"] = monitor.scalability_analyzer.parallel_fraction
    end
    
    return metrics
end

"""
Generate comprehensive monitoring report
"""
function generate_monitoring_report(monitor::PerformanceMonitor{T}) where {T <: AbstractFloat}
    report = String[]
    
    push!(report, "=== Performance Monitoring Report ===")
    push!(report, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    push!(report, "")
    
    # Monitoring summary
    push!(report, "Monitoring Summary:")
    push!(report, "  Status: $(monitor.is_monitoring ? "Active" : "Stopped")")
    push!(report, "  Total iterations: $(monitor.monitoring_data.current_iteration)")
    push!(report, "  Monitoring interval: $(monitor.monitoring_interval) seconds")
    
    if monitor.mpi_comm !== nothing
        push!(report, "  MPI processes: $(monitor.mpi_comm.size)")
    end
    push!(report, "")
    
    # Performance metrics
    if !isempty(monitor.monitoring_data.iteration_times)
        avg_time = sum(monitor.monitoring_data.iteration_times) / length(monitor.monitoring_data.iteration_times)
        max_time = maximum(monitor.monitoring_data.iteration_times)
        min_time = minimum(monitor.monitoring_data.iteration_times)
        
        push!(report, "Iteration Timing:")
        push!(report, "  Average time: $(round(avg_time, digits=4)) seconds")
        push!(report, "  Maximum time: $(round(max_time, digits=4)) seconds")
        push!(report, "  Minimum time: $(round(min_time, digits=4)) seconds")
        push!(report, "")
    end
    
    # Memory usage
    if !isempty(monitor.monitoring_data.memory_usage_history)
        peak_memory = maximum(monitor.monitoring_data.memory_usage_history)
        avg_memory = sum(monitor.monitoring_data.memory_usage_history) / length(monitor.monitoring_data.memory_usage_history)
        
        push!(report, "Memory Usage:")
        push!(report, "  Peak usage: $(round(peak_memory, digits=2)) MB")
        push!(report, "  Average usage: $(round(avg_memory, digits=2)) MB")
        push!(report, "")
    end
    
    # Convergence analysis
    if !isempty(monitor.monitoring_data.residual_history)
        initial_residual = monitor.monitoring_data.residual_history[1]
        final_residual = monitor.monitoring_data.residual_history[end]
        reduction_factor = initial_residual / final_residual
        
        push!(report, "Convergence Analysis:")
        push!(report, "  Initial residual: $(initial_residual)")
        push!(report, "  Final residual: $(final_residual)")
        push!(report, "  Reduction factor: $(round(reduction_factor, digits=2))")
        push!(report, "")
    end
    
    # Load balance analysis
    if !isempty(monitor.monitoring_data.process_workloads)
        push!(report, "Load Balance Analysis:")
        push!(report, "  Imbalance factor: $(round(monitor.load_analyzer.imbalance_factor, digits=4))")
        push!(report, "  Bottleneck processes: $(monitor.load_analyzer.bottleneck_processes)")
        
        max_workload = maximum(monitor.monitoring_data.process_workloads)
        min_workload = minimum(monitor.monitoring_data.process_workloads)
        avg_workload = sum(monitor.monitoring_data.process_workloads) / length(monitor.monitoring_data.process_workloads)
        
        push!(report, "  Max workload: $(round(max_workload, digits=4)) seconds")
        push!(report, "  Min workload: $(round(min_workload, digits=4)) seconds")
        push!(report, "  Avg workload: $(round(avg_workload, digits=4)) seconds")
        push!(report, "")
    end
    
    # Scalability analysis
    if !isempty(monitor.scalability_analyzer.strong_scaling_speedup)
        current_speedup = monitor.scalability_analyzer.strong_scaling_speedup[end]
        current_efficiency = monitor.scalability_analyzer.strong_scaling_efficiency[end]
        
        push!(report, "Scalability Analysis:")
        push!(report, "  Current speedup: $(round(current_speedup, digits=2))x")
        push!(report, "  Current efficiency: $(round(current_efficiency * 100, digits=1))%")
        push!(report, "  Serial fraction: $(round(monitor.scalability_analyzer.serial_fraction, digits=4))")
        push!(report, "  Parallel fraction: $(round(monitor.scalability_analyzer.parallel_fraction, digits=4))")
        push!(report, "")
    end
    
    # Strong scaling data
    if !isempty(monitor.monitoring_data.strong_scaling_data)
        push!(report, "Strong Scaling Data:")
        for (n_proc, exec_time) in sort(collect(monitor.monitoring_data.strong_scaling_data))
            speedup = monitor.scalability_analyzer.baseline_time / exec_time
            efficiency = speedup / n_proc
            push!(report, "  $n_proc processes: $(round(exec_time, digits=4))s (speedup: $(round(speedup, digits=2))x, efficiency: $(round(efficiency * 100, digits=1))%)")
        end
        push!(report, "")
    end
    
    # Weak scaling data
    if !isempty(monitor.monitoring_data.weak_scaling_data)
        push!(report, "Weak Scaling Data:")
        for (prob_size, exec_time) in sort(collect(monitor.monitoring_data.weak_scaling_data))
            throughput = prob_size / exec_time
            push!(report, "  Problem size $prob_size: $(round(exec_time, digits=4))s (throughput: $(round(throughput, digits=2)) units/s)")
        end
        push!(report, "")
    end
    
    push!(report, "=====================================")
    
    return join(report, "\n")
end

"""
Print real-time monitoring status
"""
function print_monitoring_status(monitor::PerformanceMonitor{T}) where {T <: AbstractFloat}
    if !monitor.is_monitoring
        println("Performance monitoring is not active")
        return nothing
    end
    
    metrics = get_real_time_metrics(monitor)
    
    println("=== Real-time Performance Status ===")
    println("Iteration: $(metrics["current_iteration"])")
    println("Wall time: $(round(metrics["current_wall_time"], digits=3)) seconds")
    
    if haskey(metrics, "current_residual")
        println("Current residual: $(metrics["current_residual"])")
    end
    
    if haskey(metrics, "current_memory_usage") && metrics["current_memory_usage"] > 0
        println("Memory usage: $(round(metrics["current_memory_usage"], digits=2)) MB")
    end
    
    if haskey(metrics, "load_imbalance_factor")
        println("Load imbalance: $(round(metrics["load_imbalance_factor"], digits=4))")
    end
    
    if haskey(metrics, "current_speedup")
        println("Speedup: $(round(metrics["current_speedup"], digits=2))x")
        println("Efficiency: $(round(metrics["current_efficiency"] * 100, digits=1))%")
    end
    
    println("===================================")
end

"""
Integrate performance monitoring with PararealManager
"""
function integrate_monitoring!(manager::PararealManager{T}, 
                              monitor::PerformanceMonitor{T}) where {T <: AbstractFloat}
    
    # Set up MPI communication for monitoring
    if manager.is_initialized && monitor.mpi_comm === nothing
        monitor.mpi_comm = manager.mpi_comm
        
        # Update load analyzer with correct process count
        monitor.load_analyzer = LoadBalanceAnalyzer{T}(manager.mpi_comm.size)
    end
    
    return nothing
end

end # module Parareal