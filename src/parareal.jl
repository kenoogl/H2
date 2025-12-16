# Parareal time parallelization module for Heat3ds
module Parareal

using MPI
using Printf
using LinearAlgebra
using FLoops
using ThreadsX

# Import Heat3ds modules
include("common.jl")
using .Common
using .Common: WorkBuffers, get_backend

export PararealManager, PararealConfig, TimeWindow, MPICommunicator
export initialize_mpi_parareal!, finalize_mpi_parareal!, run_parareal!

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
    is_initialized::Bool
    
    function PararealManager{T}(config::PararealConfig{T}) where {T <: AbstractFloat}
        # MPI communicator will be initialized later
        mpi_comm = MPICommunicator{T}(MPI.COMM_NULL)
        time_windows = Vector{TimeWindow{T}}()
        
        return new{T}(config, mpi_comm, time_windows, false)
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
    size = manager.mpi_comm.size
    
    # Validate MPI process count
    if size != manager.config.n_mpi_processes
        if rank == 0
            @warn "MPI process count mismatch: expected $(manager.config.n_mpi_processes), got $size"
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
            size,  # Use actual MPI size
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
        println("MPI Processes: $size")
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
                                    target_rank::Int) where {T <: AbstractFloat}
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
        initialize_communication_buffers!(comm, size(temperature_data))
    end
    
    # Copy data to send buffer
    comm.send_buffers[target_rank + 1] .= temperature_data
    
    # Use non-blocking communication for better performance
    tag = 100 + comm.rank  # Unique tag for each sender
    
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
function synchronize_processes!(comm::MPICommunicator{T}) where {T <: AbstractFloat}
    if comm.comm == MPI.COMM_NULL
        # For testing without MPI
        return nothing
    end
    
    MPI.Barrier(comm.comm)
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
Main Parareal algorithm (placeholder for now)
"""
function run_parareal!(manager::PararealManager{T}, initial_condition) where {T <: AbstractFloat}
    if !manager.is_initialized
        error("PararealManager not initialized. Call initialize_mpi_parareal! first.")
    end
    
    rank = manager.mpi_comm.rank
    
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
    
    # TODO: Implement actual Parareal algorithm
    # This will be implemented in subsequent tasks
    
    return nothing
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
    fine_size = size(fine_data)
    coarse_size = size(coarse_data)
    
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
    fine_size = size(fine_data)
    coarse_size = size(coarse_data)
    
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
                      problem_data::Any) where {T <: AbstractFloat}
    
    # Create coarse grid
    fine_size = size(initial_condition)
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
                    problem_data::Heat3dsProblemData{T}) where {T <: AbstractFloat}
    
    # Create working buffers for the fine solver
    grid_size = size(initial_condition)
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
    
    return copy(wk.θ)
end

"""
Apply simple diffusion step (placeholder for full Heat3ds integration)
"""
function apply_simple_diffusion_step!(temperature::Array{T,3}, dt::T, diffusivity::T) where {T <: AbstractFloat}
    grid_size = size(temperature)
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
                      problem_data::Heat3dsProblemData{T}) where {T <: AbstractFloat}
    
    # Create coarse grid
    fine_size = size(initial_condition)
    coarse_size = create_coarse_grid(fine_size, solver.spatial_resolution_factor)
    
    # Initialize coarse grid data
    coarse_solution = zeros(T, coarse_size...)
    
    # Restrict initial condition to coarse grid
    restrict_fine_to_coarse!(coarse_solution, initial_condition)
    
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
    
    # Interpolate back to fine grid
    fine_solution = zeros(T, fine_size...)
    interpolate_coarse_to_fine!(fine_solution, coarse_solution)
    
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

end # module Parareal