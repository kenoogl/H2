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

end # module Parareal