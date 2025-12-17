# Property-based test for performance monitoring accuracy
# **Feature: parareal-time-parallelization, Property 10: Performance Monitoring Accuracy**
# **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 10.1, 10.2, 10.3, 10.4, 10.5**

using Test
using Random
using Statistics
using LinearAlgebra

# Add src to path for testing
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Include the Parareal module
include("../src/parareal.jl")
using .Parareal

"""
Property-based test for performance monitoring accuracy

Property 10: Performance Monitoring Accuracy
*For any* parareal computation, the performance metrics (timing, efficiency, speedup) 
should accurately reflect the actual resource usage and computational behavior

This property validates:
- Requirements 4.1: Record coarse solver computation time for each iteration
- Requirements 4.2: Record fine solver computation time for each iteration  
- Requirements 4.3: Record communication overhead between time windows
- Requirements 4.4: Report total speedup compared to sequential execution
- Requirements 4.5: Output residual norms for each parareal iteration
- Requirements 10.1: Measure MPI communication time separately from computation time
- Requirements 10.2: Measure threading efficiency within each MPI process
- Requirements 10.3: Track load balancing across MPI processes
- Requirements 10.4: Output detailed timing breakdown for MPI and threading components
- Requirements 10.5: Report strong and weak scaling metrics for the hybrid parallelization
"""

# Test configuration
const NUM_PROPERTY_TESTS = 50
const MIN_GRID_SIZE = 8
const MAX_GRID_SIZE = 32
const MIN_TIME_STEPS = 5
const MAX_TIME_STEPS = 20
const TIMING_TOLERANCE = 0.1  # 10% tolerance for timing measurements

"""
Generate random performance monitoring test configuration
"""
function generate_random_monitoring_config(::Type{T} = Float64) where {T <: AbstractFloat}
    # Grid dimensions
    nx = rand(MIN_GRID_SIZE:MAX_GRID_SIZE)
    ny = rand(MIN_GRID_SIZE:MAX_GRID_SIZE)
    nz = rand(MIN_GRID_SIZE:MAX_GRID_SIZE)
    
    # Time parameters
    total_time = T(rand(0.5:0.1:2.0))
    n_windows = rand(2:6)
    n_iterations = rand(MIN_TIME_STEPS:MAX_TIME_STEPS)
    
    # MPI and threading parameters
    n_processes = rand(1:4)
    n_threads = rand(1:2)
    
    # Solver parameters
    dt_coarse = T(rand(0.05:0.01:0.2))
    dt_fine = T(dt_coarse / rand(5:20))
    
    return (
        grid_size = (nx, ny, nz),
        total_time = total_time,
        n_windows = n_windows,
        n_iterations = n_iterations,
        n_processes = n_processes,
        n_threads = n_threads,
        dt_coarse = dt_coarse,
        dt_fine = dt_fine
    )
end

"""
Create mock parareal computation for testing performance monitoring
"""
function create_mock_parareal_computation(config)
    T = Float64
    
    # Create PararealConfig
    parareal_config = Parareal.PararealConfig{T}(
        total_time = config.total_time,
        n_time_windows = config.n_windows,
        dt_coarse = config.dt_coarse,
        dt_fine = config.dt_fine,
        max_iterations = config.n_iterations,
        convergence_tolerance = T(1e-6),
        n_mpi_processes = config.n_processes,
        n_threads_per_process = config.n_threads
    )
    
    # Create manager
    manager = Parareal.PararealManager{T}(parareal_config)
    
    # Create performance metrics
    metrics = Parareal.create_performance_metrics(0, config.n_processes, config.n_threads)
    manager.performance_metrics = metrics
    
    return manager, metrics
end

"""
Simulate solver execution with known timing
"""
function simulate_solver_execution!(metrics, solver_type::Symbol, expected_time::T) where {T <: AbstractFloat}
    start_time = time_ns()
    
    # Simulate work (sleep for expected time)
    sleep(expected_time)
    
    actual_time = T((time_ns() - start_time) / 1e9)
    
    # Record timing
    Parareal.update_timing_data!(metrics, solver_type, actual_time)
    
    return actual_time
end

"""
Simulate MPI communication with known overhead
"""
function simulate_mpi_communication!(metrics, comm_type::Symbol, expected_time::T, data_size::Int = 1000) where {T <: AbstractFloat}
    start_time = time_ns()
    
    # Simulate communication overhead
    sleep(expected_time)
    
    actual_time = T((time_ns() - start_time) / 1e9)
    
    # Record communication overhead
    Parareal.record_communication_overhead!(metrics, comm_type, actual_time, data_size)
    
    return actual_time
end

"""
Test timing accuracy property
Requirements 4.1, 4.2: Record coarse and fine solver computation times accurately
"""
function test_timing_accuracy_property(config)
    manager, metrics = create_mock_parareal_computation(config)
    
    # Expected timing values
    expected_coarse_time = 0.1
    expected_fine_time = 0.2
    
    # Simulate solver executions
    actual_coarse_time = simulate_solver_execution!(metrics, :coarse, expected_coarse_time)
    actual_fine_time = simulate_solver_execution!(metrics, :fine, expected_fine_time)
    
    # Property: Recorded times should match actual execution times within tolerance
    recorded_coarse_time = metrics.timing_data.coarse_solver_time
    recorded_fine_time = metrics.timing_data.fine_solver_time
    
    coarse_error = abs(recorded_coarse_time - actual_coarse_time) / actual_coarse_time
    fine_error = abs(recorded_fine_time - actual_fine_time) / actual_fine_time
    
    return (
        coarse_accurate = coarse_error < TIMING_TOLERANCE,
        fine_accurate = fine_error < TIMING_TOLERANCE,
        coarse_error = coarse_error,
        fine_error = fine_error,
        recorded_coarse = recorded_coarse_time,
        recorded_fine = recorded_fine_time,
        actual_coarse = actual_coarse_time,
        actual_fine = actual_fine_time
    )
end

"""
Test communication overhead tracking property
Requirements 4.3, 10.1: Record communication overhead accurately
"""
function test_communication_tracking_property(config)
    manager, metrics = create_mock_parareal_computation(config)
    
    # Expected communication times
    expected_send_time = 0.05
    expected_recv_time = 0.03
    expected_sync_time = 0.02
    
    # Simulate communications
    actual_send_time = simulate_mpi_communication!(metrics, :send, expected_send_time, 5000)
    actual_recv_time = simulate_mpi_communication!(metrics, :receive, expected_recv_time, 5000)
    actual_sync_time = simulate_mpi_communication!(metrics, :synchronization, expected_sync_time)
    
    # Property: Communication times should be recorded accurately
    recorded_send_time = metrics.communication_metrics.send_time
    recorded_recv_time = metrics.communication_metrics.receive_time
    recorded_sync_time = metrics.communication_metrics.synchronization_time
    
    send_error = abs(recorded_send_time - actual_send_time) / actual_send_time
    recv_error = abs(recorded_recv_time - actual_recv_time) / actual_recv_time
    sync_error = abs(recorded_sync_time - actual_sync_time) / actual_sync_time
    
    return (
        send_accurate = send_error < TIMING_TOLERANCE,
        recv_accurate = recv_error < TIMING_TOLERANCE,
        sync_accurate = sync_error < TIMING_TOLERANCE,
        send_error = send_error,
        recv_error = recv_error,
        sync_error = sync_error,
        total_comm_time = recorded_send_time + recorded_recv_time + recorded_sync_time
    )
end

"""
Test efficiency metrics calculation property
Requirements 4.4, 10.2: Calculate efficiency metrics accurately
"""
function test_efficiency_calculation_property(config)
    manager, metrics = create_mock_parareal_computation(config)
    
    # Set up known timing data
    coarse_time = 0.5
    fine_time = 1.0
    comm_time = 0.2
    total_time = coarse_time + fine_time + comm_time
    
    # Update metrics with known values
    Parareal.update_timing_data!(metrics, :coarse, coarse_time)
    Parareal.update_timing_data!(metrics, :fine, fine_time)
    Parareal.record_communication_overhead!(metrics, :send, comm_time / 2)
    Parareal.record_communication_overhead!(metrics, :receive, comm_time / 2)
    
    metrics.total_wall_time = total_time
    
    # Calculate efficiency metrics
    Parareal.calculate_efficiency_metrics!(metrics)
    
    # Property: Efficiency calculations should be mathematically correct
    expected_parallel_efficiency = (coarse_time + fine_time) / total_time
    expected_comm_overhead_ratio = comm_time / total_time
    
    actual_parallel_efficiency = metrics.efficiency_metrics.parallel_efficiency
    actual_comm_overhead_ratio = metrics.efficiency_metrics.communication_overhead_ratio
    
    efficiency_error = abs(actual_parallel_efficiency - expected_parallel_efficiency)
    overhead_error = abs(actual_comm_overhead_ratio - expected_comm_overhead_ratio)
    
    return (
        efficiency_accurate = efficiency_error < 1e-10,
        overhead_accurate = overhead_error < 1e-10,
        efficiency_error = efficiency_error,
        overhead_error = overhead_error,
        expected_efficiency = expected_parallel_efficiency,
        actual_efficiency = actual_parallel_efficiency,
        expected_overhead = expected_comm_overhead_ratio,
        actual_overhead = actual_comm_overhead_ratio
    )
end

"""
Test performance summary generation property
Requirements 4.5, 10.4: Generate accurate performance summaries
"""
function test_performance_summary_property(config)
    manager, metrics = create_mock_parareal_computation(config)
    
    # Set up comprehensive performance data
    Parareal.update_timing_data!(metrics, :coarse, 0.3)
    Parareal.update_timing_data!(metrics, :fine, 0.8)
    Parareal.record_communication_overhead!(metrics, :send, 0.1)
    Parareal.record_communication_overhead!(metrics, :receive, 0.05)
    
    metrics.total_wall_time = 1.25
    metrics.parareal_iterations = 5
    metrics.peak_memory_usage_mb = 150.0
    
    # Calculate efficiency metrics
    Parareal.calculate_efficiency_metrics!(metrics)
    
    # Generate performance summary
    summary = Parareal.get_performance_summary(metrics)
    
    # Property: Summary should contain all required metrics
    required_keys = [
        "total_wall_time", "coarse_solver_time", "fine_solver_time",
        "total_communication_time", "parallel_efficiency", "speedup_factor",
        "parareal_iterations", "peak_memory_usage_mb"
    ]
    
    all_keys_present = all(haskey(summary, key) for key in required_keys)
    
    # Property: Summary values should match metrics values
    values_match = (
        summary["total_wall_time"] == metrics.total_wall_time &&
        summary["coarse_solver_time"] == metrics.timing_data.coarse_solver_time &&
        summary["fine_solver_time"] == metrics.timing_data.fine_solver_time &&
        summary["parareal_iterations"] == metrics.parareal_iterations &&
        summary["peak_memory_usage_mb"] == metrics.peak_memory_usage_mb
    )
    
    return (
        all_keys_present = all_keys_present,
        values_match = values_match,
        summary_keys = collect(keys(summary)),
        required_keys = required_keys,
        missing_keys = setdiff(required_keys, collect(keys(summary)))
    )
end

"""
Test load balance analysis property
Requirements 10.3: Track load balancing accurately
"""
function test_load_balance_analysis_property(config)
    # Create performance monitor
    monitor = Parareal.create_performance_monitor()
    
    # Create mock metrics for multiple processes with different workloads
    process_metrics = []
    expected_workloads = Float64[]
    
    for i in 1:config.n_processes
        metrics = Parareal.create_performance_metrics(i-1, config.n_processes, config.n_threads)
        
        # Assign different workloads to each process
        coarse_time = 0.2 + (i-1) * 0.1  # Increasing workload
        fine_time = 0.5 + (i-1) * 0.2
        total_workload = coarse_time + fine_time
        
        Parareal.update_timing_data!(metrics, :coarse, coarse_time)
        Parareal.update_timing_data!(metrics, :fine, fine_time)
        metrics.total_wall_time = total_workload + 0.1  # Add some overhead
        
        push!(process_metrics, metrics)
        push!(expected_workloads, total_workload)
    end
    
    # Perform load balance analysis
    Parareal.analyze_load_balance!(monitor, process_metrics)
    
    # Property: Load balance analysis should correctly identify workload distribution
    recorded_workloads = monitor.monitoring_data.process_workloads
    imbalance_factor = monitor.load_analyzer.imbalance_factor
    
    # Calculate expected imbalance factor
    max_workload = maximum(expected_workloads)
    min_workload = minimum(expected_workloads)
    avg_workload = sum(expected_workloads) / length(expected_workloads)
    expected_imbalance = (max_workload - min_workload) / avg_workload
    
    workloads_match = isapprox(recorded_workloads, expected_workloads, rtol=1e-10)
    imbalance_accurate = abs(imbalance_factor - expected_imbalance) < 1e-10
    
    return (
        workloads_match = workloads_match,
        imbalance_accurate = imbalance_accurate,
        expected_workloads = expected_workloads,
        recorded_workloads = recorded_workloads,
        expected_imbalance = expected_imbalance,
        recorded_imbalance = imbalance_factor
    )
end

"""
Test scalability metrics property
Requirements 10.5: Report strong and weak scaling metrics accurately
"""
function test_scalability_metrics_property(config)
    monitor = Parareal.create_performance_monitor()
    
    # Test strong scaling (fixed problem size, varying processes)
    baseline_time = 10.0
    problem_size = 1000
    
    # Set baseline
    Parareal.calculate_scalability_metrics!(monitor, 1, baseline_time, problem_size)
    
    # Add scaling data points
    scaling_data = [
        (2, 6.0),   # 1.67x speedup
        (4, 3.5),   # 2.86x speedup
        (8, 2.0)    # 5.0x speedup
    ]
    
    expected_speedups = Float64[]
    expected_efficiencies = Float64[]
    
    for (n_proc, exec_time) in scaling_data
        Parareal.calculate_scalability_metrics!(monitor, n_proc, exec_time, problem_size)
        
        expected_speedup = baseline_time / exec_time
        expected_efficiency = expected_speedup / n_proc
        
        push!(expected_speedups, expected_speedup)
        push!(expected_efficiencies, expected_efficiency)
    end
    
    # Property: Scalability metrics should be calculated correctly
    recorded_speedups = monitor.scalability_analyzer.strong_scaling_speedup
    recorded_efficiencies = monitor.scalability_analyzer.strong_scaling_efficiency
    
    speedups_match = isapprox(recorded_speedups, expected_speedups, rtol=1e-10)
    efficiencies_match = isapprox(recorded_efficiencies, expected_efficiencies, rtol=1e-10)
    
    # Property: Amdahl's law parameters should be reasonable
    serial_fraction = monitor.scalability_analyzer.serial_fraction
    parallel_fraction = monitor.scalability_analyzer.parallel_fraction
    
    fractions_valid = (
        serial_fraction >= 0.0 && serial_fraction <= 1.0 &&
        parallel_fraction >= 0.0 && parallel_fraction <= 1.0 &&
        abs(serial_fraction + parallel_fraction - 1.0) < 1e-10
    )
    
    return (
        speedups_match = speedups_match,
        efficiencies_match = efficiencies_match,
        fractions_valid = fractions_valid,
        expected_speedups = expected_speedups,
        recorded_speedups = recorded_speedups,
        expected_efficiencies = expected_efficiencies,
        recorded_efficiencies = recorded_efficiencies,
        serial_fraction = serial_fraction,
        parallel_fraction = parallel_fraction
    )
end

# Main property-based test suite
@testset "Property 10: Performance Monitoring Accuracy" begin
    
    @testset "Timing Accuracy Property (Requirements 4.1, 4.2)" begin
        println("Testing timing accuracy property...")
        
        for i in 1:NUM_PROPERTY_TESTS
            config = generate_random_monitoring_config(Float64)
            result = test_timing_accuracy_property(config)
            
            @test result.coarse_accurate
            @test result.fine_accurate
            
            if !result.coarse_accurate
                println("  Coarse timing error: $(result.coarse_error) (tolerance: $TIMING_TOLERANCE)")
                println("  Expected: $(result.actual_coarse), Recorded: $(result.recorded_coarse)")
            end
            
            if !result.fine_accurate
                println("  Fine timing error: $(result.fine_error) (tolerance: $TIMING_TOLERANCE)")
                println("  Expected: $(result.actual_fine), Recorded: $(result.recorded_fine)")
            end
        end
        
        println("✓ Timing accuracy property passed for $NUM_PROPERTY_TESTS test cases")
    end
    
    @testset "Communication Tracking Property (Requirements 4.3, 10.1)" begin
        println("Testing communication tracking property...")
        
        for i in 1:NUM_PROPERTY_TESTS
            config = generate_random_monitoring_config(Float64)
            result = test_communication_tracking_property(config)
            
            @test result.send_accurate
            @test result.recv_accurate
            @test result.sync_accurate
            @test result.total_comm_time > 0.0
            
            if !result.send_accurate
                println("  Send timing error: $(result.send_error) (tolerance: $TIMING_TOLERANCE)")
            end
            if !result.recv_accurate
                println("  Receive timing error: $(result.recv_error) (tolerance: $TIMING_TOLERANCE)")
            end
            if !result.sync_accurate
                println("  Sync timing error: $(result.sync_error) (tolerance: $TIMING_TOLERANCE)")
            end
        end
        
        println("✓ Communication tracking property passed for $NUM_PROPERTY_TESTS test cases")
    end
    
    @testset "Efficiency Calculation Property (Requirements 4.4, 10.2)" begin
        println("Testing efficiency calculation property...")
        
        for i in 1:NUM_PROPERTY_TESTS
            config = generate_random_monitoring_config(Float64)
            result = test_efficiency_calculation_property(config)
            
            @test result.efficiency_accurate
            @test result.overhead_accurate
            @test 0.0 <= result.actual_efficiency <= 1.0
            @test 0.0 <= result.actual_overhead <= 1.0
            
            if !result.efficiency_accurate
                println("  Efficiency error: $(result.efficiency_error)")
                println("  Expected: $(result.expected_efficiency), Actual: $(result.actual_efficiency)")
            end
            
            if !result.overhead_accurate
                println("  Overhead error: $(result.overhead_error)")
                println("  Expected: $(result.expected_overhead), Actual: $(result.actual_overhead)")
            end
        end
        
        println("✓ Efficiency calculation property passed for $NUM_PROPERTY_TESTS test cases")
    end
    
    @testset "Performance Summary Property (Requirements 4.5, 10.4)" begin
        println("Testing performance summary property...")
        
        for i in 1:NUM_PROPERTY_TESTS
            config = generate_random_monitoring_config(Float64)
            result = test_performance_summary_property(config)
            
            @test result.all_keys_present
            @test result.values_match
            
            if !result.all_keys_present
                println("  Missing keys: $(result.missing_keys)")
            end
            
            if !result.values_match
                println("  Summary values do not match metrics values")
            end
        end
        
        println("✓ Performance summary property passed for $NUM_PROPERTY_TESTS test cases")
    end
    
    @testset "Load Balance Analysis Property (Requirements 10.3)" begin
        println("Testing load balance analysis property...")
        
        for i in 1:min(NUM_PROPERTY_TESTS, 20)  # Reduce for performance
            config = generate_random_monitoring_config(Float64)
            result = test_load_balance_analysis_property(config)
            
            @test result.workloads_match
            @test result.imbalance_accurate
            @test result.recorded_imbalance >= 0.0
            
            if !result.workloads_match
                println("  Workloads mismatch:")
                println("  Expected: $(result.expected_workloads)")
                println("  Recorded: $(result.recorded_workloads)")
            end
            
            if !result.imbalance_accurate
                println("  Imbalance factor error:")
                println("  Expected: $(result.expected_imbalance), Recorded: $(result.recorded_imbalance)")
            end
        end
        
        println("✓ Load balance analysis property passed")
    end
    
    @testset "Scalability Metrics Property (Requirements 10.5)" begin
        println("Testing scalability metrics property...")
        
        for i in 1:min(NUM_PROPERTY_TESTS, 10)  # Reduce for performance
            config = generate_random_monitoring_config(Float64)
            result = test_scalability_metrics_property(config)
            
            @test result.speedups_match
            @test result.efficiencies_match
            @test result.fractions_valid
            
            if !result.speedups_match
                println("  Speedups mismatch:")
                println("  Expected: $(result.expected_speedups)")
                println("  Recorded: $(result.recorded_speedups)")
            end
            
            if !result.efficiencies_match
                println("  Efficiencies mismatch:")
                println("  Expected: $(result.expected_efficiencies)")
                println("  Recorded: $(result.recorded_efficiencies)")
            end
            
            if !result.fractions_valid
                println("  Invalid Amdahl's law fractions:")
                println("  Serial: $(result.serial_fraction), Parallel: $(result.parallel_fraction)")
            end
        end
        
        println("✓ Scalability metrics property passed")
    end
end

println("All performance monitoring accuracy property tests completed successfully!")