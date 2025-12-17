# Test performance monitoring system functionality
using Test

# Add src to path for testing
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Include the Parareal module
include("../src/parareal.jl")
using .Parareal

@testset "Performance Monitoring System Tests" begin
    
    @testset "PerformanceMonitor Creation" begin
        # Test creating performance monitor without MPI
        monitor = Parareal.create_performance_monitor()
        
        @test !monitor.is_monitoring
        @test monitor.monitoring_interval == 1.0
        @test monitor.mpi_comm === nothing
        @test monitor.monitoring_data.current_iteration == 0
        @test monitor.load_analyzer.imbalance_factor == 0.0
        @test monitor.scalability_analyzer.baseline_time == 0.0
    end
    
    @testset "Monitoring Control" begin
        monitor = Parareal.create_performance_monitor()
        
        # Test starting monitoring
        Parareal.start_monitoring!(monitor)
        @test monitor.is_monitoring
        @test monitor.last_update_time > 0
        
        # Test stopping monitoring
        Parareal.stop_monitoring!(monitor)
        @test !monitor.is_monitoring
        
        # Test double start/stop
        Parareal.start_monitoring!(monitor)
        Parareal.start_monitoring!(monitor)  # Should warn but not error
        @test monitor.is_monitoring
        
        Parareal.stop_monitoring!(monitor)
        Parareal.stop_monitoring!(monitor)  # Should warn but not error
        @test !monitor.is_monitoring
    end
    
    @testset "Real-time Data Collection" begin
        monitor = Parareal.create_performance_monitor()
        Parareal.start_monitoring!(monitor)
        
        # Test updating monitoring data
        Parareal.update_monitoring_data!(monitor, 1, 0.1, 100.0)
        @test monitor.monitoring_data.current_iteration == 1
        @test monitor.monitoring_data.current_memory_usage == 100.0
        @test length(monitor.monitoring_data.residual_history) == 1
        @test monitor.monitoring_data.residual_history[1] == 0.1
        
        # Test multiple updates
        Parareal.update_monitoring_data!(monitor, 2, 0.05, 120.0)
        Parareal.update_monitoring_data!(monitor, 3, 0.01, 110.0)
        
        @test monitor.monitoring_data.current_iteration == 3
        @test length(monitor.monitoring_data.residual_history) == 3
        @test length(monitor.monitoring_data.memory_usage_history) == 3
        @test length(monitor.monitoring_data.iteration_times) == 3
        
        Parareal.stop_monitoring!(monitor)
    end
    
    @testset "Load Balance Analysis" begin
        monitor = Parareal.create_performance_monitor()
        
        # Create mock performance metrics for multiple processes
        metrics1 = Parareal.create_performance_metrics(0, 3, 1)
        metrics2 = Parareal.create_performance_metrics(1, 3, 1)
        metrics3 = Parareal.create_performance_metrics(2, 3, 1)
        
        # Set different workloads
        Parareal.update_timing_data!(metrics1, :coarse, 2.0)
        Parareal.update_timing_data!(metrics1, :fine, 5.0)
        metrics1.total_wall_time = 8.0
        
        Parareal.update_timing_data!(metrics2, :coarse, 1.5)
        Parareal.update_timing_data!(metrics2, :fine, 4.0)
        metrics2.total_wall_time = 6.0
        
        Parareal.update_timing_data!(metrics3, :coarse, 3.0)
        Parareal.update_timing_data!(metrics3, :fine, 6.0)
        metrics3.total_wall_time = 10.0
        
        # Perform load balance analysis
        process_metrics = Any[metrics1, metrics2, metrics3]
        Parareal.analyze_load_balance!(monitor, process_metrics)
        
        # Check results
        @test length(monitor.monitoring_data.process_workloads) == 3
        @test length(monitor.monitoring_data.process_idle_times) == 3
        @test monitor.load_analyzer.imbalance_factor > 0
        
        # Check workload calculations
        expected_workloads = [7.0, 5.5, 9.0]  # total_solver_time for each process
        @test monitor.monitoring_data.process_workloads ≈ expected_workloads
        
        # Check imbalance factor calculation
        max_workload = maximum(expected_workloads)
        min_workload = minimum(expected_workloads)
        avg_workload = sum(expected_workloads) / length(expected_workloads)
        expected_imbalance = (max_workload - min_workload) / avg_workload
        @test monitor.load_analyzer.imbalance_factor ≈ expected_imbalance
    end
    
    @testset "Scalability Analysis" begin
        monitor = Parareal.create_performance_monitor()
        
        # Test strong scaling analysis
        # Baseline: 1 process, 10 seconds
        Parareal.calculate_scalability_metrics!(monitor, 1, 10.0, 1000)
        @test monitor.scalability_analyzer.baseline_time == 10.0
        @test monitor.scalability_analyzer.baseline_processes == 1
        @test monitor.scalability_analyzer.baseline_problem_size == 1000
        
        # 2 processes, 6 seconds (1.67x speedup)
        Parareal.calculate_scalability_metrics!(monitor, 2, 6.0, 1000)
        @test length(monitor.scalability_analyzer.strong_scaling_speedup) == 1
        @test monitor.scalability_analyzer.strong_scaling_speedup[1] ≈ 10.0/6.0
        @test monitor.scalability_analyzer.strong_scaling_efficiency[1] ≈ (10.0/6.0)/2
        
        # 4 processes, 3 seconds (3.33x speedup)
        Parareal.calculate_scalability_metrics!(monitor, 4, 3.0, 1000)
        @test length(monitor.scalability_analyzer.strong_scaling_speedup) == 2
        @test monitor.scalability_analyzer.strong_scaling_speedup[2] ≈ 10.0/3.0
        @test monitor.scalability_analyzer.strong_scaling_efficiency[2] ≈ (10.0/3.0)/4
        
        # Check Amdahl's law parameter estimation
        @test monitor.scalability_analyzer.serial_fraction >= 0.0
        @test monitor.scalability_analyzer.serial_fraction <= 1.0
        @test monitor.scalability_analyzer.parallel_fraction >= 0.0
        @test monitor.scalability_analyzer.parallel_fraction <= 1.0
        @test monitor.scalability_analyzer.serial_fraction + monitor.scalability_analyzer.parallel_fraction ≈ 1.0
    end
    
    @testset "Real-time Metrics Retrieval" begin
        monitor = Parareal.create_performance_monitor()
        Parareal.start_monitoring!(monitor)
        
        # Add some data
        Parareal.update_monitoring_data!(monitor, 1, 0.1, 100.0)
        Parareal.update_monitoring_data!(monitor, 2, 0.05, 120.0)
        Parareal.update_monitoring_data!(monitor, 3, 0.01, 110.0)
        
        # Get real-time metrics
        metrics = Parareal.get_real_time_metrics(monitor)
        
        @test metrics["is_monitoring"] == true
        @test metrics["current_iteration"] == 3
        @test metrics["total_iterations"] == 3
        @test metrics["iteration_count"] == 3
        @test haskey(metrics, "average_iteration_time")
        @test haskey(metrics, "max_iteration_time")
        @test haskey(metrics, "min_iteration_time")
        @test haskey(metrics, "peak_memory_usage")
        @test haskey(metrics, "average_memory_usage")
        @test haskey(metrics, "current_residual")
        @test haskey(metrics, "initial_residual")
        @test haskey(metrics, "residual_reduction")
        
        @test metrics["current_residual"] == 0.01
        @test metrics["initial_residual"] == 0.1
        @test metrics["residual_reduction"] ≈ 0.1 / 0.01
        @test metrics["peak_memory_usage"] == 120.0
        
        Parareal.stop_monitoring!(monitor)
    end
    
    @testset "Monitoring Report Generation" begin
        monitor = Parareal.create_performance_monitor()
        Parareal.start_monitoring!(monitor)
        
        # Add comprehensive test data
        for i in 1:5
            residual = 0.1 / (i^2)  # Decreasing residual
            memory = 100.0 + i * 10.0  # Increasing memory
            Parareal.update_monitoring_data!(monitor, i, residual, memory)
        end
        
        # Add load balance data
        metrics1 = Parareal.create_performance_metrics(0, 2, 1)
        metrics2 = Parareal.create_performance_metrics(1, 2, 1)
        Parareal.update_timing_data!(metrics1, :coarse, 2.0)
        Parareal.update_timing_data!(metrics2, :coarse, 3.0)
        metrics1.total_wall_time = 5.0
        metrics2.total_wall_time = 6.0
        
        Parareal.analyze_load_balance!(monitor, Any[metrics1, metrics2])
        
        # Add scalability data
        Parareal.calculate_scalability_metrics!(monitor, 1, 10.0, 1000)
        Parareal.calculate_scalability_metrics!(monitor, 2, 6.0, 1000)
        
        # Generate report
        report = Parareal.generate_monitoring_report(monitor)
        
        @test isa(report, String)
        @test occursin("Performance Monitoring Report", report)
        @test occursin("Monitoring Summary", report)
        @test occursin("Iteration Timing", report)
        @test occursin("Memory Usage", report)
        @test occursin("Convergence Analysis", report)
        @test occursin("Load Balance Analysis", report)
        @test occursin("Scalability Analysis", report)
        @test occursin("Strong Scaling Data", report)
        
        # Check specific values in report
        @test occursin("Total iterations: 5", report)
        @test occursin("Peak usage: 150.0 MB", report)
        @test occursin("Imbalance factor:", report)
        @test occursin("Current speedup:", report)
        
        Parareal.stop_monitoring!(monitor)
    end
    
    @testset "Integration with PararealManager" begin
        # Create a simple PararealManager for testing
        config = Parareal.PararealConfig(
            total_time = 1.0,
            n_time_windows = 4,
            dt_coarse = 0.1,
            dt_fine = 0.01,
            n_mpi_processes = 2,
            n_threads_per_process = 1
        )
        
        manager = Parareal.PararealManager{Float64}(config)
        monitor = Parareal.create_performance_monitor()
        
        # Test integration
        Parareal.integrate_monitoring!(manager, monitor)
        
        # Since manager is not initialized, mpi_comm should still be nothing
        @test monitor.mpi_comm === nothing
        
        # Test with initialized manager (mock)
        manager.is_initialized = true
        manager.mpi_comm = Parareal.MPICommunicator{Float64}(Parareal.MPI.COMM_NULL)
        
        Parareal.integrate_monitoring!(manager, monitor)
        @test monitor.mpi_comm === manager.mpi_comm
        @test monitor.load_analyzer.process_metrics isa Vector
    end
    
    @testset "Monitoring Status Display" begin
        monitor = Parareal.create_performance_monitor()
        
        # Test with inactive monitoring
        # This should print a message but not error
        Parareal.print_monitoring_status(monitor)
        
        # Test with active monitoring
        Parareal.start_monitoring!(monitor)
        Parareal.update_monitoring_data!(monitor, 5, 0.001, 200.0)
        
        # Add some scalability data
        Parareal.calculate_scalability_metrics!(monitor, 1, 10.0)
        Parareal.calculate_scalability_metrics!(monitor, 2, 6.0)
        
        # This should print detailed status
        Parareal.print_monitoring_status(monitor)
        
        Parareal.stop_monitoring!(monitor)
    end
end

println("Performance monitoring system tests completed successfully!")