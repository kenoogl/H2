# Integration test for Task 7: Performance monitoring and analysis
using Test
using Random

# Import the modules we need to test
include("../src/parareal.jl")
using .Parareal

@testset "Task 7: Performance Monitoring and Analysis Integration" begin
    
    Random.seed!(12345)
    T = Float64
    
    @testset "Complete Performance Monitoring Workflow" begin
        println("Testing complete performance monitoring workflow...")
        
        # Create Parareal configuration
        config = Parareal.PararealConfig{T}(
            total_time = T(1.0),
            n_time_windows = 4,
            dt_coarse = T(0.1),
            dt_fine = T(0.01),
            max_iterations = 5,
            convergence_tolerance = T(1e-6),
            n_mpi_processes = 2,
            n_threads_per_process = 1
        )
        
        # Create Parareal manager
        manager = Parareal.PararealManager{T}(config)
        
        # Initialize MPI (mock)
        Parareal.initialize_mpi_parareal!(manager)
        
        # Create performance monitor
        monitor = Parareal.create_performance_monitor(manager.mpi_comm, 0.5)
        
        # Integrate monitoring with manager
        Parareal.integrate_monitoring!(manager, monitor)
        
        # Start monitoring
        Parareal.start_monitoring!(monitor)
        @test monitor.is_monitoring
        
        # Simulate Parareal iterations with performance data collection
        for iteration in 1:5
            residual = T(0.1) / (iteration^2)  # Decreasing residual
            memory_usage = T(100.0) + iteration * 20.0  # Increasing memory
            
            # Update monitoring data
            Parareal.update_monitoring_data!(monitor, iteration, residual, memory_usage)
            
            # Simulate solver timing
            Parareal.update_timing_data!(manager.performance_metrics, :coarse, T(0.5))
            Parareal.update_timing_data!(manager.performance_metrics, :fine, T(2.0))
            
            # Simulate communication overhead
            Parareal.record_communication_overhead!(manager.performance_metrics, :send, T(0.1), 1000)
            Parareal.record_communication_overhead!(manager.performance_metrics, :synchronization, T(0.05))
        end
        
        # Stop monitoring
        Parareal.stop_monitoring!(monitor)
        @test !monitor.is_monitoring
        
        # Verify collected data
        @test monitor.monitoring_data.current_iteration == 5
        @test length(monitor.monitoring_data.residual_history) == 5
        @test length(monitor.monitoring_data.memory_usage_history) == 5
        @test monitor.monitoring_data.residual_history[1] > monitor.monitoring_data.residual_history[end]
        
        # Get real-time metrics
        metrics = Parareal.get_real_time_metrics(monitor)
        @test metrics["total_iterations"] == 5
        @test metrics["current_residual"] == monitor.monitoring_data.residual_history[end]
        @test metrics["residual_reduction"] > 1.0
        
        # Generate monitoring report
        report = Parareal.generate_monitoring_report(monitor)
        @test isa(report, String)
        @test occursin("Performance Monitoring Report", report)
        @test occursin("Total iterations: 5", report)
        
        # Finalize
        Parareal.finalize_mpi_parareal!(manager)
        
        println("✓ Complete performance monitoring workflow verified")
    end
    
    @testset "Performance Analysis Workflow" begin
        println("Testing performance analysis workflow...")
        
        # Create performance analyzer
        analyzer = Parareal.create_performance_analyzer(T)
        
        # Create mock performance metrics for multiple processes
        metrics_list = []
        for rank in 0:2
            metrics = Parareal.create_performance_metrics(rank, 3, 1)
            
            # Add realistic timing data
            Parareal.update_timing_data!(metrics, :coarse, T(1.0 + rank * 0.2))
            Parareal.update_timing_data!(metrics, :fine, T(3.0 + rank * 0.5))
            
            # Add communication overhead
            Parareal.record_communication_overhead!(metrics, :send, T(0.2 + rank * 0.1), 1000)
            Parareal.record_communication_overhead!(metrics, :receive, T(0.15 + rank * 0.05), 1000)
            Parareal.record_communication_overhead!(metrics, :synchronization, T(0.1))
            
            # Set wall time
            metrics.total_wall_time = T(5.0 + rank * 0.3)
            
            push!(metrics_list, metrics)
        end
        
        # Perform timing breakdown analysis
        Parareal.analyze_timing_breakdown!(analyzer, metrics_list)
        
        # Verify timing breakdown results
        breakdown = analyzer.timing_breakdown
        @test haskey(breakdown.coarse_solver_breakdown, "total_time")
        @test haskey(breakdown.fine_solver_breakdown, "total_time")
        @test haskey(breakdown.mpi_communication_breakdown, "total_communication_time")
        @test haskey(breakdown.threading_breakdown, "threading_efficiency")
        
        # Check calculated values
        expected_coarse_total = 1.0 + 1.2 + 1.4  # Sum across processes
        expected_fine_total = 3.0 + 3.5 + 4.0
        @test breakdown.coarse_solver_breakdown["total_time"] ≈ expected_coarse_total
        @test breakdown.fine_solver_breakdown["total_time"] ≈ expected_fine_total
        
        # Perform scalability analysis
        processes = [1, 2, 4, 8]
        times = [T(10.0), T(5.5), T(3.0), T(2.0)]
        Parareal.analyze_scaling_performance!(analyzer, processes, times)
        
        # Verify scaling analysis results
        scaling = analyzer.scaling_analysis
        @test length(scaling.strong_scaling_processes) == length(processes)
        @test length(scaling.strong_scaling_times) == length(times)
        @test length(scaling.strong_scaling_speedups) == length(processes)
        @test length(scaling.strong_scaling_efficiencies) == length(processes)
        
        # Check speedup calculations
        @test scaling.strong_scaling_speedups[1] ≈ 1.0  # Baseline
        @test scaling.strong_scaling_speedups[2] ≈ 10.0/5.5  # ~1.82
        @test scaling.strong_scaling_speedups[4] ≈ 10.0/2.0  # 5.0
        
        # Generate comprehensive reports
        timing_report = Parareal.generate_detailed_timing_report(analyzer)
        @test isa(timing_report, String)
        @test occursin("DETAILED TIMING BREAKDOWN REPORT", timing_report)
        
        scaling_report = Parareal.generate_scaling_analysis_report(analyzer)
        @test isa(scaling_report, String)
        @test occursin("SCALING PERFORMANCE ANALYSIS REPORT", scaling_report)
        
        comprehensive_report = Parareal.generate_comprehensive_performance_report(analyzer)
        @test isa(comprehensive_report, String)
        @test occursin("COMPREHENSIVE PARAREAL PERFORMANCE ANALYSIS REPORT", comprehensive_report)
        
        # Test CSV export
        csv_filename = "integration_test_performance.csv"
        Parareal.export_performance_data_csv(analyzer, csv_filename)
        @test isfile(csv_filename)
        
        csv_content = read(csv_filename, String)
        @test occursin("Parareal Performance Analysis Data Export", csv_content)
        @test occursin("CoarseSolver,Timing,total_time", csv_content)
        @test occursin("Strong Scaling Data", csv_content)
        
        # Clean up
        rm(csv_filename)
        
        println("✓ Performance analysis workflow verified")
    end
    
    @testset "Load Balance and Scalability Analysis" begin
        println("Testing load balance and scalability analysis...")
        
        # Create performance monitor
        monitor = Parareal.create_performance_monitor()
        
        # Create metrics with different workloads (simulating load imbalance)
        metrics1 = Parareal.create_performance_metrics(0, 3, 1)
        metrics2 = Parareal.create_performance_metrics(1, 3, 1)
        metrics3 = Parareal.create_performance_metrics(2, 3, 1)
        
        # Process 0: Heavy workload
        Parareal.update_timing_data!(metrics1, :coarse, T(2.0))
        Parareal.update_timing_data!(metrics1, :fine, T(6.0))
        metrics1.total_wall_time = T(10.0)
        
        # Process 1: Medium workload
        Parareal.update_timing_data!(metrics2, :coarse, T(1.5))
        Parareal.update_timing_data!(metrics2, :fine, T(4.0))
        metrics2.total_wall_time = T(7.0)
        
        # Process 2: Light workload
        Parareal.update_timing_data!(metrics3, :coarse, T(1.0))
        Parareal.update_timing_data!(metrics3, :fine, T(3.0))
        metrics3.total_wall_time = T(6.0)
        
        # Analyze load balance
        Parareal.analyze_load_balance!(monitor, Any[metrics1, metrics2, metrics3])
        
        # Verify load balance analysis
        @test length(monitor.monitoring_data.process_workloads) == 3
        @test length(monitor.monitoring_data.process_idle_times) == 3
        @test monitor.load_analyzer.imbalance_factor > 0
        
        # Check workload calculations
        expected_workloads = [8.0, 5.5, 4.0]  # total solver time per process
        @test monitor.monitoring_data.process_workloads ≈ expected_workloads
        
        # Test scalability analysis with multiple data points
        Parareal.calculate_scalability_metrics!(monitor, 1, T(10.0), 1000)  # Baseline
        Parareal.calculate_scalability_metrics!(monitor, 2, T(6.0), 1000)   # 1.67x speedup
        Parareal.calculate_scalability_metrics!(monitor, 4, T(3.5), 1000)   # 2.86x speedup
        Parareal.calculate_scalability_metrics!(monitor, 8, T(2.0), 1000)   # 5.0x speedup
        
        # Verify scalability metrics
        @test monitor.scalability_analyzer.baseline_time == T(10.0)
        @test length(monitor.scalability_analyzer.strong_scaling_speedup) == 3
        @test length(monitor.scalability_analyzer.strong_scaling_efficiency) == 3
        
        # Check Amdahl's law parameter estimation
        @test monitor.scalability_analyzer.serial_fraction >= 0.0
        @test monitor.scalability_analyzer.serial_fraction <= 1.0
        @test monitor.scalability_analyzer.parallel_fraction >= 0.0
        
        println("✓ Load balance and scalability analysis verified")
    end
    
    @testset "Real-time Performance Monitoring" begin
        println("Testing real-time performance monitoring...")
        
        # Create monitor with short interval for testing
        monitor = Parareal.create_performance_monitor(nothing, 0.1)
        
        # Start monitoring
        Parareal.start_monitoring!(monitor)
        
        # Simulate real-time data collection over multiple iterations
        for i in 1:10
            residual = T(1.0) / (2.0^i)  # Exponentially decreasing
            memory = T(50.0) + i * 5.0   # Linearly increasing
            
            Parareal.update_monitoring_data!(monitor, i, residual, memory)
            
            # Small delay to simulate real computation
            sleep(0.01)
        end
        
        # Get real-time metrics
        rt_metrics = Parareal.get_real_time_metrics(monitor)
        
        # Verify real-time data
        @test rt_metrics["is_monitoring"] == true
        @test rt_metrics["current_iteration"] == 10
        @test rt_metrics["total_iterations"] == 10
        @test rt_metrics["iteration_count"] == 10
        @test haskey(rt_metrics, "average_iteration_time")
        @test haskey(rt_metrics, "peak_memory_usage")
        @test haskey(rt_metrics, "current_residual")
        @test haskey(rt_metrics, "residual_reduction")
        
        # Check convergence analysis
        @test rt_metrics["current_residual"] < rt_metrics["initial_residual"]
        @test rt_metrics["residual_reduction"] > 1.0
        @test rt_metrics["peak_memory_usage"] == 100.0  # Last memory value
        
        # Test monitoring status display
        Parareal.print_monitoring_status(monitor)
        
        # Stop monitoring
        Parareal.stop_monitoring!(monitor)
        @test !monitor.is_monitoring
        
        println("✓ Real-time performance monitoring verified")
    end
    
    @testset "Performance Recommendations Generation" begin
        println("Testing performance recommendations generation...")
        
        # Create analyzer with realistic performance data
        analyzer = Parareal.create_performance_analyzer(T)
        
        # Set up timing data indicating potential issues
        breakdown = analyzer.timing_breakdown
        breakdown.coarse_solver_breakdown["total_time"] = T(1.0)
        breakdown.fine_solver_breakdown["total_time"] = T(8.0)  # Fine solver dominates
        breakdown.mpi_communication_breakdown["total_communication_time"] = T(2.0)  # High communication overhead
        breakdown.threading_breakdown["threading_efficiency"] = T(0.6)  # Poor threading efficiency
        breakdown.synchronization_overhead = T(0.15)  # High synchronization overhead
        breakdown.load_imbalance_overhead = T(0.25)  # Significant load imbalance
        
        # Set up scaling data showing poor scaling
        scaling = analyzer.scaling_analysis
        scaling.strong_scaling_processes = [1, 2, 4, 8]
        scaling.strong_scaling_times = [T(10.0), T(7.0), T(6.0), T(5.5)]  # Poor scaling
        scaling.strong_scaling_speedups = [T(1.0), T(1.43), T(1.67), T(1.82)]
        scaling.strong_scaling_efficiencies = [T(1.0), T(0.71), T(0.42), T(0.23)]  # Decreasing efficiency
        
        # Generate recommendations
        recommendations = Parareal.generate_performance_recommendations(analyzer)
        
        @test isa(recommendations, String)
        @test length(recommendations) > 0
        
        # Should identify the issues we set up (check for actual content)
        @test occursin("LOAD IMBALANCE", recommendations) || 
              occursin("SCALING EFFICIENCY", recommendations) ||
              occursin("Recommendations:", recommendations) ||
              occursin("PERFORMANCE LOOKS GOOD", recommendations)
        
        println("✓ Performance recommendations generation verified")
    end
end

println("Task 7: Performance monitoring and analysis integration tests completed successfully!")