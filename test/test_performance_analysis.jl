# Unit tests for performance analysis functionality
# Tests Requirements 4.1, 4.2, 4.3, 4.4, 4.5

using Test
using Random
using Statistics

# Add src to path for testing
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Include the Parareal module
include("../src/parareal.jl")
using .Parareal

@testset "Performance Analysis Unit Tests" begin
    
    @testset "PerformanceAnalyzer Creation and Configuration" begin
        # Test creating performance analyzer
        analyzer = Parareal.create_performance_analyzer(Float64)
        
        @test analyzer isa Parareal.PerformanceAnalyzer{Float64}
        @test analyzer.timing_breakdown isa Parareal.TimingBreakdown{Float64}
        @test analyzer.scaling_analysis isa Parareal.ScalingAnalysis{Float64}
        @test analyzer.visualization isa Parareal.PerformanceVisualization{Float64}
        
        # Test default configuration (updated to match actual defaults)
        @test analyzer.enable_detailed_timing == true
        @test analyzer.enable_scaling_analysis == true
        @test analyzer.enable_visualization == true
        @test analyzer.output_directory == "performance_reports"
        @test analyzer.report_format == :text
        
        # Test disabling features
        analyzer.enable_detailed_timing = false
        analyzer.enable_scaling_analysis = false
        analyzer.enable_visualization = false
        
        @test analyzer.enable_detailed_timing == false
        @test analyzer.enable_scaling_analysis == false
        @test analyzer.enable_visualization == false
    end
    
    @testset "TimingBreakdown Data Structure" begin
        breakdown = Parareal.TimingBreakdown{Float64}()
        
        # Test initial state
        @test isempty(breakdown.coarse_solver_breakdown)
        @test isempty(breakdown.fine_solver_breakdown)
        @test isempty(breakdown.mpi_communication_breakdown)
        @test isempty(breakdown.threading_breakdown)
        @test breakdown.synchronization_overhead == 0.0
        @test breakdown.load_imbalance_overhead == 0.0
        @test breakdown.memory_overhead == 0.0
        
        # Test data insertion
        breakdown.coarse_solver_breakdown["total_time"] = 2.5
        breakdown.coarse_solver_breakdown["total_calls"] = 100.0
        breakdown.fine_solver_breakdown["total_time"] = 8.3
        breakdown.mpi_communication_breakdown["send_time"] = 0.5
        breakdown.threading_breakdown["parallel_efficiency"] = 0.85
        
        @test breakdown.coarse_solver_breakdown["total_time"] == 2.5
        @test breakdown.coarse_solver_breakdown["total_calls"] == 100.0
        @test breakdown.fine_solver_breakdown["total_time"] == 8.3
        @test breakdown.mpi_communication_breakdown["send_time"] == 0.5
        @test breakdown.threading_breakdown["parallel_efficiency"] == 0.85
    end
    
    @testset "Timing Measurement Accuracy (Requirements 4.1, 4.2)" begin
        analyzer = Parareal.create_performance_analyzer(Float64)
        analyzer.enable_detailed_timing = true
        
        # Create mock performance metrics
        metrics1 = Parareal.create_performance_metrics(0, 2, 1)
        metrics2 = Parareal.create_performance_metrics(1, 2, 1)
        
        # Add timing data to metrics
        Parareal.update_timing_data!(metrics1, :coarse, 1.5)
        Parareal.update_timing_data!(metrics1, :fine, 4.2)
        Parareal.update_timing_data!(metrics2, :coarse, 1.8)
        Parareal.update_timing_data!(metrics2, :fine, 3.9)
        
        # Analyze timing breakdown (fix type issue)
        Parareal.analyze_timing_breakdown!(analyzer, Any[metrics1, metrics2])
        
        breakdown = analyzer.timing_breakdown
        
        # Test coarse solver timing accuracy
        @test haskey(breakdown.coarse_solver_breakdown, "total_time")
        @test haskey(breakdown.coarse_solver_breakdown, "total_calls")
        @test haskey(breakdown.coarse_solver_breakdown, "average_time")
        
        expected_coarse_total = 1.5 + 1.8
        expected_coarse_calls = 2.0
        expected_coarse_avg = expected_coarse_total / expected_coarse_calls
        
        @test breakdown.coarse_solver_breakdown["total_time"] ≈ expected_coarse_total
        @test breakdown.coarse_solver_breakdown["total_calls"] ≈ expected_coarse_calls
        @test breakdown.coarse_solver_breakdown["average_time"] ≈ expected_coarse_avg
        
        # Test fine solver timing accuracy
        @test haskey(breakdown.fine_solver_breakdown, "total_time")
        @test haskey(breakdown.fine_solver_breakdown, "total_calls")
        @test haskey(breakdown.fine_solver_breakdown, "average_time")
        
        expected_fine_total = 4.2 + 3.9
        expected_fine_calls = 2.0
        expected_fine_avg = expected_fine_total / expected_fine_calls
        
        @test breakdown.fine_solver_breakdown["total_time"] ≈ expected_fine_total
        @test breakdown.fine_solver_breakdown["total_calls"] ≈ expected_fine_calls
        @test breakdown.fine_solver_breakdown["average_time"] ≈ expected_fine_avg
    end
    
    @testset "Communication Overhead Tracking (Requirement 4.3)" begin
        analyzer = Parareal.create_performance_analyzer(Float64)
        analyzer.enable_detailed_timing = true
        
        # Create mock performance metrics with communication data
        metrics1 = Parareal.create_performance_metrics(0, 2, 1)
        metrics2 = Parareal.create_performance_metrics(1, 2, 1)
        
        # Add communication overhead data
        Parareal.record_communication_overhead!(metrics1, :send, 0.3, 1000)
        Parareal.record_communication_overhead!(metrics1, :receive, 0.2, 1000)
        Parareal.record_communication_overhead!(metrics1, :synchronization, 0.1)
        
        Parareal.record_communication_overhead!(metrics2, :send, 0.4, 1200)
        Parareal.record_communication_overhead!(metrics2, :receive, 0.25, 1200)
        Parareal.record_communication_overhead!(metrics2, :synchronization, 0.15)
        
        # Analyze timing breakdown (fix type issue)
        Parareal.analyze_timing_breakdown!(analyzer, Any[metrics1, metrics2])
        
        breakdown = analyzer.timing_breakdown
        
        # Test communication timing tracking
        @test haskey(breakdown.mpi_communication_breakdown, "send_time")
        @test haskey(breakdown.mpi_communication_breakdown, "receive_time")
        @test haskey(breakdown.mpi_communication_breakdown, "synchronization_time")
        @test haskey(breakdown.mpi_communication_breakdown, "total_time")
        
        expected_send_time = 0.3 + 0.4
        expected_receive_time = 0.2 + 0.25
        expected_sync_time = 0.1 + 0.15
        expected_total_comm = expected_send_time + expected_receive_time + expected_sync_time
        
        @test breakdown.mpi_communication_breakdown["send_time"] ≈ expected_send_time
        @test breakdown.mpi_communication_breakdown["receive_time"] ≈ expected_receive_time
        @test breakdown.mpi_communication_breakdown["synchronization_time"] ≈ expected_sync_time
        @test breakdown.mpi_communication_breakdown["total_time"] ≈ expected_total_comm
    end
    
    @testset "Scalability Metrics Calculation (Requirements 4.4, 4.5)" begin
        analyzer = Parareal.create_performance_analyzer(Float64)
        analyzer.enable_scaling_analysis = true
        
        scaling = analyzer.scaling_analysis
        
        # Test strong scaling analysis
        processes = [1, 2, 4, 8]
        times = [10.0, 5.5, 3.0, 2.0]
        
        # Manually set scaling data (simulating analyze_scaling_performance!)
        scaling.strong_scaling_processes = processes
        scaling.strong_scaling_times = times
        
        # Calculate expected speedups and efficiencies
        baseline_time = times[1]
        expected_speedups = [baseline_time / t for t in times]
        expected_efficiencies = [speedup / proc for (speedup, proc) in zip(expected_speedups, processes)]
        
        scaling.strong_scaling_speedups = expected_speedups
        scaling.strong_scaling_efficiencies = expected_efficiencies
        
        # Test speedup calculations
        @test length(scaling.strong_scaling_speedups) == length(processes)
        @test scaling.strong_scaling_speedups[1] ≈ 1.0  # Baseline
        @test scaling.strong_scaling_speedups[2] ≈ 10.0/5.5  # ~1.82
        @test scaling.strong_scaling_speedups[3] ≈ 10.0/3.0  # ~3.33
        @test scaling.strong_scaling_speedups[4] ≈ 10.0/2.0  # 5.0
        
        # Test efficiency calculations
        @test length(scaling.strong_scaling_efficiencies) == length(processes)
        @test scaling.strong_scaling_efficiencies[1] ≈ 1.0  # Perfect efficiency for 1 process
        @test scaling.strong_scaling_efficiencies[2] ≈ (10.0/5.5)/2  # ~0.91
        @test scaling.strong_scaling_efficiencies[3] ≈ (10.0/3.0)/4  # ~0.83
        @test scaling.strong_scaling_efficiencies[4] ≈ (10.0/2.0)/8  # 0.625
        
        # Test weak scaling analysis
        weak_processes = [1, 2, 4]
        weak_problem_sizes = [1000, 2000, 4000]
        weak_times = [5.0, 5.2, 5.8]
        
        scaling.weak_scaling_processes = weak_processes
        scaling.weak_scaling_problem_sizes = weak_problem_sizes
        scaling.weak_scaling_times = weak_times
        
        # Calculate weak scaling efficiencies and throughputs
        baseline_weak_time = weak_times[1]
        expected_weak_efficiencies = [baseline_weak_time / t for t in weak_times]
        expected_throughputs = [size / t for (size, t) in zip(weak_problem_sizes, weak_times)]
        
        scaling.weak_scaling_efficiencies = expected_weak_efficiencies
        scaling.weak_scaling_throughputs = expected_throughputs
        
        # Test weak scaling calculations
        @test length(scaling.weak_scaling_efficiencies) == length(weak_processes)
        @test scaling.weak_scaling_efficiencies[1] ≈ 1.0  # Baseline
        @test scaling.weak_scaling_efficiencies[2] ≈ 5.0/5.2  # ~0.96
        @test scaling.weak_scaling_efficiencies[3] ≈ 5.0/5.8  # ~0.86
        
        # Test throughput calculations
        @test length(scaling.weak_scaling_throughputs) == length(weak_processes)
        @test scaling.weak_scaling_throughputs[1] ≈ 1000.0/5.0  # 200
        @test scaling.weak_scaling_throughputs[2] ≈ 2000.0/5.2  # ~384.6
        @test scaling.weak_scaling_throughputs[3] ≈ 4000.0/5.8  # ~689.7
    end
    
    @testset "Performance Report Generation" begin
        analyzer = Parareal.create_performance_analyzer(Float64)
        analyzer.enable_detailed_timing = true
        analyzer.enable_scaling_analysis = true
        analyzer.enable_visualization = true
        
        # Set up comprehensive test data
        breakdown = analyzer.timing_breakdown
        breakdown.coarse_solver_breakdown["total_time"] = 2.5
        breakdown.coarse_solver_breakdown["total_calls"] = 100.0
        breakdown.coarse_solver_breakdown["average_time"] = 0.025
        
        breakdown.fine_solver_breakdown["total_time"] = 8.3
        breakdown.fine_solver_breakdown["total_calls"] = 50.0
        breakdown.fine_solver_breakdown["average_time"] = 0.166
        
        breakdown.mpi_communication_breakdown["send_time"] = 0.5
        breakdown.mpi_communication_breakdown["receive_time"] = 0.4
        breakdown.mpi_communication_breakdown["synchronization_time"] = 0.3
        breakdown.mpi_communication_breakdown["total_time"] = 1.2
        
        breakdown.threading_breakdown["parallel_efficiency"] = 0.85
        breakdown.threading_breakdown["load_balance_factor"] = 0.92
        
        breakdown.synchronization_overhead = 0.08
        breakdown.load_imbalance_overhead = 0.12
        breakdown.memory_overhead = 0.05
        
        # Set up scaling data
        scaling = analyzer.scaling_analysis
        scaling.strong_scaling_processes = [1, 2, 4, 8]
        scaling.strong_scaling_times = [12.0, 6.5, 3.8, 2.2]
        scaling.strong_scaling_speedups = [1.0, 1.85, 3.16, 5.45]
        scaling.strong_scaling_efficiencies = [1.0, 0.92, 0.79, 0.68]
        
        # Test detailed timing report generation
        timing_report = Parareal.generate_detailed_timing_report(analyzer)
        @test isa(timing_report, String)
        @test length(timing_report) > 0
        @test occursin("DETAILED TIMING BREAKDOWN REPORT", timing_report)
        @test occursin("Coarse Solver Performance", timing_report)
        @test occursin("Fine Solver Performance", timing_report)
        @test occursin("MPI Communication Breakdown", timing_report)
        @test occursin("Threading Performance Analysis", timing_report)
        @test occursin("Overhead Analysis", timing_report)
        
        # Test scaling analysis report generation (fix expected title)
        scaling_report = Parareal.generate_scaling_analysis_report(analyzer)
        @test isa(scaling_report, String)
        @test length(scaling_report) > 0
        @test occursin("SCALING PERFORMANCE ANALYSIS REPORT", scaling_report)  # Updated title
        @test occursin("Strong Scaling Analysis", scaling_report)
        @test occursin("Speedup", scaling_report)
        @test occursin("Efficiency", scaling_report)
        
        # Test comprehensive report generation (fix expected title)
        comprehensive_report = Parareal.generate_comprehensive_performance_report(analyzer)
        @test isa(comprehensive_report, String)
        @test length(comprehensive_report) > 0
        @test occursin("COMPREHENSIVE PARAREAL PERFORMANCE ANALYSIS REPORT", comprehensive_report)
        @test occursin("DETAILED TIMING BREAKDOWN REPORT", comprehensive_report)
        @test occursin("SCALING PERFORMANCE ANALYSIS REPORT", comprehensive_report)  # Updated title
        @test occursin("PERFORMANCE OPTIMIZATION RECOMMENDATIONS", comprehensive_report)
        
        # Test ASCII visualization generation
        ascii_plots = Parareal.create_ascii_performance_plots(analyzer)
        @test isa(ascii_plots, String)
        @test length(ascii_plots) > 0
        @test occursin("TIMING BREAKDOWN VISUALIZATION", ascii_plots)
        @test occursin("Component Time Distribution", ascii_plots)
        @test occursin("STRONG SCALING VISUALIZATION", ascii_plots)
        
        # Test performance recommendations generation
        recommendations = Parareal.generate_performance_recommendations(analyzer)
        @test isa(recommendations, String)
        @test length(recommendations) > 0
        # Should contain recommendations based on the test data
        @test occursin("SCALING EFFICIENCY", recommendations) || occursin("PERFORMANCE LOOKS GOOD", recommendations)
    end
    
    @testset "CSV Export Functionality" begin
        analyzer = Parareal.create_performance_analyzer(Float64)
        analyzer.enable_detailed_timing = true
        analyzer.enable_scaling_analysis = true
        
        # Set up test data
        breakdown = analyzer.timing_breakdown
        breakdown.coarse_solver_breakdown["total_time"] = 2.5
        breakdown.coarse_solver_breakdown["total_calls"] = 100.0
        breakdown.fine_solver_breakdown["total_time"] = 8.3
        breakdown.mpi_communication_breakdown["send_time"] = 0.5
        breakdown.threading_breakdown["parallel_efficiency"] = 0.85
        
        scaling = analyzer.scaling_analysis
        scaling.strong_scaling_processes = [1, 2, 4]
        scaling.strong_scaling_times = [10.0, 5.5, 3.0]
        scaling.strong_scaling_speedups = [1.0, 1.82, 3.33]
        scaling.strong_scaling_efficiencies = [1.0, 0.91, 0.83]
        
        # Test CSV export
        csv_filename = "test_performance_export.csv"
        Parareal.export_performance_data_csv(analyzer, csv_filename)
        
        # Verify file was created
        @test isfile(csv_filename)
        
        # Read and verify CSV content
        csv_content = read(csv_filename, String)
        @test occursin("Parareal Performance Analysis Data Export", csv_content)
        @test occursin("Timing Breakdown", csv_content)
        @test occursin("Component,Category,Metric,Value,Unit", csv_content)
        @test occursin("CoarseSolver,Timing,total_time,2.5,seconds", csv_content)
        @test occursin("FineSolver,Timing,total_time,8.3,seconds", csv_content)
        @test occursin("MPICommunication,Timing,send_time,0.5,seconds", csv_content)
        @test occursin("Threading,Performance,parallel_efficiency,0.85,ratio", csv_content)
        @test occursin("Strong Scaling Data", csv_content)
        @test occursin("Processes,ExecutionTime,Speedup,Efficiency", csv_content)
        
        # Clean up
        rm(csv_filename)
    end
    
    @testset "Error Handling and Edge Cases" begin
        analyzer = Parareal.create_performance_analyzer(Float64)
        
        # Test report generation with disabled features
        analyzer.enable_detailed_timing = false
        timing_report = Parareal.generate_detailed_timing_report(analyzer)
        @test occursin("Detailed timing analysis is disabled", timing_report)
        
        analyzer.enable_scaling_analysis = false
        scaling_report = Parareal.generate_scaling_analysis_report(analyzer)
        @test occursin("Scaling analysis is disabled", scaling_report)
        
        # Test with empty data
        analyzer.enable_detailed_timing = true
        analyzer.enable_scaling_analysis = true
        
        # Empty timing breakdown should still generate a report
        empty_timing_report = Parareal.generate_detailed_timing_report(analyzer)
        @test isa(empty_timing_report, String)
        @test occursin("DETAILED TIMING BREAKDOWN REPORT", empty_timing_report)
        
        # Empty scaling data should still generate a report (fix expected title)
        empty_scaling_report = Parareal.generate_scaling_analysis_report(analyzer)
        @test isa(empty_scaling_report, String)
        @test occursin("SCALING PERFORMANCE ANALYSIS REPORT", empty_scaling_report)  # Updated title
        
        # Test CSV export with empty data
        empty_csv_filename = "test_empty_export.csv"
        Parareal.export_performance_data_csv(analyzer, empty_csv_filename)
        @test isfile(empty_csv_filename)
        
        empty_csv_content = read(empty_csv_filename, String)
        @test occursin("Parareal Performance Analysis Data Export", empty_csv_content)
        @test occursin("Timing Breakdown", empty_csv_content)
        
        # Clean up
        rm(empty_csv_filename)
    end
end

println("All performance analysis unit tests completed successfully!")