# Test performance metrics functionality
using Test

# Add src to path for testing
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Include the Parareal module
include("../src/parareal.jl")
using .Parareal

@testset "Performance Metrics Tests" begin
    
    @testset "PerformanceMetrics Creation" begin
        # Test creating performance metrics
        metrics = Parareal.create_performance_metrics(0, 4, 2)
        
        @test metrics.process_rank == 0
        @test metrics.n_processes == 4
        @test metrics.n_threads_per_process == 2
        @test metrics.timing_data.coarse_solver_time == 0.0
        @test metrics.timing_data.fine_solver_time == 0.0
        @test metrics.communication_metrics.total_communication_time == 0.0
        @test metrics.efficiency_metrics.speedup_factor == 0.0
    end
    
    @testset "Timing Data Updates" begin
        metrics = Parareal.create_performance_metrics(0, 4, 2)
        
        # Test coarse solver timing
        Parareal.update_timing_data!(metrics, :coarse, 1.5)
        @test metrics.timing_data.coarse_solver_time == 1.5
        @test metrics.timing_data.coarse_solver_calls == 1
        
        # Test fine solver timing
        Parareal.update_timing_data!(metrics, :fine, 3.2)
        @test metrics.timing_data.fine_solver_time == 3.2
        @test metrics.timing_data.fine_solver_calls == 1
        
        # Test interpolation timing
        Parareal.update_timing_data!(metrics, :interpolation, 0.1)
        @test metrics.timing_data.interpolation_time == 0.1
        
        # Test restriction timing
        Parareal.update_timing_data!(metrics, :restriction, 0.05)
        @test metrics.timing_data.restriction_time == 0.05
        
        # Check total solver time
        expected_total = 1.5 + 3.2 + 0.1 + 0.05
        @test metrics.timing_data.total_solver_time ≈ expected_total
    end
    
    @testset "Communication Metrics" begin
        metrics = Parareal.create_performance_metrics(0, 4, 2)
        
        # Test send timing
        Parareal.record_communication_overhead!(metrics, :send, 0.2, 1024)
        @test metrics.communication_metrics.send_time == 0.2
        @test metrics.communication_metrics.message_count == 1
        @test metrics.communication_metrics.bytes_transferred == 1024
        
        # Test receive timing
        Parareal.record_communication_overhead!(metrics, :receive, 0.15, 512)
        @test metrics.communication_metrics.receive_time == 0.15
        @test metrics.communication_metrics.message_count == 2
        @test metrics.communication_metrics.bytes_transferred == 1536
        
        # Test synchronization timing
        Parareal.record_communication_overhead!(metrics, :synchronization, 0.05)
        @test metrics.communication_metrics.synchronization_time == 0.05
        
        # Check total communication time
        expected_total = 0.2 + 0.15 + 0.05
        @test metrics.communication_metrics.total_communication_time ≈ expected_total
    end
    
    @testset "Efficiency Calculations" begin
        metrics = Parareal.create_performance_metrics(0, 4, 2)
        
        # Set up some timing data
        metrics.total_wall_time = 10.0
        metrics.timing_data.total_solver_time = 8.0
        metrics.communication_metrics.total_communication_time = 1.5
        
        # Calculate efficiency metrics
        sequential_time = 35.0  # Hypothetical sequential time
        Parareal.calculate_efficiency_metrics!(metrics, sequential_time)
        
        # Test speedup calculation
        expected_speedup = sequential_time / metrics.total_wall_time
        @test metrics.efficiency_metrics.speedup_factor ≈ expected_speedup
        
        # Test parallel efficiency
        expected_efficiency = expected_speedup / metrics.n_processes
        @test metrics.efficiency_metrics.parallel_efficiency ≈ expected_efficiency
        
        # Test communication overhead ratio
        expected_comm_ratio = metrics.communication_metrics.total_communication_time / metrics.total_wall_time
        @test metrics.efficiency_metrics.communication_overhead_ratio ≈ expected_comm_ratio
        
        # Test load balance factor
        expected_load_balance = metrics.timing_data.total_solver_time / metrics.total_wall_time
        @test metrics.efficiency_metrics.load_balance_factor ≈ expected_load_balance
    end
    
    @testset "Performance Summary" begin
        metrics = Parareal.create_performance_metrics(0, 4, 2)
        
        # Add some data
        Parareal.update_timing_data!(metrics, :coarse, 2.0)
        Parareal.update_timing_data!(metrics, :fine, 5.0)
        Parareal.record_communication_overhead!(metrics, :send, 0.5, 2048)
        metrics.total_wall_time = 8.0
        metrics.parareal_iterations = 5
        
        # Get summary
        summary = Parareal.get_performance_summary(metrics)
        
        @test summary["process_rank"] == 0
        @test summary["n_processes"] == 4
        @test summary["n_threads_per_process"] == 2
        @test summary["parareal_iterations"] == 5
        @test summary["total_wall_time"] == 8.0
        @test summary["coarse_solver_time"] == 2.0
        @test summary["fine_solver_time"] == 5.0
        @test summary["total_solver_time"] == 7.0  # 2.0 + 5.0
        @test summary["send_time"] == 0.5
        @test summary["bytes_transferred"] == 2048
        @test summary["average_coarse_solver_time"] == 2.0  # 2.0 / 1 call
        @test summary["average_fine_solver_time"] == 5.0   # 5.0 / 1 call
    end
    
    @testset "Reset Functionality" begin
        metrics = Parareal.create_performance_metrics(0, 4, 2)
        
        # Add some data
        Parareal.update_timing_data!(metrics, :coarse, 2.0)
        Parareal.record_communication_overhead!(metrics, :send, 0.5, 1024)
        metrics.total_wall_time = 10.0
        
        # Reset metrics
        Parareal.reset_performance_metrics!(metrics)
        
        # Check that everything is reset
        @test metrics.timing_data.coarse_solver_time == 0.0
        @test metrics.timing_data.coarse_solver_calls == 0
        @test metrics.communication_metrics.send_time == 0.0
        @test metrics.communication_metrics.message_count == 0
        @test metrics.communication_metrics.bytes_transferred == 0
        @test metrics.total_wall_time == 0.0
        @test metrics.parareal_iterations == 0
        
        # Check that process info is preserved
        @test metrics.process_rank == 0
        @test metrics.n_processes == 4
        @test metrics.n_threads_per_process == 2
    end
    
    @testset "Merge Performance Metrics" begin
        # Create metrics for multiple processes
        metrics1 = Parareal.create_performance_metrics(0, 2, 1)
        metrics2 = Parareal.create_performance_metrics(1, 2, 1)
        
        # Add different data to each
        Parareal.update_timing_data!(metrics1, :coarse, 1.0)
        Parareal.update_timing_data!(metrics1, :fine, 2.0)
        metrics1.total_wall_time = 5.0
        
        Parareal.update_timing_data!(metrics2, :coarse, 1.5)
        Parareal.update_timing_data!(metrics2, :fine, 2.5)
        metrics2.total_wall_time = 6.0
        
        # Merge metrics
        merged = Parareal.merge_performance_metrics(Any[metrics1, metrics2])
        
        # Check merged results
        @test merged.timing_data.coarse_solver_time == 2.5  # 1.0 + 1.5
        @test merged.timing_data.fine_solver_time == 4.5    # 2.0 + 2.5
        @test merged.timing_data.coarse_solver_calls == 2   # 1 + 1
        @test merged.timing_data.fine_solver_calls == 2     # 1 + 1
        @test merged.total_wall_time == 6.0                 # max(5.0, 6.0)
        @test merged.process_rank == -1                     # Indicates merged
        @test merged.n_processes == 2                       # Number of merged processes
    end
    
    @testset "Measure and Record Function" begin
        metrics = Parareal.create_performance_metrics(0, 1, 1)
        
        # Test function that takes some time
        test_function = function(x)
            sleep(0.01)  # Sleep for 10ms
            return x * 2
        end
        
        # Measure and record coarse solver operation
        result = Parareal.measure_and_record!(metrics, :coarse, test_function, 5)
        
        @test result == 10  # 5 * 2
        @test metrics.timing_data.coarse_solver_time > 0.005  # Should be at least 5ms
        @test metrics.timing_data.coarse_solver_calls == 1
        
        # Measure and record communication operation
        comm_function = function()
            sleep(0.005)  # Sleep for 5ms
            return "done"
        end
        
        result2 = Parareal.measure_and_record!(metrics, :send, comm_function)
        
        @test result2 == "done"
        @test metrics.communication_metrics.send_time > 0.002  # Should be at least 2ms
        @test metrics.communication_metrics.message_count == 1
    end
end

println("Performance metrics tests completed successfully!")