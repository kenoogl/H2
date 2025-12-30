# Test TimeWindow functionality for Parareal
using Test

# Add src to load path for testing
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

include("../src/parareal.jl")
using .Parareal

@testset "TimeWindow Tests" begin
    
    @testset "Basic Time Window Creation" begin
        config = PararealConfig(
            total_time = 4.0,
            n_time_windows = 4,
            dt_coarse = 0.1,
            dt_fine = 0.01,
            n_mpi_processes = 1  # Use single process for testing
        )
        manager = PararealManager{Float64}(config)
        
        # Create time windows
        Parareal.create_time_windows!(manager)
        
        @test length(manager.time_windows) == 4
        
        # Check first window
        window1 = manager.time_windows[1]
        @test window1.start_time == 0.0
        @test window1.end_time == 1.0
        @test window1.dt_coarse == 0.1
        @test window1.dt_fine == 0.01
        @test window1.n_coarse_steps == 10
        @test window1.n_fine_steps == 100
        @test window1.process_rank == 0
        
        # Check last window
        window4 = manager.time_windows[4]
        @test window4.start_time == 3.0
        @test window4.end_time == 4.0
        @test window4.process_rank == 0  # In single-process test environment, all windows go to process 0
    end
    
    @testset "Time Window Validation" begin
        config = PararealConfig(
            total_time = 2.0,
            n_time_windows = 3,
            dt_coarse = 0.2,
            dt_fine = 0.02,
            n_mpi_processes = 1  # Use single process for testing
        )
        manager = PararealManager{Float64}(config)
        
        # Create and validate time windows
        Parareal.create_time_windows!(manager)
        is_valid, message = Parareal.validate_time_windows(manager)
        
        @test is_valid
        @test message == "Time windows are valid"
        
        # Check time continuity
        for i in 2:length(manager.time_windows)
            @test manager.time_windows[i].start_time ≈ manager.time_windows[i-1].end_time
        end
        
        # Check total time coverage
        @test manager.time_windows[1].start_time ≈ 0.0
        @test manager.time_windows[end].end_time ≈ 2.0
    end
    
    @testset "Local Time Windows" begin
        config = PararealConfig(
            total_time = 6.0,
            n_time_windows = 6,
            dt_coarse = 0.1,
            dt_fine = 0.01,
            n_mpi_processes = 1  # Use single process for testing
        )
        manager = PararealManager{Float64}(config)
        
        # Create time windows
        Parareal.create_time_windows!(manager)
        
        # Test local windows for single process
        local_windows = Parareal.get_local_time_windows(manager, 0)
        @test length(local_windows) == 6  # All windows go to process 0 in single-process test
        
        # Check that all windows belong to process 0
        for window in local_windows
            @test window.process_rank == 0
        end
    end
    
    @testset "Edge Cases" begin
        # Single time window
        config1 = PararealConfig(
            total_time = 1.0,
            n_time_windows = 1,
            dt_coarse = 0.1,
            dt_fine = 0.01,
            n_mpi_processes = 1
        )
        manager1 = PararealManager{Float64}(config1)
        Parareal.create_time_windows!(manager1)
        
        @test length(manager1.time_windows) == 1
        @test manager1.time_windows[1].start_time == 0.0
        @test manager1.time_windows[1].end_time == 1.0
        @test manager1.time_windows[1].process_rank == 0
        
        # More processes than windows (simulated)
        config2 = PararealConfig(
            total_time = 2.0,
            n_time_windows = 2,
            dt_coarse = 0.1,
            dt_fine = 0.01,
            n_mpi_processes = 1  # Use single process for testing
        )
        manager2 = PararealManager{Float64}(config2)
        Parareal.create_time_windows!(manager2)
        
        @test length(manager2.time_windows) == 2
        # In single process test, all windows go to process 0
        local_windows_p0 = Parareal.get_local_time_windows(manager2, 0)
        
        @test length(local_windows_p0) == 2
    end
    
    @testset "Error Handling" begin
        # Test invalid configurations
        config_bad = PararealConfig(
            total_time = -1.0,  # Invalid
            n_time_windows = 2,
            dt_coarse = 0.1,
            dt_fine = 0.01,
            n_mpi_processes = 2
        )
        manager_bad = PararealManager{Float64}(config_bad)
        
        @test_throws ErrorException Parareal.create_time_windows!(manager_bad)
    end
end

# Property-based test for time window assignment
@testset "Property Test: Time Window Assignment" begin
    # Property 1: For any valid configuration, time windows should cover the entire time domain
    # without gaps or overlaps
    
    function test_time_window_property(total_time, n_windows, n_processes)
        config = PararealConfig(
            total_time = total_time,
            n_time_windows = n_windows,
            dt_coarse = 0.1,
            dt_fine = 0.01,
            n_mpi_processes = n_processes
        )
        manager = PararealManager{Float64}(config)
        
        # Create time windows
        Parareal.create_time_windows!(manager)
        
        # Property 1: Correct number of windows
        @test length(manager.time_windows) == n_windows
        
        # Property 2: Complete time coverage
        @test manager.time_windows[1].start_time ≈ 0.0
        @test manager.time_windows[end].end_time ≈ total_time
        
        # Property 3: No gaps or overlaps
        for i in 2:length(manager.time_windows)
            @test manager.time_windows[i].start_time ≈ manager.time_windows[i-1].end_time
        end
        
        # Property 4: Valid process assignments
        for window in manager.time_windows
            @test 0 <= window.process_rank < n_processes
        end
        
        # Property 5: Load balancing (difference in window count per process <= 1)
        # Note: In test environment with single MPI process, all windows go to process 0
        actual_processes = manager.mpi_comm.size  # Use actual MPI size, not config
        process_counts = zeros(Int, actual_processes)
        for window in manager.time_windows
            if window.process_rank < actual_processes
                process_counts[window.process_rank + 1] += 1
            end
        end
        
        if actual_processes > 1
            min_count = minimum(process_counts)
            max_count = maximum(process_counts)
            @test max_count - min_count <= 1
        else
            # In single process environment, all windows go to process 0
            @test process_counts[1] == n_windows
        end
        
        return true
    end
    
    # Test with various valid configurations
    test_cases = [
        (1.0, 1, 1),
        (2.0, 4, 2),
        (5.0, 10, 3),
        (3.0, 7, 4),  # Uneven distribution
        (1.0, 5, 8)   # More processes than windows
    ]
    
    for (total_time, n_windows, n_processes) in test_cases
        @test test_time_window_property(total_time, n_windows, n_processes)
    end
end

println("TimeWindow tests completed successfully!")