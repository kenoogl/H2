# Test MPI initialization for Parareal
using Test
using MPI

# Add src to load path for testing
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

include("../src/parareal.jl")
using .Parareal

@testset "MPI Initialization Tests" begin
    
    @testset "PararealConfig Construction" begin
        # Test default constructor
        config = PararealConfig()
        @test config.total_time == 1.0
        @test config.n_time_windows == 4
        @test config.dt_coarse == 0.1
        @test config.dt_fine == 0.01
        @test config.time_step_ratio ≈ 10.0
        @test config.max_iterations == 10
        @test config.convergence_tolerance == 1e-6
        @test config.n_mpi_processes == 4
        @test config.n_threads_per_process == 1
        @test config.auto_optimize_parameters == false
        @test config.parameter_exploration_mode == false
    end
    
    @testset "PararealConfig Custom Parameters" begin
        # Test custom constructor
        config = PararealConfig(
            total_time = 2.0,
            n_time_windows = 8,
            dt_coarse = 0.2,
            dt_fine = 0.02,
            max_iterations = 20,
            convergence_tolerance = 1e-8,
            n_mpi_processes = 8,
            n_threads_per_process = 4
        )
        
        @test config.total_time == 2.0
        @test config.n_time_windows == 8
        @test config.dt_coarse == 0.2
        @test config.dt_fine == 0.02
        @test config.time_step_ratio ≈ 10.0
        @test config.max_iterations == 20
        @test config.convergence_tolerance == 1e-8
        @test config.n_mpi_processes == 8
        @test config.n_threads_per_process == 4
    end
    
    @testset "PararealManager Construction" begin
        config = PararealConfig()
        manager = PararealManager{Float64}(config)
        
        @test manager.config == config
        @test !manager.is_initialized
        @test isempty(manager.time_windows)
    end
    
    @testset "Time Window Creation" begin
        config = PararealConfig(
            total_time = 4.0,
            n_time_windows = 4,
            dt_coarse = 0.1,
            dt_fine = 0.01
        )
        manager = PararealManager{Float64}(config)
        
        # Manually create time windows for testing (without MPI)
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
        
        # Check last window
        window4 = manager.time_windows[4]
        @test window4.start_time == 3.0
        @test window4.end_time == 4.0
    end
    
    @testset "Time Step Ratio Validation" begin
        # Test various time step ratios
        ratios = [10, 50, 100]
        
        for ratio in ratios
            dt_fine = 0.01
            dt_coarse = ratio * dt_fine
            
            config = PararealConfig(
                dt_coarse = dt_coarse,
                dt_fine = dt_fine
            )
            
            @test config.time_step_ratio ≈ ratio
        end
    end
end

# Property-based test for MPI Environment Initialization Consistency
@testset "Property Test: MPI Environment Initialization Consistency" begin
    # Property 1: For any valid time domain specification and MPI configuration,
    # initializing parareal mode should successfully create the correct number of time windows
    
    function test_mpi_initialization_property(total_time, n_windows, dt_coarse, dt_fine)
        # Create configuration
        config = PararealConfig(
            total_time = total_time,
            n_time_windows = n_windows,
            dt_coarse = dt_coarse,
            dt_fine = dt_fine,
            n_mpi_processes = 1  # Use single process for testing
        )
        
        manager = PararealManager{Float64}(config)
        
        # Test time window creation (without actual MPI initialization)
        Parareal.create_time_windows!(manager)
        
        # Verify correct number of windows created
        @test length(manager.time_windows) == n_windows
        
        # Verify time domain coverage
        if n_windows > 0
            @test manager.time_windows[1].start_time ≈ 0.0
            @test manager.time_windows[end].end_time ≈ total_time
            
            # Verify continuous time coverage
            for i in 2:n_windows
                @test manager.time_windows[i].start_time ≈ manager.time_windows[i-1].end_time
            end
        end
        
        return true
    end
    
    # Test with various valid configurations
    test_cases = [
        (1.0, 2, 0.1, 0.01),
        (2.0, 4, 0.2, 0.02),
        (5.0, 10, 0.5, 0.05),
        (0.5, 1, 0.1, 0.01)
    ]
    
    for (total_time, n_windows, dt_coarse, dt_fine) in test_cases
        @test test_mpi_initialization_property(total_time, n_windows, dt_coarse, dt_fine)
    end
end

println("MPI initialization tests completed successfully!")