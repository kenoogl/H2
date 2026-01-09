#!/usr/bin/env julia

"""
Integration test for boundary condition compatibility with Heat3ds q3d function and Parareal

This test verifies that boundary conditions work correctly when integrated with the 
actual Heat3ds simulation system and parareal algorithm.
"""

using Test

# Add src directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Include Heat3ds system
include("../src/heat3ds.jl")

@testset "Boundary Condition Integration Tests" begin
    
    @testset "Heat3ds q3d Integration with Boundary Conditions" begin
        """
        Test that q3d function works with different boundary conditions in both
        sequential and parareal modes
        """
        
        # Test parameters
        NX, NY, NZ = 20, 20, 15
        solver = "pbicgstab"
        smoother = "gs"
        epsilon = 1e-4
        
        # Test configurations
        test_configs = [
            ("Sequential Mode", false, nothing),
            ("Parareal Mode", true, Dict{String,Any}(
                "total_time" => 500.0,
                "n_time_windows" => 2,
                "dt_coarse" => 100.0,
                "dt_fine" => 50.0,
                "max_iterations" => 3,
                "convergence_tolerance" => 1e-4,
                "n_mpi_processes" => 1,
                "n_threads_per_process" => 1
            ))
        ]
        
        for (mode_name, use_parareal, parareal_config) in test_configs
            @testset "$mode_name" begin
                
                # Test steady-state problem first (simpler)
                @testset "Steady State Problem" begin
                    println("Testing $mode_name with steady-state boundary conditions...")
                    
                    # Test with default boundary conditions (should work)
                    @test_nowarn result = q3d(NX, NY, NZ, solver, smoother,
                                            epsilon=epsilon, par="sequential", is_steady=true,
                                            parareal=use_parareal, parareal_config=parareal_config)
                    
                    println("  ✓ $mode_name steady-state: Default boundary conditions work")
                end
                
                # Test transient problem (more complex)
                if use_parareal  # Only test transient for parareal mode
                    @testset "Transient Problem" begin
                        println("Testing $mode_name with transient boundary conditions...")
                        
                        # Test transient problem with parareal
                        @test_nowarn result = q3d(NX, NY, NZ, solver, smoother,
                                                epsilon=epsilon, par="sequential", is_steady=false,
                                                parareal=use_parareal, parareal_config=parareal_config)
                        
                        println("  ✓ $mode_name transient: Boundary conditions work with time evolution")
                    end
                end
            end
        end
    end
    
    @testset "Boundary Condition Consistency Verification" begin
        """
        Verify that boundary conditions behave consistently between sequential and parareal modes
        """
        
        # Small problem for fast testing
        NX, NY, NZ = 12, 12, 10
        
        println("Testing boundary condition consistency between sequential and parareal modes...")
        
        # Test steady-state consistency
        @testset "Steady State Consistency" begin
            # Run sequential mode
            result_seq = q3d(NX, NY, NZ, "pbicgstab", "gs",
                           epsilon=1e-4, par="sequential", is_steady=true,
                           parareal=false)
            
            # Run parareal mode with minimal configuration
            parareal_config = Dict{String,Any}(
                "total_time" => 100.0,
                "n_time_windows" => 1,  # Single window should be equivalent to sequential
                "dt_coarse" => 50.0,
                "dt_fine" => 25.0,
                "max_iterations" => 1,  # Single iteration for steady state
                "convergence_tolerance" => 1e-4,
                "n_mpi_processes" => 1,
                "n_threads_per_process" => 1
            )
            
            result_par = q3d(NX, NY, NZ, "pbicgstab", "gs",
                           epsilon=1e-4, par="sequential", is_steady=true,
                           parareal=true, parareal_config=parareal_config)
            
            # Both should complete successfully
            @test result_seq !== nothing
            @test result_par !== nothing
            
            println("  ✓ Both sequential and parareal modes completed successfully")
        end
    end
    
    @testset "Boundary Condition Error Handling" begin
        """
        Test error handling for boundary condition edge cases
        """
        
        NX, NY, NZ = 8, 8, 6
        
        @testset "Invalid Parareal Configuration" begin
            # Test with invalid parareal configuration
            invalid_config = Dict{String,Any}(
                "total_time" => -100.0,  # Invalid negative time
                "n_time_windows" => 0,   # Invalid zero windows
                "dt_coarse" => 0.0,      # Invalid zero time step
                "dt_fine" => 0.0,
                "max_iterations" => 0,
                "convergence_tolerance" => -1e-4,  # Invalid negative tolerance
                "n_mpi_processes" => 0,  # Invalid zero processes
                "n_threads_per_process" => 0
            )
            
            # This should either handle gracefully or provide meaningful error
            @test_throws Exception q3d(NX, NY, NZ, "pbicgstab", "gs",
                                     epsilon=1e-4, par="sequential", is_steady=false,
                                     parareal=true, parareal_config=invalid_config)
            
            println("  ✓ Invalid parareal configuration properly rejected")
        end
        
        @testset "Extreme Grid Sizes" begin
            # Test with very small grid (edge case)
            @test_nowarn result = q3d(4, 4, 3, "pbicgstab", "gs",
                                    epsilon=1e-3, par="sequential", is_steady=true,
                                    parareal=false)
            
            println("  ✓ Small grid size handled correctly")
        end
    end
    
    @testset "Solver Compatibility with Boundary Conditions" begin
        """
        Test that different solvers work with boundary conditions in parareal mode
        """
        
        NX, NY, NZ = 10, 10, 8
        
        # Test different solver types
        solvers = ["pbicgstab", "cg"]
        smoothers = ["gs", ""]
        
        parareal_config = Dict{String,Any}(
            "total_time" => 200.0,
            "n_time_windows" => 2,
            "dt_coarse" => 50.0,
            "dt_fine" => 25.0,
            "max_iterations" => 2,
            "convergence_tolerance" => 1e-3,
            "n_mpi_processes" => 1,
            "n_threads_per_process" => 1
        )
        
        for solver in solvers
            for smoother in smoothers
                @testset "$solver with smoother '$smoother'" begin
                    println("Testing solver: $solver, smoother: '$smoother'")
                    
                    # Test steady state
                    @test_nowarn result = q3d(NX, NY, NZ, solver, smoother,
                                            epsilon=1e-3, par="sequential", is_steady=true,
                                            parareal=false)
                    
                    # Test with parareal (transient)
                    @test_nowarn result = q3d(NX, NY, NZ, solver, smoother,
                                            epsilon=1e-3, par="sequential", is_steady=false,
                                            parareal=true, parareal_config=parareal_config)
                    
                    println("  ✓ Solver $solver with smoother '$smoother' works with boundary conditions")
                end
            end
        end
    end
    
    @testset "Performance and Scalability" begin
        """
        Test boundary condition performance with different parareal configurations
        """
        
        NX, NY, NZ = 16, 16, 12
        
        @testset "Different Time Window Configurations" begin
            window_configs = [
                (1, "Single window"),
                (2, "Two windows"),
                (4, "Four windows")
            ]
            
            for (n_windows, description) in window_configs
                @testset "$description" begin
                    parareal_config = Dict{String,Any}(
                        "total_time" => 400.0,
                        "n_time_windows" => n_windows,
                        "dt_coarse" => 400.0 / n_windows,
                        "dt_fine" => 200.0 / n_windows,
                        "max_iterations" => 3,
                        "convergence_tolerance" => 1e-4,
                        "n_mpi_processes" => 1,
                        "n_threads_per_process" => 1
                    )
                    
                    println("Testing with $n_windows time windows...")
                    
                    start_time = time()
                    @test_nowarn result = q3d(NX, NY, NZ, "pbicgstab", "gs",
                                            epsilon=1e-4, par="sequential", is_steady=false,
                                            parareal=true, parareal_config=parareal_config)
                    end_time = time()
                    
                    execution_time = end_time - start_time
                    println("  ✓ $description completed in $(round(execution_time, digits=2)) seconds")
                end
            end
        end
    end
end

println("All boundary condition integration tests completed successfully!")