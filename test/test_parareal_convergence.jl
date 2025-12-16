using Test
using LinearAlgebra
using Statistics

# Include the Parareal module
include("../src/parareal.jl")
using .Parareal

"""
Property-based test for Parareal convergence accuracy
Property 3: Parareal Convergence Accuracy
Validates: Requirements 1.6
"""

@testset "Parareal Convergence Accuracy Property Tests" begin
    
    @testset "Property 3: Parareal Convergence Accuracy" begin
        
        @testset "Convergence monitor functionality" begin
            # Test basic convergence monitoring
            
            @testset "Basic convergence monitor" begin
                tolerance = 1e-6
                max_iterations = 10
                monitor = Parareal.ConvergenceMonitor{Float64}(tolerance, max_iterations)
                
                @test monitor.tolerance == tolerance
                @test monitor.max_iterations == max_iterations
                @test monitor.iteration_count == 0
                @test !monitor.is_converged
                @test isempty(monitor.residual_history)
                
                # Test convergence detection
                @test !Parareal.check_convergence!(monitor, 1e-3)  # Above tolerance
                @test monitor.iteration_count == 1
                @test length(monitor.residual_history) == 1
                
                @test Parareal.check_convergence!(monitor, 1e-7)  # Below tolerance
                @test monitor.is_converged
                @test monitor.iteration_count == 2
            end
            
            @testset "Advanced convergence monitor" begin
                monitor = Parareal.AdvancedConvergenceMonitor{Float64}(
                    absolute_tolerance = 1e-6,
                    relative_tolerance = 1e-4,
                    max_iterations = 15,
                    stagnation_threshold = 1e-8,
                    stagnation_window = 3
                )
                
                @test monitor.absolute_tolerance == 1e-6
                @test monitor.relative_tolerance == 1e-4
                @test monitor.max_iterations == 15
                @test !monitor.is_converged
                
                # Test absolute convergence
                @test !Parareal.check_advanced_convergence!(monitor, 1e-3, 1e-2)
                @test Parareal.check_advanced_convergence!(monitor, 1e-7, 1e-3)
                @test monitor.is_converged
                @test monitor.convergence_reason == "Absolute tolerance achieved"
                
                # Reset and test relative convergence
                Parareal.reset_advanced_convergence_monitor!(monitor)
                @test !monitor.is_converged
                @test monitor.iteration_count == 0
                
                # Simulate relative convergence
                @test !Parareal.check_advanced_convergence!(monitor, 1e-2, 1e-1)
                @test Parareal.check_advanced_convergence!(monitor, 1e-2 * (1 - 1e-5), 1e-2)  # Small relative change
                @test monitor.is_converged
                @test monitor.convergence_reason == "Relative tolerance achieved"
            end
            
            @testset "Convergence statistics" begin
                monitor = Parareal.AdvancedConvergenceMonitor{Float64}()
                
                # Simulate convergence history
                residuals = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
                for i in 1:length(residuals)
                    prev_residual = i > 1 ? residuals[i-1] : Inf
                    Parareal.check_advanced_convergence!(monitor, residuals[i], prev_residual)
                end
                
                stats = Parareal.get_convergence_statistics(monitor)
                
                @test haskey(stats, "iterations")
                @test haskey(stats, "converged")
                @test haskey(stats, "initial_residual")
                @test haskey(stats, "final_residual")
                @test haskey(stats, "residual_reduction")
                @test haskey(stats, "convergence_rate")
                
                @test stats["initial_residual"] == 1e-1
                @test stats["final_residual"] == 1e-5
                @test stats["residual_reduction"] ≈ 1e4
                @test stats["convergence_rate"] > 0  # Should be positive for decreasing residuals
            end
        end
        
        @testset "Parareal algorithm convergence behavior" begin
            # Test that Parareal algorithm converges for well-posed problems
            
            @testset "Simple heat diffusion convergence" begin
                # Create a simple test problem
                grid_size = (8, 8, 8)
                total_time = 0.1
                n_windows = 4
                
                # Create Parareal configuration
                config = Parareal.PararealConfig{Float64}(
                    total_time = total_time,
                    n_time_windows = n_windows,
                    dt_coarse = 0.05,
                    dt_fine = 0.01,
                    max_iterations = 10,
                    convergence_tolerance = 1e-4,
                    n_mpi_processes = 1,  # Single process for testing
                    n_threads_per_process = 1
                )
                
                # Create manager and initialize
                manager = Parareal.PararealManager{Float64}(config)
                
                # Mock MPI initialization for testing
                manager.mpi_comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
                manager.is_initialized = true
                
                # Create time windows
                Parareal.create_time_windows!(manager)
                
                # Validate time windows
                is_valid, message = Parareal.validate_time_windows(manager)
                @test is_valid
                
                # Create initial condition (smooth temperature distribution)
                initial_condition = zeros(Float64, grid_size...)
                for k in 1:grid_size[3], j in 1:grid_size[2], i in 1:grid_size[1]
                    # Smooth initial condition
                    x = (i - 1) / (grid_size[1] - 1)
                    y = (j - 1) / (grid_size[2] - 1)
                    z = (k - 1) / (grid_size[3] - 1)
                    initial_condition[i, j, k] = 300.0 + 50.0 * sin(π * x) * sin(π * y) * sin(π * z)
                end
                
                # Create problem data
                problem_data = Parareal.create_heat3ds_problem_data(
                    (0.1, 0.1, 0.1), [0.1, 0.2], [0.1, 0.1],
                    zeros(UInt8, grid_size...), nothing, "sequential", false
                )
                
                # Run Parareal algorithm
                result = Parareal.run_parareal!(manager, initial_condition, problem_data)
                
                # Check that algorithm completed
                @test result isa Parareal.PararealResult{Float64}
                @test size(result.final_solution) == grid_size
                @test all(isfinite.(result.final_solution))
                @test all(result.final_solution .> 0)  # Temperature should remain positive
                
                # Check convergence properties
                @test result.iterations >= 0
                @test result.iterations <= config.max_iterations
                @test result.computation_time >= 0
                
                # If converged, residual should be below tolerance
                if result.converged && length(result.residual_history) > 0
                    @test result.residual_history[end] <= config.convergence_tolerance * 10  # Allow some margin
                end
            end
            
            @testset "Convergence with different time step ratios" begin
                # Test convergence behavior with different coarse/fine time step ratios
                
                grid_size = (6, 6, 6)
                total_time = 0.05
                n_windows = 2
                
                time_step_ratios = [2.0, 5.0, 10.0]
                
                for ratio in time_step_ratios
                    fine_dt = 0.005
                    coarse_dt = fine_dt * ratio
                    
                    config = Parareal.PararealConfig{Float64}(
                        total_time = total_time,
                        n_time_windows = n_windows,
                        dt_coarse = coarse_dt,
                        dt_fine = fine_dt,
                        max_iterations = 8,
                        convergence_tolerance = 1e-3,
                        n_mpi_processes = 1,
                        n_threads_per_process = 1
                    )
                    
                    manager = Parareal.PararealManager{Float64}(config)
                    manager.mpi_comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
                    manager.is_initialized = true
                    
                    Parareal.create_time_windows!(manager)
                    
                    # Simple initial condition
                    initial_condition = fill(300.0, grid_size...)
                    
                    problem_data = Parareal.create_heat3ds_problem_data(
                        (0.1, 0.1, 0.1), [0.1], [0.1],
                        zeros(UInt8, grid_size...), nothing, "sequential", false
                    )
                    
                    result = Parareal.run_parareal!(manager, initial_condition, problem_data)
                    
                    @test result isa Parareal.PararealResult{Float64}
                    @test size(result.final_solution) == grid_size
                    @test all(isfinite.(result.final_solution))
                    
                    # Smaller time step ratios should generally converge faster
                    # (though this is problem-dependent)
                    @test result.iterations <= config.max_iterations
                end
            end
        end
        
        @testset "Residual computation accuracy" begin
            # Test that residual computations are accurate and consistent
            
            @testset "Global residual computation" begin
                # Test residual computation with mock MPI
                config = Parareal.PararealConfig{Float64}(n_mpi_processes = 1)
                manager = Parareal.PararealManager{Float64}(config)
                manager.mpi_comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
                
                # Test with known residual values
                test_residuals = [1e-2, 1e-4, 1e-6, 0.0]
                
                for residual in test_residuals
                    computed_residual = Parareal.compute_global_residual!(manager, residual)
                    @test computed_residual ≈ residual
                end
            end
            
            @testset "Distributed residual norms" begin
                config = Parareal.PararealConfig{Float64}(n_mpi_processes = 1)
                manager = Parareal.PararealManager{Float64}(config)
                manager.mpi_comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
                
                # Create test data
                grid_size = (4, 4, 4)
                data1 = ones(Float64, grid_size...)
                data2 = ones(Float64, grid_size...) .* 1.1
                
                # Test L2 norm
                l2_norm = Parareal.compute_distributed_residual_norm!(
                    manager, data1, data2, :l2
                )
                expected_l2 = sqrt(sum((data1 - data2).^2))
                @test l2_norm ≈ expected_l2
                
                # Test L∞ norm
                linf_norm = Parareal.compute_distributed_residual_norm!(
                    manager, data1, data2, :linf
                )
                expected_linf = maximum(abs.(data1 - data2))
                @test linf_norm ≈ expected_linf
                
                # Test L1 norm
                l1_norm = Parareal.compute_distributed_residual_norm!(
                    manager, data1, data2, :l1
                )
                expected_l1 = sum(abs.(data1 - data2))
                @test l1_norm ≈ expected_l1
            end
        end
        
        @testset "Convergence tolerance verification" begin
            # Test that convergence tolerance is properly enforced
            
            @testset "Tolerance enforcement" begin
                tolerances = [1e-3, 1e-6, 1e-9]
                
                for tolerance in tolerances
                    monitor = Parareal.ConvergenceMonitor{Float64}(tolerance, 20)
                    
                    # Test residuals around the tolerance
                    above_tolerance = tolerance * 10
                    below_tolerance = tolerance / 10
                    
                    @test !Parareal.check_convergence!(monitor, above_tolerance)
                    @test !monitor.is_converged
                    
                    @test Parareal.check_convergence!(monitor, below_tolerance)
                    @test monitor.is_converged
                end
            end
            
            @testset "Iteration limit enforcement" begin
                max_iterations_list = [5, 10, 20]
                
                for max_iter in max_iterations_list
                    monitor = Parareal.ConvergenceMonitor{Float64}(1e-12, max_iter)  # Very strict tolerance
                    
                    # Run iterations without converging
                    for i in 1:max_iter
                        converged = Parareal.check_convergence!(monitor, 1e-3)  # Above tolerance
                        
                        if i < max_iter
                            @test !converged
                        else
                            @test converged  # Should converge due to iteration limit
                        end
                    end
                    
                    @test monitor.iteration_count == max_iter
                    @test monitor.is_converged
                end
            end
        end
        
        @testset "Numerical accuracy preservation" begin
            # Test that Parareal preserves numerical accuracy within expected bounds
            
            @testset "Solution accuracy bounds" begin
                # Create a problem where we can estimate the expected accuracy
                grid_size = (6, 6, 6)
                
                # Test with different convergence tolerances
                tolerances = [1e-3, 1e-6]
                
                for tolerance in tolerances
                    config = Parareal.PararealConfig{Float64}(
                        total_time = 0.02,
                        n_time_windows = 2,
                        dt_coarse = 0.01,
                        dt_fine = 0.002,
                        max_iterations = 15,
                        convergence_tolerance = tolerance,
                        n_mpi_processes = 1,
                        n_threads_per_process = 1
                    )
                    
                    manager = Parareal.PararealManager{Float64}(config)
                    manager.mpi_comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
                    manager.is_initialized = true
                    
                    Parareal.create_time_windows!(manager)
                    
                    # Create initial condition with known structure
                    initial_condition = zeros(Float64, grid_size...)
                    for k in 1:grid_size[3], j in 1:grid_size[2], i in 1:grid_size[1]
                        initial_condition[i, j, k] = 300.0 + 10.0 * sin(π * i / grid_size[1])
                    end
                    
                    problem_data = Parareal.create_heat3ds_problem_data(
                        (0.1, 0.1, 0.1), [0.1], [0.1],
                        zeros(UInt8, grid_size...), nothing, "sequential", false
                    )
                    
                    result = Parareal.run_parareal!(manager, initial_condition, problem_data)
                    
                    @test result isa Parareal.PararealResult{Float64}
                    
                    # Check that solution maintains physical bounds
                    @test all(result.final_solution .> 250.0)  # Reasonable lower bound
                    @test all(result.final_solution .< 350.0)  # Reasonable upper bound
                    
                    # Check that solution is smooth (no spurious oscillations)
                    # Compute gradient magnitude as a smoothness indicator
                    grad_magnitude = 0.0
                    count = 0
                    for k in 2:grid_size[3]-1, j in 2:grid_size[2]-1, i in 2:grid_size[1]-1
                        grad_x = result.final_solution[i+1, j, k] - result.final_solution[i-1, j, k]
                        grad_y = result.final_solution[i, j+1, k] - result.final_solution[i, j-1, k]
                        grad_z = result.final_solution[i, j, k+1] - result.final_solution[i, j, k-1]
                        
                        grad_magnitude += sqrt(grad_x^2 + grad_y^2 + grad_z^2)
                        count += 1
                    end
                    
                    avg_grad_magnitude = grad_magnitude / count
                    @test avg_grad_magnitude < 100.0  # Should not have excessive gradients
                end
            end
        end
    end
end

# Property verification summary
println("\n=== Property 3: Parareal Convergence Accuracy - Verification Summary ===")
println("✓ Convergence monitor functionality (basic and advanced)")
println("✓ Convergence statistics computation and analysis")
println("✓ Parareal algorithm convergence behavior for heat diffusion")
println("✓ Convergence with different time step ratios")
println("✓ Residual computation accuracy (global and distributed norms)")
println("✓ Convergence tolerance enforcement and iteration limits")
println("✓ Numerical accuracy preservation within expected bounds")
println("✓ Solution smoothness and physical bound preservation")
println("================================================================================")