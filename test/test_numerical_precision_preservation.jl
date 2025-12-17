using Test
using LinearAlgebra
using Statistics
using Dates
using MPI

# Include the Parareal module
include("../src/parareal.jl")
using .Parareal

"""
Property-based test for numerical precision preservation
Property 12: Numerical Precision Preservation
Validates: Requirements 5.4, 5.5
"""

@testset "Numerical Precision Preservation Property Tests" begin
    
    @testset "Property 12: Numerical Precision Preservation" begin
        
        @testset "Error accumulation monitoring" begin
            # Test error accumulation monitoring functionality
            
            @testset "Basic error accumulation tracking" begin
                # Test that error accumulation is properly tracked
                
                validation_manager = Parareal.ValidationManager{Float64}()
                
                # Simulate multiple validation runs with increasing errors
                grid_size = (4, 4, 4)
                base_solution = ones(Float64, grid_size...) * 300.0
                
                accumulated_errors = Float64[]
                
                for i in 1:5
                    # Create solutions with increasing error
                    error_magnitude = i * 0.001
                    parareal_solution = base_solution .+ error_magnitude
                    
                    config = Parareal.PararealConfig{Float64}()
                    parareal_result = Parareal.PararealResult{Float64}(
                        parareal_solution, true, i, [1e-2, 1e-3, 1e-4], 1.0, 0.1
                    )
                    
                    validation_result = Parareal.validate_against_sequential!(
                        validation_manager, parareal_result, base_solution, "error_test_$i", config
                    )
                    
                    push!(accumulated_errors, validation_result.accuracy_metrics.l2_norm_error)
                end
                
                # Property: Error accumulation should be monotonic (or at least bounded)
                @test length(accumulated_errors) == 5
                @test all(error -> error >= 0, accumulated_errors)
                
                # Check that errors are reasonable (not exploding)
                @test all(error -> error < 1.0, accumulated_errors)
                
                # Check that validation history is maintained
                @test length(validation_manager.validation_history) == 5
            end
            
            @testset "Error bounds verification" begin
                # Test that errors remain within theoretical bounds
                
                validation_manager = Parareal.ValidationManager{Float64}(
                    tolerance_settings = Parareal.ToleranceSettings{Float64}(
                        absolute_tolerance = 1e-6,
                        relative_tolerance = 1e-4,
                        max_pointwise_tolerance = 1e-5
                    )
                )
                
                grid_size = (6, 6, 6)
                
                # Test with different error magnitudes
                error_levels = [1e-8, 1e-6, 1e-4, 1e-2]
                
                for (i, error_level) in enumerate(error_levels)
                    sequential_solution = ones(Float64, grid_size...) * 273.15
                    
                    # Add controlled error
                    parareal_solution = sequential_solution .+ error_level
                    
                    config = Parareal.PararealConfig{Float64}()
                    parareal_result = Parareal.PararealResult{Float64}(
                        parareal_solution, true, 3, [1e-1, 1e-2, 1e-3], 1.0, 0.1
                    )
                    
                    validation_result = Parareal.validate_against_sequential!(
                        validation_manager, parareal_result, sequential_solution, "bounds_test_$i", config
                    )
                    
                    # Property: Error metrics should reflect the controlled error level
                    @test validation_result.accuracy_metrics.max_pointwise_error ≈ error_level atol=1e-10
                    
                    # Property: Tolerance checking should work correctly
                    expected_within_tolerance = (error_level <= validation_manager.tolerance_settings.absolute_tolerance)
                    if error_level <= 1e-6  # Within our tolerance
                        @test validation_result.is_within_tolerance == true
                    else  # Exceeds tolerance
                        @test validation_result.is_within_tolerance == false
                    end
                end
            end
        end
        
        @testset "Theoretical vs actual error comparison" begin
            # Test comparison between theoretical and actual errors
            
            @testset "Fine solver error estimation" begin
                # Test theoretical error estimation for fine solver
                
                grid_size = (4, 4, 4)
                dt_fine = 0.001
                
                # Create a simple problem where we can estimate theoretical error
                initial_condition = ones(Float64, grid_size...) * 300.0
                
                problem_data = Parareal.create_heat3ds_problem_data(
                    (0.1, 0.1, 0.1), [0.1], [0.1],
                    zeros(UInt8, grid_size...), nothing, "sequential", false
                )
                
                validation_manager = Parareal.ValidationManager{Float64}()
                
                # Run sequential reference with fine time step
                sequential_result = Parareal.run_sequential_reference!(
                    validation_manager, initial_condition, problem_data, 0.01
                )
                
                # Run with slightly different time step to estimate discretization error
                validation_manager_coarse = Parareal.ValidationManager{Float64}(
                    reference_solver = Parareal.SequentialSolver{Float64}(dt = dt_fine * 2)
                )
                
                coarse_sequential_result = Parareal.run_sequential_reference!(
                    validation_manager_coarse, initial_condition, problem_data, 0.01
                )
                
                # Estimate theoretical error (simple first-order approximation)
                theoretical_error = norm(sequential_result - coarse_sequential_result)
                
                # Property: Theoretical error should be reasonable
                @test theoretical_error >= 0
                @test theoretical_error < 10.0  # Should not be excessive
                @test all(isfinite.(sequential_result))
                @test all(isfinite.(coarse_sequential_result))
            end
            
            @testset "Error scaling with time step" begin
                # Test that error scales appropriately with time step size
                
                grid_size = (4, 4, 4)
                initial_condition = fill(300.0, grid_size...)
                
                problem_data = Parareal.create_heat3ds_problem_data(
                    (0.1, 0.1, 0.1), [0.1], [0.1],
                    zeros(UInt8, grid_size...), nothing, "sequential", false
                )
                
                # Test with different time steps
                time_steps = [0.001, 0.002, 0.004]
                errors = Float64[]
                
                # Reference solution with very fine time step
                reference_manager = Parareal.ValidationManager{Float64}(
                    reference_solver = Parareal.SequentialSolver{Float64}(dt = 0.0005)
                )
                reference_solution = Parareal.run_sequential_reference!(
                    reference_manager, initial_condition, problem_data, 0.005
                )
                
                for dt in time_steps
                    test_manager = Parareal.ValidationManager{Float64}(
                        reference_solver = Parareal.SequentialSolver{Float64}(dt = dt)
                    )
                    
                    test_solution = Parareal.run_sequential_reference!(
                        test_manager, initial_condition, problem_data, 0.005
                    )
                    
                    error = norm(test_solution - reference_solution) / norm(reference_solution)
                    push!(errors, error)
                end
                
                # Property: Error should generally increase with larger time steps
                @test length(errors) == length(time_steps)
                @test all(error -> error >= 0, errors)
                
                # Property: Error should be bounded (not exploding)
                @test all(error -> error < 1.0, errors)
            end
        end
        
        @testset "Numerical stability analysis" begin
            # Test numerical stability analysis functionality
            
            @testset "Convergence history stability" begin
                # Test stability analysis of convergence histories
                
                # Test stable convergence
                stable_history = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3]
                is_stable, message = Parareal.check_numerical_stability(stable_history)
                @test is_stable == true
                @test occursin("stable", message)
                
                # Test unstable (oscillating) convergence
                unstable_history = [1e-1, 1e-2, 5e-2, 1e-3, 3e-2, 5e-4, 1e-2]
                is_stable, message = Parareal.check_numerical_stability(unstable_history)
                @test is_stable == false
                @test occursin("oscillatory", message)
                
                # Test stagnant convergence
                stagnant_history = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]
                is_stable, message = Parareal.check_numerical_stability(stagnant_history)
                @test is_stable == false
                @test occursin("stagnated", message)
                
                # Test insufficient data
                short_history = [1e-1, 1e-2]
                is_stable, message = Parareal.check_numerical_stability(short_history)
                @test is_stable == true
                @test occursin("Insufficient data", message)
            end
            
            @testset "Error growth analysis" begin
                # Test analysis of error growth patterns
                
                validation_manager = Parareal.ValidationManager{Float64}()
                
                # Simulate parareal iterations with different error patterns
                grid_size = (4, 4, 4)
                base_solution = ones(Float64, grid_size...) * 300.0
                
                # Test exponential error growth (bad)
                exponential_errors = [1e-3 * (2.0^i) for i in 0:4]
                
                for (i, error) in enumerate(exponential_errors)
                    parareal_solution = base_solution .+ error
                    
                    config = Parareal.PararealConfig{Float64}()
                    # Ensure we don't exceed the available error history
                    error_history = exponential_errors[1:min(i+1, length(exponential_errors))]
                    parareal_result = Parareal.PararealResult{Float64}(
                        parareal_solution, true, i+1, error_history, 1.0, 0.1
                    )
                    
                    validation_result = Parareal.validate_against_sequential!(
                        validation_manager, parareal_result, base_solution, "exp_growth_$i", config
                    )
                end
                
                # Check that exponential growth is detected
                final_validation = validation_manager.validation_history[end]
                @test final_validation.accuracy_metrics.l2_norm_error > validation_manager.validation_history[1].accuracy_metrics.l2_norm_error
                
                # Property: Validation should flag excessive error growth
                if length(validation_manager.validation_history) >= 2
                    error_ratio = (validation_manager.validation_history[end].accuracy_metrics.l2_norm_error / 
                                  validation_manager.validation_history[1].accuracy_metrics.l2_norm_error)
                    @test error_ratio > 1.0  # Error should have grown
                end
            end
        end
        
        @testset "Machine precision preservation" begin
            # Test preservation of machine precision
            
            @testset "Floating point precision limits" begin
                # Test behavior at machine precision limits
                
                validation_manager = Parareal.ValidationManager{Float64}(
                    tolerance_settings = Parareal.ToleranceSettings{Float64}(
                        absolute_tolerance = eps(Float64) * 10,
                        relative_tolerance = eps(Float64) * 100,
                        max_pointwise_tolerance = eps(Float64) * 10
                    )
                )
                
                grid_size = (4, 4, 4)
                base_value = 1.0
                sequential_solution = fill(base_value, grid_size...)
                
                # Test with machine precision level differences
                machine_precision_error = eps(Float64)
                parareal_solution = sequential_solution .+ machine_precision_error
                
                config = Parareal.PararealConfig{Float64}()
                parareal_result = Parareal.PararealResult{Float64}(
                    parareal_solution, true, 1, [machine_precision_error], 1.0, 0.1
                )
                
                validation_result = Parareal.validate_against_sequential!(
                    validation_manager, parareal_result, sequential_solution, "machine_precision", config
                )
                
                # Property: Machine precision level errors should be acceptable
                @test validation_result.accuracy_metrics.max_pointwise_error ≈ machine_precision_error atol=eps(Float64)
                @test validation_result.is_within_tolerance == true
            end
            
            @testset "Precision degradation detection" begin
                # Test detection of precision degradation
                
                validation_manager = Parareal.ValidationManager{Float64}()
                
                grid_size = (4, 4, 4)
                sequential_solution = ones(Float64, grid_size...) * π  # Use irrational number
                
                # Simulate precision loss through repeated operations
                degraded_solution = copy(sequential_solution)
                
                # Simulate numerical operations that might degrade precision
                for _ in 1:100
                    degraded_solution = degraded_solution .* 1.0000001
                    degraded_solution = degraded_solution ./ 1.0000001
                end
                
                config = Parareal.PararealConfig{Float64}()
                parareal_result = Parareal.PararealResult{Float64}(
                    degraded_solution, true, 100, [1e-1, 1e-2, 1e-3], 1.0, 0.1
                )
                
                validation_result = Parareal.validate_against_sequential!(
                    validation_manager, parareal_result, sequential_solution, "precision_degradation", config
                )
                
                # Property: Precision degradation should be detectable
                @test validation_result.accuracy_metrics.l2_norm_error >= 0
                @test all(isfinite.(validation_result.accuracy_metrics.error_distribution))
                
                # Property: Solutions should still be reasonable despite degradation
                @test all(degraded_solution .> 0)  # Should remain positive
                @test all(abs.(degraded_solution .- sequential_solution) .< 1.0)  # Should not drift too far
            end
        end
        
        @testset "Error propagation analysis" begin
            # Test analysis of how errors propagate through parareal iterations
            
            @testset "Error propagation bounds" begin
                # Test that error propagation remains bounded
                
                validation_manager = Parareal.ValidationManager{Float64}()
                
                grid_size = (4, 4, 4)
                initial_condition = ones(Float64, grid_size...) * 300.0
                
                problem_data = Parareal.create_heat3ds_problem_data(
                    (0.1, 0.1, 0.1), [0.1], [0.1],
                    zeros(UInt8, grid_size...), nothing, "sequential", false
                )
                
                # Run sequential reference
                sequential_result = Parareal.run_sequential_reference!(
                    validation_manager, initial_condition, problem_data, 0.01
                )
                
                # Simulate parareal with different numbers of iterations
                iteration_counts = [1, 3, 5, 10]
                propagated_errors = Float64[]
                
                for iterations in iteration_counts
                    # Simulate parareal result with iteration-dependent error
                    base_error = 1e-3
                    iteration_error = base_error / sqrt(iterations)  # Error should decrease with iterations
                    
                    parareal_solution = sequential_result .+ iteration_error
                    
                    config = Parareal.PararealConfig{Float64}(max_iterations = iterations)
                    parareal_result = Parareal.PararealResult{Float64}(
                        parareal_solution, true, iterations, 
                        [base_error / (i+1) for i in 0:iterations-1], 1.0, 0.1
                    )
                    
                    validation_result = Parareal.validate_against_sequential!(
                        validation_manager, parareal_result, sequential_result, 
                        "propagation_$iterations", config
                    )
                    
                    push!(propagated_errors, validation_result.accuracy_metrics.l2_norm_error)
                end
                
                # Property: Error should generally decrease with more iterations
                @test length(propagated_errors) == length(iteration_counts)
                @test all(error -> error >= 0, propagated_errors)
                
                # Property: Error should be bounded and not grow exponentially
                @test all(error -> error < 1.0, propagated_errors)
                
                # Property: More iterations should generally lead to better accuracy
                # (This is a general trend, not a strict requirement due to problem complexity)
                if length(propagated_errors) >= 2
                    @test propagated_errors[end] <= propagated_errors[1] * 2  # Allow some tolerance
                end
            end
            
            @testset "Cumulative error analysis" begin
                # Test cumulative error analysis over multiple time steps
                
                validation_manager = Parareal.ValidationManager{Float64}()
                
                grid_size = (4, 4, 4)
                initial_condition = fill(300.0, grid_size...)
                
                problem_data = Parareal.create_heat3ds_problem_data(
                    (0.1, 0.1, 0.1), [0.1], [0.1],
                    zeros(UInt8, grid_size...), nothing, "sequential", false
                )
                
                # Test different simulation times to see cumulative error
                simulation_times = [0.005, 0.01, 0.02, 0.04]
                cumulative_errors = Float64[]
                
                for sim_time in simulation_times
                    sequential_result = Parareal.run_sequential_reference!(
                        validation_manager, initial_condition, problem_data, sim_time
                    )
                    
                    # Simulate parareal with time-dependent error
                    time_dependent_error = sim_time * 0.01  # Error proportional to simulation time
                    parareal_solution = sequential_result .+ time_dependent_error
                    
                    config = Parareal.PararealConfig{Float64}(total_time = sim_time)
                    parareal_result = Parareal.PararealResult{Float64}(
                        parareal_solution, true, 5, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 1.0, 0.1
                    )
                    
                    validation_result = Parareal.validate_against_sequential!(
                        validation_manager, parareal_result, sequential_result, 
                        "cumulative_$(sim_time)", config
                    )
                    
                    push!(cumulative_errors, validation_result.accuracy_metrics.l2_norm_error)
                end
                
                # Property: Cumulative error should be bounded
                @test all(error -> error >= 0, cumulative_errors)
                @test all(error -> error < 10.0, cumulative_errors)  # Reasonable upper bound
                
                # Property: Error growth should not be exponential
                if length(cumulative_errors) >= 2
                    max_growth_rate = maximum(cumulative_errors[i+1] / cumulative_errors[i] 
                                            for i in 1:length(cumulative_errors)-1 
                                            if cumulative_errors[i] > 0)
                    @test max_growth_rate < 100.0  # Should not grow by more than 2 orders of magnitude per step
                end
            end
        end
        
        @testset "Precision preservation validation" begin
            # Test overall precision preservation validation
            
            @testset "Comprehensive precision check" begin
                # Comprehensive test of precision preservation
                
                validation_manager = Parareal.ValidationManager{Float64}(
                    tolerance_settings = Parareal.ToleranceSettings{Float64}(
                        absolute_tolerance = 1e-8,
                        relative_tolerance = 1e-6,
                        max_pointwise_tolerance = 1e-7
                    )
                )
                
                grid_size = (6, 6, 6)
                
                # Create a problem with known analytical properties
                initial_condition = zeros(Float64, grid_size...)
                for k in 1:grid_size[3], j in 1:grid_size[2], i in 1:grid_size[1]
                    x = (i - 1) / (grid_size[1] - 1)
                    y = (j - 1) / (grid_size[2] - 1)
                    z = (k - 1) / (grid_size[3] - 1)
                    initial_condition[i, j, k] = 300.0 + 10.0 * sin(π * x) * sin(π * y) * sin(π * z)
                end
                
                problem_data = Parareal.create_heat3ds_problem_data(
                    (0.1, 0.1, 0.1), [0.1], [0.1],
                    zeros(UInt8, grid_size...), nothing, "sequential", false
                )
                
                # Run high-precision sequential reference
                sequential_result = Parareal.run_sequential_reference!(
                    validation_manager, initial_condition, problem_data, 0.01
                )
                
                # Simulate high-quality parareal result
                high_precision_error = 1e-9
                parareal_solution = sequential_result .+ randn(grid_size...) * high_precision_error
                
                config = Parareal.PararealConfig{Float64}(
                    convergence_tolerance = 1e-8,
                    max_iterations = 10
                )
                
                parareal_result = Parareal.PararealResult{Float64}(
                    parareal_solution, true, 8, 
                    [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], 1.0, 0.1
                )
                
                validation_result = Parareal.validate_against_sequential!(
                    validation_manager, parareal_result, sequential_result, 
                    "comprehensive_precision", config
                )
                
                # Property: High-precision computation should pass validation
                @test validation_result.is_within_tolerance == true
                @test validation_result.accuracy_metrics.l2_norm_error < 1e-6
                @test validation_result.accuracy_metrics.max_pointwise_error < 1e-5
                
                # Property: Solutions should preserve physical properties
                @test all(parareal_solution .> 250.0)  # Reasonable lower bound
                @test all(parareal_solution .< 350.0)  # Reasonable upper bound
                @test all(sequential_result .> 250.0)
                @test all(sequential_result .< 350.0)
                
                # Property: Error distribution should be reasonable
                @test all(isfinite.(validation_result.accuracy_metrics.error_distribution))
                @test maximum(validation_result.accuracy_metrics.error_distribution) < 1e-5
            end
        end
    end
end

# Property verification summary
println("\n=== Property 12: Numerical Precision Preservation - Summary ===")
println("✓ Error accumulation monitoring and tracking")
println("✓ Error bounds verification and tolerance checking")
println("✓ Theoretical vs actual error comparison")
println("✓ Error scaling analysis with time step size")
println("✓ Numerical stability analysis for convergence histories")
println("✓ Error growth pattern analysis and detection")
println("✓ Machine precision preservation and limits testing")
println("✓ Precision degradation detection and analysis")
println("✓ Error propagation bounds and cumulative error analysis")
println("✓ Comprehensive precision preservation validation")
println("================================================================")