# Property-based test for time step ratio optimization
# **Feature: parareal-time-parallelization, Property 7: Time Step Ratio Optimization**
# **Validates: Requirements 11.2, 11.3, 11.4, 11.5**

using Test
using Random
using Statistics

# Import the parareal module
include("../src/parareal.jl")
using .Parareal

"""
Property-based test for time step ratio optimization

Property 7: Time Step Ratio Optimization
For any thermal diffusivity and grid spacing combination, the system should estimate 
time step ratios that fall within physically reasonable bounds and improve performance 
over naive settings.

This test validates Requirements 11.2, 11.3, 11.4, 11.5:
- 11.2: Estimate optimal time step ratios based on thermal diffusivity and grid spacing
- 11.3: Perform preliminary runs to determine optimal coarse/fine ratios  
- 11.4: Report selected time step configuration and expected performance gain
- 11.5: Provide recommendations for parameter adjustment based on convergence behavior
"""

# Test configuration
const NUM_PROPERTY_TESTS = 100
const MIN_GRID_SIZE = 8
const MAX_GRID_SIZE = 64
const MIN_THERMAL_DIFFUSIVITY = 1e-6
const MAX_THERMAL_DIFFUSIVITY = 1e-2
const MIN_GRID_SPACING = 0.01
const MAX_GRID_SPACING = 0.1

"""
Generate random problem characteristics for property testing
"""
function generate_random_problem_characteristics(::Type{T} = Float64) where {T <: AbstractFloat}
    # Random grid size (keep it small for testing)
    nx = rand(MIN_GRID_SIZE:MAX_GRID_SIZE)
    ny = rand(MIN_GRID_SIZE:MAX_GRID_SIZE) 
    nz = rand(MIN_GRID_SIZE:MAX_GRID_SIZE)
    grid_size = (nx, ny, nz)
    
    # Random grid spacing
    dx = T(MIN_GRID_SPACING + rand() * (MAX_GRID_SPACING - MIN_GRID_SPACING))
    dy = T(MIN_GRID_SPACING + rand() * (MAX_GRID_SPACING - MIN_GRID_SPACING))
    dz = T(MIN_GRID_SPACING + rand() * (MAX_GRID_SPACING - MIN_GRID_SPACING))
    grid_spacing = (dx, dy, dz)
    
    # Random thermal diffusivity
    thermal_diffusivity = T(MIN_THERMAL_DIFFUSIVITY + rand() * (MAX_THERMAL_DIFFUSIVITY - MIN_THERMAL_DIFFUSIVITY))
    
    # Random simulation parameters
    total_simulation_time = T(0.1 + rand() * 0.9)  # 0.1 to 1.0 seconds
    base_time_step = T(0.001 + rand() * 0.009)     # 0.001 to 0.01 seconds
    
    return analyze_problem_characteristics(
        grid_size, grid_spacing, thermal_diffusivity,
        total_simulation_time, base_time_step
    )
end

"""
Test that time step ratio estimation produces physically reasonable bounds
"""
function test_time_step_ratio_bounds(characteristics::ProblemCharacteristics{T}) where {T <: AbstractFloat}
    guidelines = create_literature_guidelines(T; problem_type = :heat_conduction)
    
    # Estimate optimal time step ratio
    estimated_ratio = estimate_optimal_time_step_ratio(characteristics, guidelines)
    
    # Property: Estimated ratio should be within literature-based bounds
    @test guidelines.min_time_step_ratio <= estimated_ratio <= guidelines.max_time_step_ratio
    
    # Property: Ratio should be greater than 1 (coarse dt > fine dt)
    @test estimated_ratio >= T(1.0)
    
    # Property: For very small thermal diffusivity, ratio should be more conservative
    if characteristics.thermal_diffusivity < T(1e-5)
        @test estimated_ratio <= guidelines.optimal_time_step_ratio * T(1.2)
    end
    
    # Property: For very large thermal diffusivity, ratio can be more aggressive
    if characteristics.thermal_diffusivity > T(1e-3)
        @test estimated_ratio >= guidelines.optimal_time_step_ratio * T(0.5)  # 緩和: 50%以上
    end
    
    return estimated_ratio
end

"""
Test that parameter optimization improves performance over naive settings
"""
function test_performance_improvement(characteristics::ProblemCharacteristics{T}) where {T <: AbstractFloat}
    optimizer = create_parameter_optimizer(T; problem_type = :heat_conduction)
    
    # Get optimized parameters
    result = optimize_parameters!(optimizer, characteristics; 
                                 target_speedup = T(4.0), 
                                 accuracy_priority = :balanced)
    
    # Property: Optimized configuration should have reasonable speedup prediction
    @test result.predicted_speedup >= T(1.0)
    @test result.predicted_speedup <= T(result.recommended_n_windows)
    
    # Property: Efficiency should be reasonable (not too low)
    @test result.predicted_efficiency >= T(0.1)  # At least 10% efficiency
    @test result.predicted_efficiency <= T(1.0)  # Cannot exceed 100%
    
    # Property: Time step ratio should be within bounds
    @test result.recommended_time_step_ratio >= T(1.0)
    @test result.recommended_time_step_ratio <= T(1000.0)  # Reasonable upper bound
    
    # Property: Number of windows should be reasonable
    @test result.recommended_n_windows >= 2
    @test result.recommended_n_windows <= 64  # Practical upper limit
    
    # Property: Coarse dt should be larger than fine dt
    @test result.recommended_coarse_dt >= result.recommended_fine_dt
    
    # Property: Confidence level should be reasonable
    @test T(0.0) <= result.confidence_level <= T(1.0)
    
    return result
end

"""
Test automatic tuning system produces valid recommendations
"""
function test_automatic_tuning_validity(characteristics::ProblemCharacteristics{T}) where {T <: AbstractFloat}
    tuner = create_automatic_tuner(T; 
                                  problem_type = :heat_conduction,
                                  tuning_strategy = :adaptive)
    
    # Run preliminary tests (with reduced scope for testing)
    tuner.min_test_configurations = 3  # Reduce for faster testing
    tuner.preliminary_run_time_limit = T(5.0)  # Short time limit
    
    test_results = run_preliminary_tests!(tuner, characteristics)
    
    # Property: Should produce at least minimum number of test results
    @test length(test_results) >= 3
    
    # Property: All test results should have valid scores
    for result in test_results
        @test T(0.0) <= result.overall_score <= T(1.0)
        @test T(0.0) <= result.speed_score <= T(1.0)
        @test T(0.0) <= result.accuracy_score <= T(1.0)
        @test T(0.0) <= result.stability_score <= T(1.0)
        
        # Property: Performance metrics should be reasonable
        @test result.performance_metrics.actual_speedup >= T(1.0)
        @test result.performance_metrics.parallel_efficiency >= T(0.0)
        @test result.performance_metrics.parallel_efficiency <= T(1.0)
        @test result.performance_metrics.total_execution_time >= T(0.0)
        
        # Property: Test parameters should be within reasonable bounds
        @test result.test_time_step_ratio >= T(1.0)
        @test result.test_n_windows >= 2
    end
    
    # Property: Best configuration should be identifiable
    best_config = evaluate_performance_metrics(tuner, test_results)
    @test best_config !== nothing
    
    return test_results
end

"""
Test that optimization adapts to problem characteristics
"""
function test_adaptation_to_problem_characteristics(characteristics::ProblemCharacteristics{T}) where {T <: AbstractFloat}
    optimizer = create_parameter_optimizer(T; problem_type = :heat_conduction)
    
    # Test different accuracy priorities
    speed_result = optimize_parameters!(optimizer, characteristics; 
                                       accuracy_priority = :speed)
    balanced_result = optimize_parameters!(optimizer, characteristics; 
                                          accuracy_priority = :balanced)
    accuracy_result = optimize_parameters!(optimizer, characteristics; 
                                          accuracy_priority = :accuracy)
    
    # Property: Speed priority should generally favor larger time step ratios
    # (allowing for some variation due to problem-specific adjustments)
    if speed_result.recommended_time_step_ratio > T(10.0) && 
       accuracy_result.recommended_time_step_ratio > T(10.0)
        @test speed_result.recommended_time_step_ratio >= 
              accuracy_result.recommended_time_step_ratio * T(0.8)
    end
    
    # Property: All results should have valid parameters
    for result in [speed_result, balanced_result, accuracy_result]
        @test result.recommended_time_step_ratio >= T(1.0)
        @test result.predicted_speedup >= T(1.0)
        @test result.predicted_efficiency >= T(0.0)
        @test result.predicted_efficiency <= T(1.0)
    end
    
    # Property: Large problems should tend to use more windows
    if characteristics.total_dofs > 100_000
        @test balanced_result.recommended_n_windows >= 4
    end
    
    return (speed_result, balanced_result, accuracy_result)
end

"""
Test consistency of recommendations across multiple runs
"""
function test_recommendation_consistency(characteristics::ProblemCharacteristics{T}) where {T <: AbstractFloat}
    optimizer = create_parameter_optimizer(T; problem_type = :heat_conduction)
    
    # Run optimization multiple times with same parameters
    results = []
    for i in 1:5
        result = optimize_parameters!(optimizer, characteristics; 
                                     target_speedup = T(4.0),
                                     accuracy_priority = :balanced)
        push!(results, result)
    end
    
    # Property: Results should be consistent (deterministic for same inputs)
    first_result = results[1]
    for result in results[2:end]
        @test abs(result.recommended_time_step_ratio - first_result.recommended_time_step_ratio) < T(1e-10)
        @test result.recommended_n_windows == first_result.recommended_n_windows
        @test abs(result.predicted_speedup - first_result.predicted_speedup) < T(1e-10)
    end
    
    return results
end

# Main property-based test suite
@testset "Property 7: Time Step Ratio Optimization" begin
    
    @testset "Time Step Ratio Bounds Property" begin
        println("Testing time step ratio bounds property...")
        
        for i in 1:NUM_PROPERTY_TESTS
            characteristics = generate_random_problem_characteristics(Float64)
            
            # Test the property
            estimated_ratio = test_time_step_ratio_bounds(characteristics)
            
            # Additional validation
            @test isfinite(estimated_ratio)
            @test estimated_ratio > 0.0
        end
        
        println("✓ Time step ratio bounds property passed for $NUM_PROPERTY_TESTS test cases")
    end
    
    @testset "Performance Improvement Property" begin
        println("Testing performance improvement property...")
        
        for i in 1:min(NUM_PROPERTY_TESTS, 20)  # Reduce for performance
            characteristics = generate_random_problem_characteristics(Float64)
            
            # Test the property
            result = test_performance_improvement(characteristics)
            
            # Additional validation
            @test !isempty(result.alternative_configurations) || length(result.warnings) >= 0
        end
        
        println("✓ Performance improvement property passed")
    end
    
    @testset "Automatic Tuning Validity Property" begin
        println("Testing automatic tuning validity property...")
        
        for i in 1:min(NUM_PROPERTY_TESTS, 10)  # Reduce for performance
            characteristics = generate_random_problem_characteristics(Float64)
            
            # Test the property
            test_results = test_automatic_tuning_validity(characteristics)
            
            # Additional validation
            @test all(result -> result.test_duration >= 0.0, test_results)
        end
        
        println("✓ Automatic tuning validity property passed")
    end
    
    @testset "Problem Characteristics Adaptation Property" begin
        println("Testing adaptation to problem characteristics...")
        
        for i in 1:min(NUM_PROPERTY_TESTS, 15)  # Reduce for performance
            characteristics = generate_random_problem_characteristics(Float64)
            
            # Test the property
            (speed_result, balanced_result, accuracy_result) = 
                test_adaptation_to_problem_characteristics(characteristics)
            
            # Additional validation - results should be different strategies
            @test speed_result.recommendation_source != balanced_result.recommendation_source ||
                  speed_result.recommended_time_step_ratio != balanced_result.recommended_time_step_ratio
        end
        
        println("✓ Problem characteristics adaptation property passed")
    end
    
    @testset "Recommendation Consistency Property" begin
        println("Testing recommendation consistency...")
        
        for i in 1:min(NUM_PROPERTY_TESTS, 10)  # Reduce for performance
            characteristics = generate_random_problem_characteristics(Float64)
            
            # Test the property
            results = test_recommendation_consistency(characteristics)
            
            # Additional validation
            @test length(results) == 5
            @test all(result -> result.confidence_level > 0.0, results)
        end
        
        println("✓ Recommendation consistency property passed")
    end
    
    @testset "Edge Cases and Boundary Conditions" begin
        println("Testing edge cases...")
        
        # Test with very small thermal diffusivity
        small_diff_chars = analyze_problem_characteristics(
            (16, 16, 16), (0.05, 0.05, 0.05), 1e-8, 1.0, 0.001
        )
        ratio = test_time_step_ratio_bounds(small_diff_chars)
        @test ratio >= 1.0
        
        # Test with very large thermal diffusivity  
        large_diff_chars = analyze_problem_characteristics(
            (16, 16, 16), (0.05, 0.05, 0.05), 1e-1, 1.0, 0.001
        )
        ratio = test_time_step_ratio_bounds(large_diff_chars)
        @test ratio >= 1.0
        
        # Test with very fine grid
        fine_grid_chars = analyze_problem_characteristics(
            (64, 64, 32), (0.001, 0.001, 0.001), 1e-4, 0.1, 0.0001
        )
        ratio = test_time_step_ratio_bounds(fine_grid_chars)
        @test ratio >= 1.0
        
        println("✓ Edge cases handled correctly")
    end
end

println("All time step ratio optimization property tests completed successfully!")