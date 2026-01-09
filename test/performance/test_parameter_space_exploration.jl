# Property-based test for parameter space exploration
# **Feature: parareal-time-parallelization, Property 8: Parameter Space Exploration Completeness**
# **Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5**

using Test
using Random
using Statistics

# Import the parareal module
include("../src/parareal.jl")
using .Parareal

"""
Property-based test for parameter space exploration

Property 8: Parameter Space Exploration Completeness
For any parameter exploration configuration, the system should systematically test 
the specified parameter combinations and generate comprehensive performance metrics.

This test validates Requirements 12.1, 12.2, 12.3, 12.4, 12.5:
- 12.1: Automatically test multiple coarse/fine time step ratio combinations
- 12.2: Evaluate parareal efficiency metrics including speedup and convergence rate
- 12.3: Test different numbers of time windows and MPI process configurations
- 12.4: Generate performance maps showing optimal parameter regions
- 12.5: Save configuration files for future use with similar problems
"""

# Test configuration
const NUM_PROPERTY_TESTS = 5  # Further reduced for performance
const MIN_EXPLORATION_POINTS = 5
const MAX_EXPLORATION_POINTS = 25

"""
Generate random problem characteristics for exploration testing
"""
function generate_random_exploration_problem(::Type{T} = Float64) where {T <: AbstractFloat}
    # Random grid size (moderate size for testing)
    nx = rand(16:32)
    ny = rand(16:32) 
    nz = rand(16:32)
    grid_size = (nx, ny, nz)
    
    # Random grid spacing
    dx = T(0.02 + rand() * 0.08)  # 0.02 to 0.1
    dy = T(0.02 + rand() * 0.08)
    dz = T(0.02 + rand() * 0.08)
    grid_spacing = (dx, dy, dz)
    
    # Random thermal diffusivity
    thermal_diffusivity = T(1e-5 + rand() * 1e-3)  # 1e-5 to 1e-3
    
    # Random simulation parameters
    total_simulation_time = T(0.5 + rand() * 1.5)  # 0.5 to 2.0 seconds
    base_time_step = T(0.005 + rand() * 0.015)     # 0.005 to 0.02 seconds
    
    return analyze_problem_characteristics(
        grid_size, grid_spacing, thermal_diffusivity,
        total_simulation_time, base_time_step
    )
end

"""
Test that parameter space exploration generates the expected number of points
"""
function test_exploration_point_generation(
    explorer::ParameterSpaceExplorer{T},
    characteristics::ProblemCharacteristics{T}
) where {T <: AbstractFloat}
    
    # Property: Explorer should generate the requested number of exploration points
    max_points = explorer.max_exploration_points
    
    # Generate exploration points
    points = Parareal.ParameterOptimization.generate_exploration_points(explorer, characteristics)
    
    # Property: Number of points should not exceed maximum
    @test length(points) <= max_points
    
    # Property: Should generate at least minimum number of points
    @test length(points) >= min(MIN_EXPLORATION_POINTS, max_points)
    
    # Property: All points should have required parameters
    for point in points
        @test haskey(point, :time_step_ratio)
        @test haskey(point, :n_windows)
        @test haskey(point, :convergence_tolerance)
        @test haskey(point, :max_iterations)
        
        # Property: Parameters should be within valid ranges
        @test point[:time_step_ratio] >= T(1.0)
        @test point[:time_step_ratio] <= T(1000.0)  # Reasonable upper bound
        @test point[:n_windows] >= 2
        @test point[:n_windows] <= 64  # Reasonable upper bound
        @test point[:convergence_tolerance] > T(0.0)
        @test point[:max_iterations] > 0
    end
    
    return points
end

"""
Test that exploration results contain comprehensive performance metrics
"""
function test_exploration_performance_metrics(
    explorer::ParameterSpaceExplorer{T},
    characteristics::ProblemCharacteristics{T}
) where {T <: AbstractFloat}
    
    # Run a small exploration
    explorer.max_exploration_points = 5  # Small for testing
    explorer.exploration_timeout = T(30.0)  # Short timeout
    
    results = explore_parameter_space!(explorer, characteristics; save_results = false)
    
    # Property: Should produce results
    @test !isempty(results)
    @test length(results) <= explorer.max_exploration_points
    
    # Property: All results should have comprehensive metrics
    for result in results
        @test result.performance_metrics.actual_speedup >= T(1.0)
        @test T(0.0) <= result.performance_metrics.parallel_efficiency <= T(1.0)
        @test result.performance_metrics.total_execution_time >= T(0.0)
        @test result.performance_metrics.iterations_to_convergence >= 0
        @test T(0.0) <= result.overall_score <= T(1.0)
        
        # Property: Timestamps should be reasonable
        @test result.timestamp > 0.0
        @test !isempty(result.exploration_id)
        
        # Property: Problem characteristics should match
        @test result.problem_characteristics.grid_size == characteristics.grid_size
        @test result.problem_characteristics.thermal_diffusivity == characteristics.thermal_diffusivity
    end
    
    return results
end

"""
Test that different exploration strategies produce valid results
"""
function test_exploration_strategies(characteristics::ProblemCharacteristics{T}) where {T <: AbstractFloat}
    
    strategies = [:grid, :random, :adaptive]
    
    for strategy in strategies
        explorer = create_parameter_space_explorer(T; exploration_strategy = strategy)
        explorer.max_exploration_points = 8  # Small for testing
        explorer.exploration_timeout = T(20.0)
        
        results = explore_parameter_space!(explorer, characteristics; save_results = false)
        
        # Property: Each strategy should produce valid results
        @test !isempty(results)
        @test all(r -> r.overall_score >= T(0.0), results)
        @test all(r -> r.performance_metrics.actual_speedup >= T(1.0), results)
        
        # Property: Results should have diversity (not all identical)
        if length(results) > 1
            ratios = [r.time_step_ratio for r in results]
            windows = [r.n_windows for r in results]
            
            # At least some diversity in parameters
            @test length(unique(ratios)) > 1 || length(unique(windows)) > 1
        end
    end
end

"""
Test that performance maps are generated correctly
"""
function test_performance_map_generation(results::Vector{ExplorationResult{T}}) where {T <: AbstractFloat}
    
    if isempty(results)
        return  # Skip if no results
    end
    
    characteristics = results[1].problem_characteristics
    performance_map = generate_performance_map(results, characteristics)
    
    # Property: Performance map should have valid structure
    @test !isempty(performance_map.time_step_ratios)
    @test !isempty(performance_map.n_windows_range)
    @test size(performance_map.speedup_map, 1) == length(performance_map.time_step_ratios)
    @test size(performance_map.speedup_map, 2) == length(performance_map.n_windows_range)
    @test size(performance_map.efficiency_map) == size(performance_map.speedup_map)
    @test size(performance_map.convergence_map) == size(performance_map.speedup_map)
    
    # Property: Maps should contain valid values
    @test all(speedup -> speedup >= T(0.0), performance_map.speedup_map)
    @test all(eff -> T(0.0) <= eff <= T(1.0), performance_map.efficiency_map)
    
    # Property: Metadata should be present
    @test !isempty(performance_map.problem_id)
    @test performance_map.generation_timestamp > 0.0
    
    return performance_map
end

"""
Test that optimal configurations can be found
"""
function test_optimal_configuration_search(results::Vector{ExplorationResult{T}}) where {T <: AbstractFloat}
    
    if length(results) < 2
        return  # Skip if insufficient results
    end
    
    # Test different search criteria
    criteria_list = [:overall_score, :speedup, :efficiency]
    
    for criteria in criteria_list
        optimal_configs = find_optimal_configurations(results; criteria = criteria, top_n = 3)
        
        # Property: Should return requested number of configurations (or all if fewer)
        expected_count = min(3, length(results))
        @test length(optimal_configs) == expected_count
        
        # Property: Results should be sorted by the specified criteria
        if criteria == :overall_score
            scores = [config.overall_score for config in optimal_configs]
            @test issorted(scores, rev = true)
        elseif criteria == :speedup
            speedups = [config.performance_metrics.actual_speedup for config in optimal_configs]
            @test issorted(speedups, rev = true)
        elseif criteria == :efficiency
            efficiencies = [config.performance_metrics.parallel_efficiency for config in optimal_configs]
            @test issorted(efficiencies, rev = true)
        end
        
        # Property: All returned configurations should be from the original results
        for config in optimal_configs
            @test config in results
        end
    end
end

"""
Test that exploration handles edge cases properly
"""
function test_exploration_edge_cases(::Type{T} = Float64) where {T <: AbstractFloat}
    
    # Test with minimal problem
    minimal_characteristics = analyze_problem_characteristics(
        (8, 8, 8), (T(0.1), T(0.1), T(0.1)), T(1e-4), T(0.1), T(0.01)
    )
    
    explorer = create_parameter_space_explorer(T; exploration_strategy = :grid)
    explorer.max_exploration_points = 4  # Minimal
    explorer.exploration_timeout = T(15.0)
    
    results = explore_parameter_space!(explorer, minimal_characteristics; save_results = false)
    
    # Property: Should handle minimal cases gracefully
    @test !isempty(results)
    @test all(r -> isfinite(r.overall_score), results)
    @test all(r -> isfinite(r.performance_metrics.actual_speedup), results)
    
    # Test with very restrictive ranges
    restrictive_explorer = create_parameter_space_explorer(T)
    restrictive_explorer.time_step_ratio_range = (T(10.0), T(12.0))  # Very narrow range
    restrictive_explorer.n_windows_range = (4, 6)  # Very narrow range
    restrictive_explorer.max_exploration_points = 3
    
    restrictive_results = explore_parameter_space!(restrictive_explorer, minimal_characteristics; save_results = false)
    
    # Property: Should handle restrictive ranges
    @test !isempty(restrictive_results)
    for result in restrictive_results
        @test T(10.0) <= result.time_step_ratio <= T(12.0)
        @test 4 <= result.n_windows <= 6
    end
end

"""
Test exploration consistency across multiple runs
"""
function test_exploration_consistency(characteristics::ProblemCharacteristics{T}) where {T <: AbstractFloat}
    
    # Run the same exploration multiple times
    explorer = create_parameter_space_explorer(T; exploration_strategy = :grid)
    explorer.max_exploration_points = 6
    explorer.exploration_timeout = T(20.0)
    
    results1 = explore_parameter_space!(explorer, characteristics; save_results = false)
    results2 = explore_parameter_space!(explorer, characteristics; save_results = false)
    
    # Property: Grid exploration should be deterministic
    @test length(results1) == length(results2)
    
    # Property: Same parameters should produce similar results (within tolerance)
    for (r1, r2) in zip(results1, results2)
        if r1.time_step_ratio == r2.time_step_ratio && r1.n_windows == r2.n_windows
            # Performance metrics should be similar (allowing for small variations)
            speedup_diff = abs(r1.performance_metrics.actual_speedup - r2.performance_metrics.actual_speedup)
            @test speedup_diff < T(0.1)  # Small tolerance for simulation variations
        end
    end
end

# Main property-based test suite
@testset "Property 8: Parameter Space Exploration Completeness" begin
    
    @testset "Exploration Point Generation Property" begin
        println("Testing exploration point generation property...")
        
        for i in 1:NUM_PROPERTY_TESTS
            characteristics = generate_random_exploration_problem(Float64)
            
            # Test different strategies
            for strategy in [:grid, :random, :adaptive]
                explorer = create_parameter_space_explorer(Float64; exploration_strategy = strategy)
                explorer.max_exploration_points = rand(MIN_EXPLORATION_POINTS:MAX_EXPLORATION_POINTS)
                
                # Test the property
                points = test_exploration_point_generation(explorer, characteristics)
                
                # Additional validation
                @test all(point -> isa(point[:time_step_ratio], Real), points)
                @test all(point -> isa(point[:n_windows], Integer), points)
            end
        end
        
        println("✓ Exploration point generation property passed for $NUM_PROPERTY_TESTS test cases")
    end
    
    @testset "Performance Metrics Completeness Property" begin
        println("Testing performance metrics completeness property...")
        
        for i in 1:min(NUM_PROPERTY_TESTS, 10)  # Reduce for performance
            characteristics = generate_random_exploration_problem(Float64)
            explorer = create_parameter_space_explorer(Float64; exploration_strategy = :random)
            
            # Test the property
            results = test_exploration_performance_metrics(explorer, characteristics)
            
            # Additional validation
            @test all(r -> r.timestamp > 0, results)
            @test all(r -> !isempty(r.exploration_id), results)
        end
        
        println("✓ Performance metrics completeness property passed")
    end
    
    @testset "Exploration Strategy Validity Property" begin
        println("Testing exploration strategy validity property...")
        
        for i in 1:min(NUM_PROPERTY_TESTS, 8)  # Reduce for performance
            characteristics = generate_random_exploration_problem(Float64)
            
            # Test the property
            test_exploration_strategies(characteristics)
        end
        
        println("✓ Exploration strategy validity property passed")
    end
    
    @testset "Performance Map Generation Property" begin
        println("Testing performance map generation property...")
        
        for i in 1:min(NUM_PROPERTY_TESTS, 8)  # Reduce for performance
            characteristics = generate_random_exploration_problem(Float64)
            explorer = create_parameter_space_explorer(Float64; exploration_strategy = :grid)
            explorer.max_exploration_points = 6
            
            results = explore_parameter_space!(explorer, characteristics; save_results = false)
            
            # Test the property
            performance_map = test_performance_map_generation(results)
            
            # Additional validation
            if performance_map !== nothing
                @test length(performance_map.time_step_ratios) > 0
                @test length(performance_map.n_windows_range) > 0
            end
        end
        
        println("✓ Performance map generation property passed")
    end
    
    @testset "Optimal Configuration Search Property" begin
        println("Testing optimal configuration search property...")
        
        for i in 1:min(NUM_PROPERTY_TESTS, 8)  # Reduce for performance
            characteristics = generate_random_exploration_problem(Float64)
            explorer = create_parameter_space_explorer(Float64; exploration_strategy = :random)
            explorer.max_exploration_points = 8
            
            results = explore_parameter_space!(explorer, characteristics; save_results = false)
            
            # Test the property
            test_optimal_configuration_search(results)
        end
        
        println("✓ Optimal configuration search property passed")
    end
    
    @testset "Edge Cases and Boundary Conditions" begin
        println("Testing edge cases...")
        
        # Test edge cases
        test_exploration_edge_cases(Float64)
        
        println("✓ Edge cases handled correctly")
    end
    
    @testset "Exploration Consistency Property" begin
        println("Testing exploration consistency...")
        
        for i in 1:min(NUM_PROPERTY_TESTS, 5)  # Reduce for performance
            characteristics = generate_random_exploration_problem(Float64)
            
            # Test the property
            test_exploration_consistency(characteristics)
        end
        
        println("✓ Exploration consistency property passed")
    end
end

println("All parameter space exploration property tests completed successfully!")