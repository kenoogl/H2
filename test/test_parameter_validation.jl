# Simple parameter validation test
# **Feature: parareal-time-parallelization, Property 4: Parameter Validation Completeness**
# **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

using Test

# Import the parareal module
include("../src/parareal.jl")
using .Parareal

"""
Property-based test for parameter validation

Property 4: Parameter Validation Completeness
For any valid parareal parameter (time steps, window count, iterations, tolerance), 
the system should accept and correctly store the configuration.

This test validates Requirements 2.1, 2.2, 2.3, 2.4, 2.5:
- 2.1: Accept coarse time step size specification
- 2.2: Accept fine time step size specification  
- 2.3: Accept number of time windows specification
- 2.4: Accept maximum parareal iterations specification
- 2.5: Accept parareal convergence tolerance specification
"""

# Test configuration
const NUM_PROPERTY_TESTS = 100

"""
Generate random valid parameters for testing
"""
function generate_random_valid_parameters(::Type{T} = Float64) where {T <: AbstractFloat}
    # Generate valid time steps (coarse > fine > 0)
    fine_dt = T(0.001 + rand() * 0.01)  # 0.001 to 0.011
    coarse_dt = fine_dt * (2.0 + rand() * 98.0)  # 2x to 100x fine_dt
    
    # Generate other valid parameters
    n_time_windows = rand(2:32)
    total_time = T(0.1 + rand() * 2.0)  # 0.1 to 2.1
    max_iterations = rand(1:20)
    convergence_tolerance = T(10.0^(-rand(4:8)))  # 1e-4 to 1e-8
    n_mpi_processes = rand(1:16)
    n_threads_per_process = rand(1:8)
    
    return (
        coarse_dt = coarse_dt,
        fine_dt = fine_dt,
        n_time_windows = n_time_windows,
        total_time = total_time,
        max_iterations = max_iterations,
        convergence_tolerance = convergence_tolerance,
        n_mpi_processes = n_mpi_processes,
        n_threads_per_process = n_threads_per_process
    )
end

"""
Test that valid parameters are accepted and stored correctly
"""
function test_parameter_acceptance_and_storage(params_tuple)
    # Create parameters
    params = Parareal.ParameterOptimization.create_parareal_parameters(
        Float64;
        coarse_dt = params_tuple.coarse_dt,
        fine_dt = params_tuple.fine_dt,
        n_time_windows = params_tuple.n_time_windows,
        total_time = params_tuple.total_time,
        max_iterations = params_tuple.max_iterations,
        convergence_tolerance = params_tuple.convergence_tolerance,
        n_mpi_processes = params_tuple.n_mpi_processes,
        n_threads_per_process = params_tuple.n_threads_per_process
    )
    
    # Property: Parameters should be stored correctly
    @test params.coarse_dt == params_tuple.coarse_dt
    @test params.fine_dt == params_tuple.fine_dt
    @test params.n_time_windows == params_tuple.n_time_windows
    @test params.total_time == params_tuple.total_time
    @test params.max_iterations == params_tuple.max_iterations
    @test params.convergence_tolerance == params_tuple.convergence_tolerance
    @test params.n_mpi_processes == params_tuple.n_mpi_processes
    @test params.n_threads_per_process == params_tuple.n_threads_per_process
    
    # Property: Time step ratio should be calculated correctly
    expected_ratio = params_tuple.coarse_dt / params_tuple.fine_dt
    @test abs(params.time_step_ratio - expected_ratio) < 1e-10
    
    return params
end

"""
Test parameter validation functionality
"""
function test_parameter_validation(params)
    # Validate parameters
    validation_result = Parareal.ParameterOptimization.validate_parameters(params)
    
    # Property: Validation should complete without error
    @test isa(validation_result, Parareal.ParameterOptimization.ValidationResult)
    
    # Property: Validation result should have expected fields
    @test isa(validation_result.is_valid, Bool)
    @test isa(validation_result.warnings, Vector{String})
    @test isa(validation_result.recommendations, Vector{String})
    @test validation_result.time_steps_per_window >= 0
    @test validation_result.total_coarse_steps >= 0
    @test validation_result.total_fine_steps >= 0
    @test validation_result.estimated_memory_gb >= 0.0
    
    return validation_result
end

"""
Test invalid parameter rejection
"""
function test_invalid_parameter_rejection()
    # Test coarse_dt <= fine_dt (should fail)
    @test_throws Exception Parareal.ParameterOptimization.create_parareal_parameters(
        Float64; coarse_dt = 0.01, fine_dt = 0.02
    )
    
    # Test negative time steps (should fail)
    @test_throws Exception Parareal.ParameterOptimization.create_parareal_parameters(
        Float64; coarse_dt = -0.01, fine_dt = 0.001
    )
    
    @test_throws Exception Parareal.ParameterOptimization.create_parareal_parameters(
        Float64; coarse_dt = 0.01, fine_dt = -0.001
    )
    
    # Test invalid time windows (should fail)
    @test_throws Exception Parareal.ParameterOptimization.create_parareal_parameters(
        Float64; coarse_dt = 0.01, fine_dt = 0.001, n_time_windows = 1
    )
    
    # Test invalid total time (should fail)
    @test_throws Exception Parareal.ParameterOptimization.create_parareal_parameters(
        Float64; coarse_dt = 0.01, fine_dt = 0.001, total_time = -1.0
    )
    
    # Test invalid iterations (should fail)
    @test_throws Exception Parareal.ParameterOptimization.create_parareal_parameters(
        Float64; coarse_dt = 0.01, fine_dt = 0.001, max_iterations = 0
    )
    
    # Test invalid tolerance (should fail)
    @test_throws Exception Parareal.ParameterOptimization.create_parareal_parameters(
        Float64; coarse_dt = 0.01, fine_dt = 0.001, convergence_tolerance = -1e-6
    )
    
    # Test invalid MPI processes (should fail)
    @test_throws Exception Parareal.ParameterOptimization.create_parareal_parameters(
        Float64; coarse_dt = 0.01, fine_dt = 0.001, n_mpi_processes = 0
    )
    
    # Test invalid threads per process (should fail)
    @test_throws Exception Parareal.ParameterOptimization.create_parareal_parameters(
        Float64; coarse_dt = 0.01, fine_dt = 0.001, n_threads_per_process = 0
    )
end

"""
Test parameter summary printing
"""
function test_parameter_summary_printing(params)
    # Property: Summary printing should not throw errors
    @test_nowarn Parareal.ParameterOptimization.print_parameter_summary(params)
end

# Main property-based test suite
@testset "Property 4: Parameter Validation Completeness" begin
    
    @testset "Valid Parameter Acceptance and Storage" begin
        println("Testing valid parameter acceptance and storage...")
        
        for i in 1:NUM_PROPERTY_TESTS
            params_tuple = generate_random_valid_parameters(Float64)
            
            # Test the property
            params = test_parameter_acceptance_and_storage(params_tuple)
            
            # Additional validation
            @test params.time_step_ratio > 1.0  # Should always be > 1
            @test params.coarse_dt > params.fine_dt  # Coarse should be larger
        end
        
        println("✓ Valid parameter acceptance property passed for $NUM_PROPERTY_TESTS test cases")
    end
    
    @testset "Parameter Validation Functionality" begin
        println("Testing parameter validation functionality...")
        
        for i in 1:min(NUM_PROPERTY_TESTS, 50)  # Reduce for performance
            params_tuple = generate_random_valid_parameters(Float64)
            params = Parareal.ParameterOptimization.create_parareal_parameters(Float64; params_tuple...)
            
            # Test the property
            validation_result = test_parameter_validation(params)
            
            # Additional checks
            @test validation_result.time_steps_per_window > 0
            @test validation_result.total_coarse_steps > 0
            @test validation_result.total_fine_steps > 0
        end
        
        println("✓ Parameter validation functionality property passed")
    end
    
    @testset "Invalid Parameter Rejection" begin
        println("Testing invalid parameter rejection...")
        
        # Test the property
        test_invalid_parameter_rejection()
        
        println("✓ Invalid parameter rejection property passed")
    end
    
    @testset "Parameter Summary Printing" begin
        println("Testing parameter summary printing...")
        
        for i in 1:min(NUM_PROPERTY_TESTS, 10)  # Small sample for printing tests
            params_tuple = generate_random_valid_parameters(Float64)
            params = Parareal.ParameterOptimization.create_parareal_parameters(Float64; params_tuple...)
            
            # Test the property
            test_parameter_summary_printing(params)
        end
        
        println("✓ Parameter summary printing property passed")
    end
    
    @testset "Edge Cases and Boundary Conditions" begin
        println("Testing edge cases...")
        
        # Test minimum valid values
        min_params = Parareal.ParameterOptimization.create_parareal_parameters(
            Float64;
            coarse_dt = 0.002,  # Just above fine_dt
            fine_dt = 0.001,
            n_time_windows = 2,  # Minimum
            total_time = 0.001,  # Very small
            max_iterations = 1,  # Minimum
            convergence_tolerance = 1e-12,  # Very strict
            n_mpi_processes = 1,  # Minimum
            n_threads_per_process = 1  # Minimum
        )
        
        @test min_params.time_step_ratio ≈ 2.0
        @test min_params.n_time_windows == 2
        
        # Test large valid values
        max_params = Parareal.ParameterOptimization.create_parareal_parameters(
            Float64;
            coarse_dt = 1.0,
            fine_dt = 0.001,  # Large ratio
            n_time_windows = 100,  # Large number
            total_time = 100.0,  # Large time
            max_iterations = 100,  # Large iterations
            convergence_tolerance = 1e-3,  # Loose tolerance
            n_mpi_processes = 64,  # Large number
            n_threads_per_process = 16  # Large number
        )
        
        @test max_params.time_step_ratio ≈ 1000.0
        @test max_params.n_time_windows == 100
        
        println("✓ Edge cases handled correctly")
    end
end

println("All parameter validation property tests completed successfully!")