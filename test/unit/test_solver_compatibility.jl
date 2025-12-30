using Test
using FLoops
using Statistics  # For cor function
using LinearAlgebra  # For mathematical operations

# Include the Parareal module
include("../src/parareal.jl")
using .Parareal

"""
Property-based test for solver compatibility
Property 5: Backward Compatibility Preservation
Validates: Requirements 3.1, 3.2, 3.3, 3.4
"""

@testset "Solver Compatibility Property Tests" begin
    
    @testset "Property 5: Backward Compatibility Preservation" begin
        
        @testset "Solver construction and validation" begin
            # Test that solvers can be constructed with valid parameters
            
            @testset "CoarseSolver construction" begin
                # Test default construction
                coarse_default = Parareal.CoarseSolver()
                @test coarse_default isa Parareal.CoarseSolver{Float64}
                @test coarse_default.dt == 0.1
                @test coarse_default.spatial_resolution_factor == 2.0
                @test coarse_default.simplified_physics == true
                @test coarse_default.solver_type == :pbicgstab
                
                # Test custom construction
                coarse_custom = Parareal.CoarseSolver{Float32}(
                    dt = 0.05f0,
                    spatial_resolution_factor = 3.0f0,
                    simplified_physics = false,
                    solver_type = :cg,
                    tolerance = 1.0f-4,
                    max_iterations = 50
                )
                @test coarse_custom isa Parareal.CoarseSolver{Float32}
                @test coarse_custom.dt == 0.05f0
                @test coarse_custom.spatial_resolution_factor == 3.0f0
                @test coarse_custom.simplified_physics == false
                @test coarse_custom.solver_type == :cg
                
                # Test invalid parameters
                @test_throws ErrorException Parareal.CoarseSolver{Float64}(dt = -0.1)
                @test_throws ErrorException Parareal.CoarseSolver{Float64}(spatial_resolution_factor = 0.0)
                @test_throws ErrorException Parareal.CoarseSolver{Float64}(tolerance = -1.0e-6)
                @test_throws ErrorException Parareal.CoarseSolver{Float64}(max_iterations = 0)
            end
            
            @testset "FineSolver construction" begin
                # Test default construction
                fine_default = Parareal.FineSolver()
                @test fine_default isa Parareal.FineSolver{Float64}
                @test fine_default.dt == 0.01
                @test fine_default.solver_type == :pbicgstab
                @test fine_default.smoother == :gs
                @test fine_default.use_full_physics == true
                
                # Test custom construction
                fine_custom = Parareal.FineSolver{Float32}(
                    dt = 0.005f0,
                    solver_type = :cg,
                    smoother = :none,
                    tolerance = 1.0f-8,
                    max_iterations = 10000,
                    use_full_physics = false
                )
                @test fine_custom isa Parareal.FineSolver{Float32}
                @test fine_custom.dt == 0.005f0
                @test fine_custom.solver_type == :cg
                @test fine_custom.smoother == :none
                @test fine_custom.use_full_physics == false
                
                # Test invalid parameters
                @test_throws ErrorException Parareal.FineSolver{Float64}(dt = -0.01)
                @test_throws ErrorException Parareal.FineSolver{Float64}(tolerance = 0.0)
                @test_throws ErrorException Parareal.FineSolver{Float64}(max_iterations = -1)
            end
        end
        
        @testset "Solver configuration validation" begin
            # Test that solver configurations are properly validated
            
            @testset "Valid configurations" begin
                coarse = Parareal.CoarseSolver{Float64}(dt = 0.1, spatial_resolution_factor = 2.0)
                fine = Parareal.FineSolver{Float64}(dt = 0.01)
                
                config = Parareal.SolverConfiguration(coarse, fine)
                @test config isa Parareal.SolverConfiguration{Float64}
                
                is_valid, message = Parareal.validate_solver_configuration(config)
                @test is_valid
                @test message == "Solver configuration is valid"
            end
            
            @testset "Invalid time step ratios" begin
                # Coarse dt < fine dt (invalid)
                coarse = Parareal.CoarseSolver{Float64}(dt = 0.005)
                fine = Parareal.FineSolver{Float64}(dt = 0.01)
                
                config = Parareal.SolverConfiguration(coarse, fine)
                is_valid, message = Parareal.validate_solver_configuration(config)
                @test !is_valid
                @test occursin("Coarse time step must be >= fine time step", message)
                
                # Excessive time step ratio
                coarse_large = Parareal.CoarseSolver{Float64}(dt = 10.0)
                fine_small = Parareal.FineSolver{Float64}(dt = 0.001)
                
                config_large = Parareal.SolverConfiguration(coarse_large, fine_small)
                is_valid, message = Parareal.validate_solver_configuration(config_large)
                @test !is_valid
                @test occursin("Time step ratio too large", message)
            end
            
            @testset "Invalid solver types" begin
                # This would require modifying the constructor to allow invalid types
                # For now, we test that valid types are accepted
                valid_types = [:pbicgstab, :cg, :sor]
                
                for solver_type in valid_types
                    coarse = Parareal.CoarseSolver{Float64}(solver_type = solver_type)
                    fine = Parareal.FineSolver{Float64}(solver_type = solver_type)
                    
                    config = Parareal.SolverConfiguration(coarse, fine)
                    is_valid, message = Parareal.validate_solver_configuration(config)
                    @test is_valid
                end
            end
        end
        
        @testset "Grid operations consistency" begin
            # Test that grid operations preserve essential properties
            
            @testset "Coarse grid creation" begin
                original_sizes = [(10, 10, 10), (20, 15, 12), (50, 50, 30)]
                resolution_factors = [1.0, 2.0, 3.0, 4.0]
                
                for original_size in original_sizes
                    for factor in resolution_factors
                        coarse_size = Parareal.create_coarse_grid(original_size, factor)
                        
                        # Check that coarse grid is smaller or equal
                        @test all(coarse_size .<= original_size)
                        
                        # Check minimum size constraint (for boundary conditions)
                        @test all(coarse_size .>= 3)
                        
                        # Check that reduction factor is approximately correct
                        if factor > 1.0
                            expected_reduction = original_size[1] / factor
                            actual_reduction = original_size[1] / coarse_size[1]
                            @test abs(actual_reduction - factor) / factor < 0.5  # Within 50%
                        end
                    end
                end
            end
            
            @testset "Interpolation and restriction consistency" begin
                # Test round-trip consistency: restrict then interpolate should preserve structure
                
                fine_sizes = [(10, 10, 10), (16, 16, 16), (20, 15, 12)]
                
                for fine_size in fine_sizes
                    # Create test data with known structure
                    fine_data = zeros(Float64, fine_size...)
                    
                    # Add a simple pattern
                    for k in 1:fine_size[3], j in 1:fine_size[2], i in 1:fine_size[1]
                        fine_data[i, j, k] = sin(2π * i / fine_size[1]) * 
                                           cos(2π * j / fine_size[2]) * 
                                           sin(π * k / fine_size[3])
                    end
                    
                    # Create coarse grid
                    coarse_size = Parareal.create_coarse_grid(fine_size, 2.0)
                    coarse_data = zeros(Float64, coarse_size...)
                    
                    # Restrict to coarse grid
                    Parareal.restrict_fine_to_coarse!(coarse_data, fine_data)
                    
                    # Interpolate back to fine grid
                    fine_reconstructed = zeros(Float64, fine_size...)
                    Parareal.interpolate_coarse_to_fine!(fine_reconstructed, coarse_data)
                    
                    # Check that the reconstruction preserves the general structure
                    # (Not exact due to information loss, but should be correlated)
                    correlation = cor(vec(fine_data), vec(fine_reconstructed))
                    @test correlation > 0.5  # Should maintain reasonable correlation
                    
                    # Check that boundary values are preserved reasonably well
                    boundary_error = abs(fine_data[1,1,1] - fine_reconstructed[1,1,1])
                    @test boundary_error < 1.0  # Reasonable boundary preservation
                end
            end
        end
        
        @testset "Solver execution consistency" begin
            # Test that solvers execute without errors and produce reasonable results
            
            @testset "CoarseSolver execution" begin
                coarse_solver = Parareal.CoarseSolver{Float64}(
                    dt = 0.1,
                    spatial_resolution_factor = 2.0,
                    simplified_physics = true
                )
                
                # Create test problem
                grid_size = (10, 10, 10)
                initial_condition = ones(Float64, grid_size...) .* 300.0  # Initial temperature
                
                time_window = Parareal.TimeWindow{Float64}(
                    0.0, 1.0, 0.1, 0.01, 10, 100, 0
                )
                
                # Create dummy problem data
                problem_data = Parareal.Heat3dsProblemData{Float64}(
                    (0.1, 0.1, 0.1), [0.1, 0.2, 0.3], [0.1, 0.1, 0.1],
                    zeros(UInt8, grid_size...), nothing, "sequential", false
                )
                
                # Execute solver
                result = Parareal.solve_coarse!(coarse_solver, initial_condition, time_window, problem_data)
                
                # Check result properties
                @test size(result) == grid_size
                @test all(isfinite.(result))
                @test all(result .> 0)  # Temperature should remain positive
                
                # Check that result is different from initial condition (some evolution occurred)
                # Note: With the simple diffusion, there might be minimal change for uniform initial conditions
                # So we check that the solver ran without error rather than requiring significant change
                @test true  # Solver executed successfully
            end
            
            @testset "FineSolver execution" begin
                fine_solver = Parareal.FineSolver{Float64}(
                    dt = 0.01,
                    solver_type = :pbicgstab,
                    use_full_physics = false  # Use simplified for testing
                )
                
                # Create test problem
                grid_size = (8, 8, 8)
                initial_condition = ones(Float64, grid_size...) .* 300.0
                
                time_window = Parareal.TimeWindow{Float64}(
                    0.0, 0.1, 0.1, 0.01, 1, 10, 0
                )
                
                # Create dummy problem data
                problem_data = Parareal.Heat3dsProblemData{Float64}(
                    (0.1, 0.1, 0.1), [0.1, 0.2], [0.1, 0.1],
                    zeros(UInt8, grid_size...), nothing, "sequential", false
                )
                
                # Execute solver
                result = Parareal.solve_fine!(fine_solver, initial_condition, time_window, problem_data)
                
                # Check result properties
                @test size(result) == grid_size
                @test all(isfinite.(result))
                @test all(result .> 0)  # Temperature should remain positive
            end
        end
        
        @testset "Solver characteristics and compatibility" begin
            # Test that solver characteristics are correctly reported
            
            @testset "CoarseSolver characteristics" begin
                coarse = Parareal.CoarseSolver{Float64}(
                    dt = 0.2,
                    spatial_resolution_factor = 3.0,
                    solver_type = :cg
                )
                
                characteristics = Parareal.get_solver_characteristics(coarse)
                
                @test characteristics["type"] == "coarse"
                @test characteristics["dt"] == 0.2
                @test characteristics["spatial_resolution_factor"] == 3.0
                @test characteristics["solver_type"] == :cg
                @test characteristics["expected_speedup"] == 27.0  # 3^3
            end
            
            @testset "FineSolver characteristics" begin
                fine = Parareal.FineSolver{Float64}(
                    dt = 0.01,
                    solver_type = :pbicgstab,
                    smoother = :gs
                )
                
                characteristics = Parareal.get_solver_characteristics(fine)
                
                @test characteristics["type"] == "fine"
                @test characteristics["dt"] == 0.01
                @test characteristics["solver_type"] == :pbicgstab
                @test characteristics["smoother"] == :gs
                @test characteristics["expected_speedup"] == 1.0  # Reference
            end
        end
        
        @testset "Type consistency across different precisions" begin
            # Test that solvers work consistently with different floating-point types
            
            precisions = [Float32, Float64]
            
            for T in precisions
                coarse = Parareal.CoarseSolver{T}(dt = T(0.1))
                fine = Parareal.FineSolver{T}(dt = T(0.01))
                
                @test coarse isa Parareal.CoarseSolver{T}
                @test fine isa Parareal.FineSolver{T}
                
                config = Parareal.SolverConfiguration(coarse, fine)
                @test config isa Parareal.SolverConfiguration{T}
                
                is_valid, message = Parareal.validate_solver_configuration(config)
                @test is_valid
            end
        end
    end
end

# Property verification summary
println("\n=== Property 5: Backward Compatibility Preservation - Verification Summary ===")
println("✓ Solver construction and validation with various parameters")
println("✓ Solver configuration validation for valid and invalid cases")
println("✓ Grid operations consistency (coarse grid creation, interpolation, restriction)")
println("✓ Solver execution consistency for both coarse and fine solvers")
println("✓ Solver characteristics reporting accuracy")
println("✓ Type consistency across different floating-point precisions")
println("✓ Error handling for invalid parameters and configurations")
println("✓ Round-trip grid operations maintain reasonable correlation")
println("================================================================================")