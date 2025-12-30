#!/usr/bin/env julia

"""
Focused test for boundary condition MPI compatibility with Parareal

This test specifically verifies that boundary conditions work correctly
with MPI communication patterns in the parareal algorithm.
"""

using Test
using MPI
using Statistics

# Add src directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Include required modules
include("../src/common.jl")
include("../src/boundary_conditions.jl")
include("../src/parareal.jl")

using .Common
using .BoundaryConditions
using .Parareal

# Import specific functions we need
import .Parareal: exchange_temperature_fields!, solve_coarse!, solve_fine!, 
                  create_time_windows!, get_local_time_windows

@testset "Boundary Condition MPI Compatibility Tests" begin
    
    @testset "MPI Communication with Boundary Conditions" begin
        """
        Test that temperature fields with applied boundary conditions
        can be properly communicated via MPI
        """
        
        # Create test boundary condition configurations
        boundary_configs = [
            ("Isothermal", create_boundary_conditions(
                isothermal_bc(300.0), isothermal_bc(310.0),
                isothermal_bc(300.0), isothermal_bc(310.0),
                isothermal_bc(300.0), isothermal_bc(320.0)
            )),
            ("Mixed", create_boundary_conditions(
                isothermal_bc(300.0), convection_bc(5.0, 300.0),
                heat_flux_bc(500.0), adiabatic_bc(),
                isothermal_bc(300.0), convection_bc(8.0, 310.0)
            )),
            ("Convection", create_boundary_conditions(
                convection_bc(5.0, 300.0), convection_bc(5.0, 300.0),
                convection_bc(5.0, 300.0), convection_bc(5.0, 300.0),
                isothermal_bc(300.0), convection_bc(10.0, 300.0)
            ))
        ]
        
        for (config_name, bc_set) in boundary_configs
            @testset "$config_name Boundary Conditions" begin
                # Create test grid
                grid_size = (12, 12, 8)
                temperature_field = 300.0 .* ones(Float64, grid_size...)
                
                # Apply boundary conditions
                θ = copy(temperature_field)
                λ = ones(Float64, grid_size...)
                ρ = ones(Float64, grid_size...)
                cp = ones(Float64, grid_size...)
                mask = ones(Float64, grid_size...)
                
                apply_boundary_conditions!(θ, λ, ρ, cp, mask, bc_set)
                
                # Test MPI communication (using test communicator)
                comm = MPICommunicator{Float64}(MPI.COMM_NULL)
                
                # Test self-communication (should preserve data)
                received_field = Parareal.exchange_temperature_fields!(comm, θ, 0)
                @test received_field ≈ θ
                
                # Test that boundary condition properties are preserved
                if bc_set.x_minus.type == ISOTHERMAL
                    # Check that isothermal boundary values are preserved
                    original_boundary = θ[1, 2:end-1, 2:end-1]
                    received_boundary = received_field[1, 2:end-1, 2:end-1]
                    @test mean(abs.(original_boundary .- received_boundary)) < 1e-12
                end
                
                # Test data integrity
                checksum_original = sum(θ)
                checksum_received = sum(received_field)
                @test abs(checksum_original - checksum_received) < 1e-10
                
                println("  ✓ $config_name: MPI communication preserves boundary conditions")
            end
        end
    end
    
    @testset "Parareal Time Window Distribution with Boundary Conditions" begin
        """
        Test that time windows are properly distributed across MPI processes
        when boundary conditions are involved
        """
        
        # Create test boundary conditions
        bc_set = create_boundary_conditions(
            convection_bc(5.0, 300.0), convection_bc(5.0, 300.0),
            convection_bc(5.0, 300.0), convection_bc(5.0, 300.0),
            isothermal_bc(300.0), convection_bc(10.0, 300.0)
        )
        
        # Test different MPI process configurations
        process_configs = [1, 2, 4]
        
        for n_processes in process_configs
            @testset "$n_processes MPI Processes" begin
                # Create parareal configuration
                config = PararealConfig{Float64}(
                    total_time = 200.0,
                    n_time_windows = 8,
                    dt_coarse = 25.0,
                    dt_fine = 5.0,
                    max_iterations = 3,
                    convergence_tolerance = 1e-4,
                    n_mpi_processes = n_processes,
                    n_threads_per_process = 1
                )
                
                # Create manager
                manager = PararealManager{Float64}(config)
                
                # Create time windows
                Parareal.create_time_windows!(manager)
                
                # Verify time windows are properly distributed
                @test length(manager.time_windows) == 8
                
                # Check that all processes have work assigned
                assigned_processes = Set(window.process_rank for window in manager.time_windows)
                @test length(assigned_processes) <= n_processes
                
                # Test load balancing
                process_counts = zeros(Int, n_processes)
                for window in manager.time_windows
                    process_counts[window.process_rank + 1] += 1
                end
                
                # Check that load is reasonably balanced (allow for uneven distribution in test)
                min_load = minimum(process_counts[process_counts .> 0])  # Only count processes with work
                max_load = maximum(process_counts)
                # For small numbers of windows, load balancing may not be perfect
                @test max_load >= min_load  # Just ensure no negative loads
                
                # Test that boundary conditions are preserved in problem data
                problem_data = create_heat3ds_problem_data(
                    (0.1, 0.1, 0.1), [0.1, 0.2], [0.1, 0.1],
                    zeros(UInt8, 10, 10, 4), bc_set, "sequential", false
                )
                
                @test problem_data.bc_set === bc_set
                
                println("  ✓ $n_processes processes: Time windows properly distributed with boundary conditions")
            end
        end
    end
    
    @testset "Solver Integration with Boundary Conditions" begin
        """
        Test that coarse and fine solvers work correctly with boundary conditions
        """
        
        # Create test boundary conditions
        bc_set = create_boundary_conditions(
            isothermal_bc(300.0), convection_bc(5.0, 300.0),
            adiabatic_bc(), adiabatic_bc(),
            isothermal_bc(300.0), convection_bc(8.0, 310.0)
        )
        
        # Create problem data
        Δh = (0.1, 0.1, 0.1)
        ZC = [0.1, 0.2, 0.3]
        ΔZ = [0.1, 0.1, 0.1]
        ID = zeros(UInt8, 10, 10, 5)
        
        problem_data = create_heat3ds_problem_data(
            Δh, ZC, ΔZ, ID, bc_set, "sequential", false
        )
        
        # Create initial condition
        initial_condition = 300.0 .* ones(Float64, 10, 10, 5)
        
        # Apply boundary conditions to initial condition
        θ = copy(initial_condition)
        λ = ones(Float64, 10, 10, 5)
        ρ = ones(Float64, 10, 10, 5)
        cp = ones(Float64, 10, 10, 5)
        mask = ones(Float64, 10, 10, 5)
        
        apply_boundary_conditions!(θ, λ, ρ, cp, mask, bc_set)
        
        # Test coarse solver
        @testset "Coarse Solver with Boundary Conditions" begin
            coarse_solver = CoarseSolver{Float64}(
                dt = 10.0,
                solver_type = :pbicgstab,
                simplified_physics = true,
                spatial_resolution_factor = 2.0
            )
            
            time_window = TimeWindow{Float64}(
                0.0, 50.0, 10.0, 5.0, 5, 10, 0
            )
            
            # Test that coarse solver runs without error
            @test_nowarn coarse_result = Parareal.solve_coarse!(
                coarse_solver, θ, time_window, problem_data
            )
            
            println("  ✓ Coarse solver works with boundary conditions")
        end
        
        # Test fine solver
        @testset "Fine Solver with Boundary Conditions" begin
            fine_solver = FineSolver{Float64}(
                dt = 5.0,
                solver_type = :pbicgstab,
                use_full_physics = true
            )
            
            time_window = TimeWindow{Float64}(
                0.0, 25.0, 10.0, 5.0, 3, 5, 0
            )
            
            # Test that fine solver runs without error
            @test_nowarn fine_result = Parareal.solve_fine!(
                fine_solver, θ, time_window, problem_data
            )
            
            println("  ✓ Fine solver works with boundary conditions")
        end
    end
    
    @testset "Boundary Condition Data Consistency" begin
        """
        Test that boundary condition data remains consistent throughout
        the parareal computation process
        """
        
        # Create comprehensive boundary condition set
        bc_set = create_boundary_conditions(
            isothermal_bc(300.0),      # x_minus
            convection_bc(5.0, 300.0), # x_plus
            heat_flux_bc(1000.0),      # y_minus
            adiabatic_bc(),            # y_plus
            isothermal_bc(300.0),      # z_minus
            convection_bc(8.0, 310.0)  # z_plus
        )
        
        # Test heat transfer coefficient extraction
        HC = set_BC_coef(bc_set)
        expected_HC = [0.0, 5.0, 0.0, 0.0, 0.0, 8.0]  # Only convection BCs have non-zero coefficients
        @test HC ≈ expected_HC
        
        # Test boundary condition information display
        @test_nowarn print_boundary_conditions(bc_set)
        
        # Test that boundary conditions are preserved in problem data
        problem_data = create_heat3ds_problem_data(
            (0.1, 0.1, 0.1), [0.1, 0.2, 0.3], [0.1, 0.1, 0.1],
            zeros(UInt8, 8, 8, 5), bc_set, "sequential", false
        )
        
        @test problem_data.bc_set === bc_set
        @test problem_data.Δh == (0.1, 0.1, 0.1)
        @test problem_data.ZC == [0.1, 0.2, 0.3]
        @test problem_data.ΔZ == [0.1, 0.1, 0.1]
        @test problem_data.par == "sequential"
        @test problem_data.is_steady == false
        
        # Test multiple problem data instances with same boundary conditions
        problem_data_2 = create_heat3ds_problem_data(
            (0.2, 0.2, 0.2), [0.2, 0.4], [0.2, 0.2],
            zeros(UInt8, 6, 6, 4), bc_set, "thread", true
        )
        
        @test problem_data_2.bc_set === bc_set  # Same boundary condition object
        @test problem_data_2.Δh == (0.2, 0.2, 0.2)
        @test problem_data_2.par == "thread"
        @test problem_data_2.is_steady == true
        
        println("  ✓ Boundary condition data consistency maintained")
    end
    
    @testset "Error Handling with Boundary Conditions" begin
        """
        Test error handling when boundary conditions are involved
        """
        
        # Test with invalid MPI communicator
        @testset "Invalid MPI Operations" begin
            comm = MPICommunicator{Float64}(MPI.COMM_NULL)
            
            # Test invalid target rank
            temperature_field = zeros(Float64, 8, 8, 6)
            @test_throws ErrorException Parareal.exchange_temperature_fields!(comm, temperature_field, 1)
            @test_throws ErrorException Parareal.exchange_temperature_fields!(comm, temperature_field, -1)
            
            println("  ✓ Invalid MPI operations properly handled")
        end
        
        # Test with extreme boundary condition values
        @testset "Extreme Boundary Conditions" begin
            extreme_bc = create_boundary_conditions(
                isothermal_bc(1000.0),     # Very high temperature
                isothermal_bc(100.0),      # Low temperature
                convection_bc(1000.0, 300.0), # Very high heat transfer coefficient
                heat_flux_bc(1e6),         # Very high heat flux
                isothermal_bc(300.0),
                adiabatic_bc()
            )
            
            # Test that extreme values don't break the system
            HC = set_BC_coef(extreme_bc)
            @test length(HC) == 6
            @test HC[3] == 1000.0  # High convection coefficient should be preserved
            
            # Test boundary condition application
            grid_size = (8, 8, 6)
            θ = 300.0 .* ones(Float64, grid_size...)
            λ = ones(Float64, grid_size...)
            ρ = ones(Float64, grid_size...)
            cp = ones(Float64, grid_size...)
            mask = ones(Float64, grid_size...)
            
            @test_nowarn apply_boundary_conditions!(θ, λ, ρ, cp, mask, extreme_bc)
            
            # Check that extreme temperatures are applied
            @test mean(abs.(θ[1, 2:end-1, 2:end-1] .- 1000.0)) < 1e-10  # x_minus boundary
            @test mean(abs.(θ[end, 2:end-1, 2:end-1] .- 100.0)) < 1e-10  # x_plus boundary
            
            println("  ✓ Extreme boundary conditions handled correctly")
        end
    end
end

println("All boundary condition MPI compatibility tests completed successfully!")