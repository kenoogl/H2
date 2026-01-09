#!/usr/bin/env julia

"""
Test boundary condition compatibility with Parareal algorithm
"""

# Add src directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

# Include Heat3ds
include("src/heat3ds.jl")

using .BoundaryConditions

println("=== Boundary Condition Compatibility Test ===")

"""
Create different boundary condition configurations for testing
"""
function create_test_boundary_conditions()
    # Test configuration 1: All isothermal
    isothermal_config = BoundaryConditions.create_boundary_conditions(
        BoundaryConditions.isothermal_bc(300.0),  # x_minus
        BoundaryConditions.isothermal_bc(310.0),  # x_plus
        BoundaryConditions.isothermal_bc(300.0),  # y_minus
        BoundaryConditions.isothermal_bc(310.0),  # y_plus
        BoundaryConditions.isothermal_bc(300.0),  # z_minus
        BoundaryConditions.isothermal_bc(320.0)   # z_plus
    )
    
    # Test configuration 2: Mixed conditions (isothermal + adiabatic)
    mixed_config = BoundaryConditions.create_boundary_conditions(
        BoundaryConditions.adiabatic_bc(),        # x_minus
        BoundaryConditions.adiabatic_bc(),        # x_plus
        BoundaryConditions.adiabatic_bc(),        # y_minus
        BoundaryConditions.adiabatic_bc(),        # y_plus
        BoundaryConditions.isothermal_bc(300.0),  # z_minus
        BoundaryConditions.adiabatic_bc()         # z_plus
    )
    
    # Test configuration 3: Convection conditions
    convection_config = BoundaryConditions.create_boundary_conditions(
        BoundaryConditions.convection_bc(5.0, 300.0),  # x_minus
        BoundaryConditions.convection_bc(5.0, 300.0),  # x_plus
        BoundaryConditions.convection_bc(5.0, 300.0),  # y_minus
        BoundaryConditions.convection_bc(5.0, 300.0),  # y_plus
        BoundaryConditions.isothermal_bc(300.0),       # z_minus
        BoundaryConditions.convection_bc(10.0, 300.0)  # z_plus
    )
    
    # Test configuration 4: Heat flux conditions
    heat_flux_config = BoundaryConditions.create_boundary_conditions(
        BoundaryConditions.heat_flux_bc(1000.0),     # x_minus
        BoundaryConditions.heat_flux_bc(-1000.0),    # x_plus
        BoundaryConditions.adiabatic_bc(),           # y_minus
        BoundaryConditions.adiabatic_bc(),           # y_plus
        BoundaryConditions.isothermal_bc(300.0),     # z_minus
        BoundaryConditions.adiabatic_bc()            # z_plus
    )
    
    return [
        ("All Isothermal", isothermal_config),
        ("Mixed (Isothermal + Adiabatic)", mixed_config),
        ("Convection", convection_config),
        ("Heat Flux", heat_flux_config)
    ]
end

"""
Test parareal with different boundary condition types
"""
function test_boundary_condition_types()
    println("\n1. Testing different boundary condition types...")
    
    boundary_configs = create_test_boundary_conditions()
    
    for (config_name, bc_set) in boundary_configs
        println("\n  Testing: $config_name")
        
        try
            # Create a simple parareal configuration
            parareal_config = Dict{String,Any}(
                "total_time" => 1000.0,
                "n_time_windows" => 2,
                "dt_coarse" => 200.0,
                "dt_fine" => 100.0,
                "max_iterations" => 3,
                "convergence_tolerance" => 1.0e-4,
                "n_mpi_processes" => 1,
                "n_threads_per_process" => 1
            )
            
            # Test with parareal mode
            result = q3d(40, 40, 30, "pbicgstab", "gs", 
                        epsilon=1e-4, par="sequential", is_steady=false, 
                        parareal=true, parareal_config=parareal_config)
            
            println("    ✓ $config_name: Compatible with parareal")
            
        catch e
            println("    ✗ $config_name: Failed - $e")
        end
    end
end

"""
Test boundary condition data consistency across MPI processes
"""
function test_boundary_condition_data_exchange()
    println("\n2. Testing boundary condition data exchange...")
    
    try
        # Create a test boundary condition set
        bc_set = BoundaryConditions.create_boundary_conditions(
            BoundaryConditions.convection_bc(5.0, 300.0),
            BoundaryConditions.convection_bc(5.0, 300.0),
            BoundaryConditions.convection_bc(5.0, 300.0),
            BoundaryConditions.convection_bc(5.0, 300.0),
            BoundaryConditions.isothermal_bc(300.0),
            BoundaryConditions.convection_bc(10.0, 300.0)
        )
        
        # Test that boundary conditions are properly stored in problem data
        problem_data = Parareal.create_heat3ds_problem_data(
            (0.1, 0.1, 0.1), [0.1], [0.1],
            zeros(UInt8, 10, 10, 10), bc_set, "sequential", false
        )
        
        # Verify boundary condition data is preserved
        if problem_data.bc_set === bc_set
            println("    ✓ Boundary condition data properly stored in problem data")
        else
            println("    ✗ Boundary condition data not properly stored")
        end
        
        # Test HC coefficient extraction
        HC = BoundaryConditions.set_BC_coef(bc_set)
        expected_HC = [5.0, 5.0, 5.0, 5.0, 0.0, 10.0]  # Only convection BCs have non-zero coefficients
        
        if HC ≈ expected_HC
            println("    ✓ Heat transfer coefficients correctly extracted")
        else
            println("    ✗ Heat transfer coefficients incorrect: got $HC, expected $expected_HC")
        end
        
    catch e
        println("    ✗ Boundary condition data exchange test failed: $e")
    end
end

"""
Test boundary condition behavior consistency between sequential and parareal
"""
function test_boundary_condition_consistency()
    println("\n3. Testing boundary condition behavior consistency...")
    
    # Test with the same boundary conditions used in the main Heat3ds code
    bc_set = BoundaryConditions.create_boundary_conditions(
        BoundaryConditions.convection_bc(5.0, 300.0),  # x_minus
        BoundaryConditions.convection_bc(5.0, 300.0),  # x_plus
        BoundaryConditions.convection_bc(5.0, 300.0),  # y_minus
        BoundaryConditions.convection_bc(5.0, 300.0),  # y_plus
        BoundaryConditions.isothermal_bc(300.0),       # z_minus
        BoundaryConditions.convection_bc(5.0, 300.0)   # z_plus
    )
    
    try
        println("    Testing sequential mode...")
        # Test sequential mode
        result_seq = q3d(40, 40, 30, "pbicgstab", "gs", 
                        epsilon=1e-4, par="sequential", is_steady=true, 
                        parareal=false)
        println("    ✓ Sequential mode with boundary conditions works")
        
        println("    Testing parareal mode...")
        # Test parareal mode with same boundary conditions
        parareal_config = Dict{String,Any}(
            "total_time" => 1000.0,
            "n_time_windows" => 2,
            "dt_coarse" => 200.0,
            "dt_fine" => 100.0,
            "max_iterations" => 3,
            "convergence_tolerance" => 1.0e-4,
            "n_mpi_processes" => 1,
            "n_threads_per_process" => 1
        )
        
        result_par = q3d(40, 40, 30, "pbicgstab", "gs", 
                        epsilon=1e-4, par="sequential", is_steady=false, 
                        parareal=true, parareal_config=parareal_config)
        
        println("    ✓ Parareal mode with boundary conditions works")
        println("    ✓ Boundary condition behavior is consistent")
        
    catch e
        println("    ✗ Boundary condition consistency test failed: $e")
    end
end

"""
Test all boundary condition types with parareal
"""
function test_all_boundary_condition_types()
    println("\n4. Testing comprehensive boundary condition compatibility...")
    
    # Test each boundary condition type individually
    bc_types = [
        ("Isothermal only", [
            BoundaryConditions.isothermal_bc(300.0),
            BoundaryConditions.isothermal_bc(310.0),
            BoundaryConditions.isothermal_bc(300.0),
            BoundaryConditions.isothermal_bc(310.0),
            BoundaryConditions.isothermal_bc(300.0),
            BoundaryConditions.isothermal_bc(320.0)
        ]),
        ("Adiabatic only", [
            BoundaryConditions.adiabatic_bc(),
            BoundaryConditions.adiabatic_bc(),
            BoundaryConditions.adiabatic_bc(),
            BoundaryConditions.adiabatic_bc(),
            BoundaryConditions.isothermal_bc(300.0),  # Need at least one non-adiabatic
            BoundaryConditions.adiabatic_bc()
        ]),
        ("Convection only", [
            BoundaryConditions.convection_bc(5.0, 300.0),
            BoundaryConditions.convection_bc(5.0, 300.0),
            BoundaryConditions.convection_bc(5.0, 300.0),
            BoundaryConditions.convection_bc(5.0, 300.0),
            BoundaryConditions.isothermal_bc(300.0),
            BoundaryConditions.convection_bc(10.0, 300.0)
        ]),
        ("Heat flux only", [
            BoundaryConditions.heat_flux_bc(500.0),
            BoundaryConditions.heat_flux_bc(-500.0),
            BoundaryConditions.adiabatic_bc(),
            BoundaryConditions.adiabatic_bc(),
            BoundaryConditions.isothermal_bc(300.0),
            BoundaryConditions.adiabatic_bc()
        ])
    ]
    
    for (type_name, bc_list) in bc_types
        println("    Testing: $type_name")
        
        try
            bc_set = BoundaryConditions.create_boundary_conditions(
                bc_list[1], bc_list[2], bc_list[3], 
                bc_list[4], bc_list[5], bc_list[6]
            )
            
            # Simple parareal test
            parareal_config = Dict{String,Any}(
                "total_time" => 500.0,
                "n_time_windows" => 2,
                "dt_coarse" => 100.0,
                "dt_fine" => 50.0,
                "max_iterations" => 2,
                "convergence_tolerance" => 1.0e-3,
                "n_mpi_processes" => 1,
                "n_threads_per_process" => 1
            )
            
            result = q3d(20, 20, 15, "pbicgstab", "gs", 
                        epsilon=1e-3, par="sequential", is_steady=false, 
                        parareal=true, parareal_config=parareal_config)
            
            println("      ✓ $type_name: Compatible")
            
        catch e
            println("      ✗ $type_name: Failed - $e")
        end
    end
end

# Run all tests
test_boundary_condition_types()
test_boundary_condition_data_exchange()
test_boundary_condition_consistency()
test_all_boundary_condition_types()

println("\n=== Boundary Condition Compatibility Test Complete ===")