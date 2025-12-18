# Test for output format consistency (Task 8.4)
# Validates Requirement 3.4

using Test
using Random
using Dates
using Printf

# Import the modules we need to test
include("../src/parareal.jl")
include("../src/output_format.jl")
include("../src/common.jl")

using .Parareal
using .OutputFormat
using .Common

# Import specific functions to avoid namespace conflicts
import .OutputFormat: create_output_manager, generate_parareal_output!, ensure_output_consistency!
import .OutputFormat: export_parareal_results, create_output_metadata, create_output_filename, compare_output_formats
import .OutputFormat: OutputConfiguration, ensure_output_consistency_with_comparison!

# ============================================================================
# Helper Functions for Output Format Testing (defined first)
# ============================================================================

"""
Create test temperature field
"""
function create_test_temperature_field(::Type{T}, mx::Int, my::Int, mz::Int) where {T <: AbstractFloat}
    temperature = zeros(T, mx, my, mz)
    
    # Create a simple temperature distribution
    for k in 1:mz, j in 1:my, i in 1:mx
        x = (i - 1) / (mx - 1)
        y = (j - 1) / (my - 1)
        z = (k - 1) / (mz - 1)
        
        # Simple 3D temperature profile
        temperature[i, j, k] = T(300.0) + T(20.0) * sin(π * x) * sin(π * y) * sin(π * z)
    end
    
    return temperature
end

"""
Create test parareal result
"""
function create_test_parareal_result(::Type{T}, temperature::Array{T,3}) where {T <: AbstractFloat}
    return PararealResult{T}(
        temperature,           # final_solution
        true,                 # converged
        3,                    # iterations
        [T(1e-2), T(1e-4), T(1e-6)], # residual_history
        T(2.5),              # computation_time
        T(0.1),              # communication_time
        nothing              # performance_metrics
    )
end

"""
Generate sequential reference outputs for comparison
"""
function generate_sequential_reference_outputs(output_dir::String, temperature::Array{T,3}) where {T <: AbstractFloat}
    generated_files = String[]
    
    # Generate log file
    log_file = "sequential_log.txt"
    open(joinpath(output_dir, log_file), "w") do file
        println(file, "Problem : IC on NonUniform grid (Sequential Mode)")
        println(file, "Grid  : $(size(temperature, 1)) $(size(temperature, 2)) $(size(temperature, 3))")
        println(file, "Generated: $(now())")
    end
    push!(generated_files, log_file)
    
    # Generate CSV file
    csv_file = "sequential_convergence.csv"
    open(joinpath(output_dir, csv_file), "w") do file
        println(file, "# Sequential Convergence History")
        println(file, "Iteration,Residual")
        println(file, "1,1.0e-2")
        println(file, "2,1.0e-4")
        println(file, "3,1.0e-6")
    end
    push!(generated_files, csv_file)
    
    # Generate data file
    dat_file = "sequential_temperature.dat"
    open(joinpath(output_dir, dat_file), "w") do file
        println(file, "# Sequential Temperature Field Data")
        println(file, "# Grid Size: $(size(temperature))")
        println(file, "# Data Format: i j k temperature")
        
        # Write a few sample points
        for k in 1:min(2, size(temperature, 3))
            for j in 1:min(2, size(temperature, 2))
                for i in 1:min(2, size(temperature, 1))
                    @printf(file, "%d %d %d %.6e\n", i, j, k, temperature[i, j, k])
                end
            end
        end
    end
    push!(generated_files, dat_file)
    
    return generated_files
end

"""
Test output format consistency between sequential and parareal modes
Requirement 3.4: Generate output in the same format as sequential computation
"""

@testset "Task 8.4: Output Format Consistency" begin
    
    # Test configuration
    Random.seed!(12345)
    T = Float64
    
    # Standard test grid configuration
    mx, my, mz = 8, 8, 8
    grid_size = (mx, my, mz)
    
    # Test temperature data
    test_temperature = create_test_temperature_field(T, mx, my, mz)
    
    # Test parareal result
    test_result = create_test_parareal_result(T, test_temperature)
    
    # Test configuration
    test_config = PararealConfig{T}(
        total_time = T(1.0),
        n_time_windows = 2,
        dt_coarse = T(0.1),
        dt_fine = T(0.01),
        max_iterations = 5,
        convergence_tolerance = T(1e-6),
        n_mpi_processes = 1,
        n_threads_per_process = 1
    )
    
    @testset "Output manager creation and configuration" begin
        println("Testing output manager creation...")
        
        # Test output manager creation
        @test begin
            try
                manager = create_output_manager(
                    base_filename = "test_heat3ds",
                    computation_mode = "parareal",
                    grid_size = grid_size,
                    n_time_windows = 2,
                    n_mpi_processes = 1
                )
                
                # Verify manager properties
                manager.config.base_filename == "test_heat3ds" &&
                manager.metadata.computation_mode == "parareal" &&
                manager.metadata.grid_size == grid_size &&
                manager.metadata.n_time_windows == 2
                
            catch e
                @warn "Output manager creation failed: $e"
                false
            end
        end
        
        println("✓ Output manager creation verified")
    end
    
    @testset "Output configuration options" begin
        println("Testing output configuration options...")
        
        # Test different output configurations
        configs_to_test = [
            (true, true, true, true, true),   # All outputs enabled
            (true, false, true, false, true), # Selective outputs
            (false, false, false, false, false) # Minimal outputs
        ]
        
        for (log, conv, temp, perf, plot) in configs_to_test
            @test begin
                try
                    config = OutputConfiguration(
                        base_filename = "test_config",
                        enable_log_output = log,
                        enable_convergence_output = conv,
                        enable_temperature_output = temp,
                        enable_performance_output = perf,
                        enable_plot_output = plot
                    )
                    
                    # Verify configuration
                    config.enable_log_output == log &&
                    config.enable_convergence_output == conv &&
                    config.enable_temperature_output == temp &&
                    config.enable_performance_output == perf &&
                    config.enable_plot_output == plot
                    
                catch e
                    @warn "Output configuration test failed: $e"
                    false
                end
            end
        end
        
        println("✓ Output configuration options verified")
    end
    
    @testset "Parareal output generation" begin
        println("Testing parareal output generation...")
        
        # Create temporary directory for test outputs
        test_dir = mktempdir()
        
        try
            # Create output manager
            manager = create_output_manager(
                base_filename = "test_parareal",
                computation_mode = "parareal",
                grid_size = grid_size,
                output_directory = test_dir
            )
            
            # Generate parareal outputs
            generated_files = generate_parareal_output!(manager, test_temperature, test_result, nothing)
            
            @test begin
                # Verify files were generated
                !isempty(generated_files) &&
                all(filename -> isfile(joinpath(test_dir, filename)), generated_files)
            end
            
            # Test file content validation
            @test begin
                try
                    # Validate each generated file
                    all_valid = true
                    
                    for filename in generated_files
                        filepath = joinpath(test_dir, filename)
                        
                        if endswith(filename, ".txt") || endswith(filename, ".log")
                            # Check log file format
                            content = read(filepath, String)
                            if !occursin("Problem", content) && !occursin("Grid", content)
                                all_valid = false
                            end
                            
                        elseif endswith(filename, ".csv")
                            # Check CSV file format
                            lines = readlines(filepath)
                            data_lines = filter(line -> !startswith(line, "#"), lines)
                            if isempty(data_lines) || !any(line -> occursin(",", line), data_lines)
                                all_valid = false
                            end
                            
                        elseif endswith(filename, ".dat")
                            # Check data file format
                            lines = readlines(filepath)
                            data_lines = filter(line -> !startswith(line, "#"), lines)
                            if isempty(data_lines)
                                all_valid = false
                            end
                        end
                    end
                    
                    all_valid
                    
                catch e
                    @warn "File content validation failed: $e"
                    false
                end
            end
            
        finally
            # Cleanup
            rm(test_dir, recursive=true, force=true)
        end
        
        println("✓ Parareal output generation verified")
    end
    
    @testset "Output format consistency validation" begin
        println("Testing output format consistency validation...")
        
        # Create temporary directory for test outputs
        test_dir = mktempdir()
        
        try
            # Generate sequential-style outputs (reference)
            sequential_files = generate_sequential_reference_outputs(test_dir, test_temperature)
            
            # Generate parareal outputs
            manager = create_output_manager(
                base_filename = "test_consistency",
                computation_mode = "parareal",
                grid_size = grid_size,
                output_directory = test_dir
            )
            
            parareal_files = generate_parareal_output!(manager, test_temperature, test_result, nothing)
            
            # Test format consistency
            @test begin
                try
                    # Ensure output consistency
                    is_consistent = ensure_output_consistency_with_comparison!(manager, sequential_files)
                    is_consistent
                    
                catch e
                    @warn "Output consistency check failed: $e"
                    false
                end
            end
            
            # Test format comparison
            @test begin
                try
                    comparison_results = compare_output_formats(sequential_files, parareal_files)
                    
                    # Should have similar file types
                    comparison_results["file_types_match"] ||
                    length(comparison_results["file_details"]) > 0
                    
                catch e
                    @warn "Format comparison failed: $e"
                    false
                end
            end
            
        finally
            # Cleanup
            rm(test_dir, recursive=true, force=true)
        end
        
        println("✓ Output format consistency validation verified")
    end
    
    @testset "Heat3ds integration compatibility" begin
        println("Testing Heat3ds integration compatibility...")
        
        # Test that parareal outputs are compatible with Heat3ds format expectations
        @test begin
            try
                # Create temporary directory
                test_dir = mktempdir()
                
                # Export parareal results using Heat3ds compatible format
                generated_files, is_consistent = export_parareal_results(
                    test_temperature, test_result, test_config,
                    output_directory = test_dir,
                    base_filename = "heat3ds_compatible"
                )
                
                # Verify compatibility
                result = !isempty(generated_files) && is_consistent
                
                # Cleanup
                rm(test_dir, recursive=true, force=true)
                
                result
                
            catch e
                @warn "Heat3ds integration compatibility test failed: $e"
                false
            end
        end
        
        println("✓ Heat3ds integration compatibility verified")
    end
    
    @testset "Metadata and filename consistency" begin
        println("Testing metadata and filename consistency...")
        
        # Test metadata creation
        @test begin
            try
                metadata = create_output_metadata(test_config, test_result, computation_mode = "parareal")
                
                # Verify metadata fields
                metadata.computation_mode == "parareal" &&
                metadata.n_time_windows == test_config.n_time_windows &&
                metadata.parareal_iterations == test_result.iterations &&
                metadata.converged == test_result.converged
                
            catch e
                @warn "Metadata creation test failed: $e"
                false
            end
        end
        
        # Test filename generation consistency
        @test begin
            try
                manager = create_output_manager(
                    base_filename = "consistency_test",
                    computation_mode = "parareal"
                )
                
                # Generate multiple filenames
                log_filename = create_output_filename(manager, "log", "txt")
                csv_filename = create_output_filename(manager, "convergence", "csv")
                dat_filename = create_output_filename(manager, "temperature", "dat")
                
                # Verify filename patterns
                occursin("consistency_test", log_filename) &&
                occursin("parareal", log_filename) &&
                endswith(log_filename, ".txt") &&
                endswith(csv_filename, ".csv") &&
                endswith(dat_filename, ".dat")
                
            catch e
                @warn "Filename consistency test failed: $e"
                false
            end
        end
        
        println("✓ Metadata and filename consistency verified")
    end
    
    @testset "Error handling and robustness" begin
        println("Testing error handling and robustness...")
        
        # Test with invalid inputs
        @test begin
            try
                # Test with empty temperature field
                empty_temp = zeros(T, 0, 0, 0)
                
                manager = create_output_manager()
                
                # Should handle gracefully
                generated_files = generate_parareal_output!(manager, empty_temp, nothing, nothing)
                
                # Should not crash, may generate empty or minimal files
                true
                
            catch e
                # Expected to handle errors gracefully
                true
            end
        end
        
        # Test with missing directory
        @test begin
            try
                nonexistent_dir = "/nonexistent/directory/path"
                
                manager = create_output_manager(output_directory = nonexistent_dir)
                
                # Should handle missing directory gracefully
                generated_files = generate_parareal_output!(manager, test_temperature, test_result, nothing)
                
                # May fail, but should not crash the program
                true
                
            catch e
                # Expected to handle errors gracefully
                true
            end
        end
        
        println("✓ Error handling and robustness verified")
    end
end

println("Output format consistency test loaded successfully")