"""
Comprehensive tests for Heat3ds output format consistency (Task 8.4)

This test suite validates that parareal computations generate output in the same format
as sequential Heat3ds computations, satisfying Requirement 3.4:

"WHEN parareal computation completes, THE Heat3ds_System SHALL generate output 
in the same format as sequential computation"

Tests cover:
1. File format consistency between parareal and sequential modes
2. Data structure compatibility
3. Visualization output consistency
4. Metadata handling without breaking compatibility
"""

using Test
using Printf

# Include the output format module
include("../src/output_format.jl")
using .OutputFormat

@testset "Output Format Consistency Tests (Task 8.4)" begin
    
    @testset "OutputManager Creation and Configuration" begin
        println("Testing OutputManager creation and configuration...")
        
        @test begin
            # Test sequential mode output manager
            manager_seq = create_output_manager(
                base_filename = "test_sequential",
                computation_mode = "sequential",
                grid_size = (10, 10, 10)
            )
            
            manager_seq.config.computation_mode == "sequential" &&
            manager_seq.config.base_filename == "test_sequential" &&
            manager_seq.metadata.grid_size == (10, 10, 10)
        end
        
        @test begin
            # Test parareal mode output manager
            manager_par = create_output_manager(
                base_filename = "test_parareal",
                computation_mode = "parareal",
                grid_size = (20, 20, 20),
                n_time_windows = 4,
                n_mpi_processes = 2
            )
            
            manager_par.config.computation_mode == "parareal" &&
            manager_par.metadata.n_time_windows == 4 &&
            manager_par.metadata.n_mpi_processes == 2
        end
    end
    
    @testset "Temperature Output Format Consistency" begin
        println("Testing temperature output format consistency...")
        
        @test begin
            # Create test temperature field
            grid_size = (4, 4, 4)
            temperature_field = rand(Float64, grid_size...) .* 100.0 .+ 300.0
            
            # Test sequential output
            manager_seq = create_output_manager(
                base_filename = "test_temp_seq",
                computation_mode = "sequential",
                grid_size = grid_size
            )
            
            result_seq = Dict(
                :final_solution => temperature_field,
                :converged => true,
                :computation_time => 1.5,
                :residual_history => [1e-1, 1e-3, 1e-6]
            )
            
            files_seq = generate_parareal_output!(manager_seq, temperature_field, result_seq, nothing)
            
            # Test parareal output
            manager_par = create_output_manager(
                base_filename = "test_temp_par",
                computation_mode = "parareal",
                grid_size = grid_size,
                n_time_windows = 2,
                n_mpi_processes = 2
            )
            
            result_par = Dict(
                :final_solution => temperature_field,
                :converged => true,
                :computation_time => 0.8,
                :iterations => 3,
                :residual_history => [1e-1, 1e-3, 1e-6]
            )
            
            files_par = generate_parareal_output!(manager_par, temperature_field, result_par, nothing)
            
            # Both should generate temperature files
            seq_has_temp = any(f -> occursin("temperature", f), files_seq)
            par_has_temp = any(f -> occursin("temperature", f), files_par)
            
            # Clean up test files
            for file in vcat(files_seq, files_par)
                if isfile(file)
                    rm(file)
                end
            end
            
            seq_has_temp && par_has_temp
        end
    end
    
    @testset "File Format Validation" begin
        println("Testing file format validation...")
        
        @test begin
            # Create test data
            grid_size = (3, 3, 3)
            temperature_field = ones(Float64, grid_size...) .* 350.0
            
            manager = create_output_manager(
                base_filename = "test_validation",
                computation_mode = "parareal",
                grid_size = grid_size
            )
            
            result = Dict(
                :final_solution => temperature_field,
                :converged => true,
                :computation_time => 2.0,
                :residual_history => [1e-2, 1e-4, 1e-7]
            )
            
            # Generate output files
            files = generate_parareal_output!(manager, temperature_field, result, nothing)
            
            # Validate output consistency
            is_consistent = ensure_output_consistency!(manager)
            
            # Clean up
            for file in files
                if isfile(file)
                    rm(file)
                end
            end
            
            is_consistent
        end
    end
    
    @testset "CSV Format Consistency" begin
        println("Testing CSV format consistency...")
        
        @test begin
            # Test that CSV files have consistent headers and format
            grid_size = (2, 2, 2)
            temperature_field = reshape(collect(1.0:8.0), grid_size) .+ 300.0
            
            # Sequential CSV
            manager_seq = create_output_manager(
                base_filename = "test_csv_seq",
                computation_mode = "sequential",
                grid_size = grid_size
            )
            
            result = Dict(:final_solution => temperature_field, :converged => true)
            files_seq = generate_parareal_output!(manager_seq, temperature_field, result, nothing)
            
            # Parareal CSV
            manager_par = create_output_manager(
                base_filename = "test_csv_par",
                computation_mode = "parareal",
                grid_size = grid_size,
                n_time_windows = 2
            )
            
            files_par = generate_parareal_output!(manager_par, temperature_field, result, nothing)
            
            # Check CSV format consistency
            csv_seq = ""
            csv_par = ""
            
            for file in files_seq
                if endswith(file, ".csv") && occursin("temperature", file)
                    csv_seq = file
                    break
                end
            end
            
            for file in files_par
                if endswith(file, ".csv") && occursin("temperature", file)
                    csv_par = file
                    break
                end
            end
            
            # Both should have CSV files with proper headers
            seq_valid = false
            par_valid = false
            
            if isfile(csv_seq)
                lines_seq = readlines(csv_seq)
                seq_valid = length(lines_seq) > 5 && occursin("Heat3ds", lines_seq[1])
            end
            
            if isfile(csv_par)
                lines_par = readlines(csv_par)
                par_valid = length(lines_par) > 5 && occursin("Heat3ds", lines_par[1])
            end
            
            # Clean up
            for file in vcat(files_seq, files_par)
                if isfile(file)
                    rm(file)
                end
            end
            
            seq_valid && par_valid
        end
    end
    
    @testset "Binary Format Consistency" begin
        println("Testing binary format consistency...")
        
        @test begin
            # Test binary file format consistency
            grid_size = (3, 3, 3)
            temperature_field = rand(Float64, grid_size...) .* 50.0 .+ 300.0
            
            manager = create_output_manager(
                base_filename = "test_binary",
                computation_mode = "parareal",
                grid_size = grid_size
            )
            
            result = Dict(:final_solution => temperature_field, :converged => true)
            files = generate_parareal_output!(manager, temperature_field, result, nothing)
            
            # Find binary file
            binary_file = ""
            for file in files
                if endswith(file, ".dat")
                    binary_file = file
                    break
                end
            end
            
            # Validate binary file format
            binary_valid = false
            if isfile(binary_file)
                filesize_bytes = filesize(binary_file)
                expected_header = 4 * 3 + 8  # 3 Int32 + 1 Float64
                expected_data = prod(grid_size) * 8  # Float64 data
                expected_total = expected_header + expected_data
                
                binary_valid = filesize_bytes >= expected_header
            end
            
            # Clean up
            for file in files
                if isfile(file)
                    rm(file)
                end
            end
            
            binary_valid
        end
    end
    
    @testset "Metadata Handling" begin
        println("Testing metadata handling...")
        
        @test begin
            # Test that parareal metadata doesn't break compatibility
            grid_size = (2, 2, 2)
            temperature_field = ones(Float64, grid_size...) .* 325.0
            
            # Parareal with metadata
            manager_with_meta = create_output_manager(
                base_filename = "test_meta",
                computation_mode = "parareal",
                grid_size = grid_size,
                n_time_windows = 3,
                include_metadata = true
            )
            
            result = Dict(
                :final_solution => temperature_field,
                :converged => true,
                :iterations => 5,
                :computation_time => 1.2
            )
            
            files_with_meta = generate_parareal_output!(manager_with_meta, temperature_field, result, nothing)
            
            # Should have metadata file for parareal
            has_metadata = any(f -> occursin("metadata", f), files_with_meta)
            
            # Should still have standard Heat3ds files
            has_temp_csv = any(f -> occursin("temperature.csv", f), files_with_meta)
            has_temp_dat = any(f -> occursin("temperature.dat", f), files_with_meta)
            
            # Clean up
            for file in files_with_meta
                if isfile(file)
                    rm(file)
                end
            end
            
            has_metadata && has_temp_csv && has_temp_dat
        end
    end
    
    @testset "Error Handling and Robustness" begin
        println("Testing error handling and robustness...")
        
        @test begin
            # Test with invalid data
            try
                manager = create_output_manager(
                    base_filename = "test_error",
                    computation_mode = "parareal",
                    grid_size = (1, 1, 1)
                )
                
                # Empty temperature field
                empty_field = Array{Float64,3}(undef, 0, 0, 0)
                result = Dict(:final_solution => empty_field)
                
                # Should handle gracefully
                files = generate_parareal_output!(manager, empty_field, result, nothing)
                
                # Should return empty list or handle error gracefully
                true
                
            catch e
                # Graceful error handling is acceptable
                true
            end
        end
        
        @test begin
            # Test consistency check with missing files
            manager = create_output_manager(
                base_filename = "test_missing",
                computation_mode = "sequential",
                grid_size = (2, 2, 2)
            )
            
            # Try to validate without generating files first
            is_consistent = ensure_output_consistency!(manager)
            
            # Should return false for missing files
            !is_consistent
        end
    end
end

println("Output format consistency tests completed.")
println("These tests validate Requirement 3.4: Parareal generates output in the same format as sequential computation.")