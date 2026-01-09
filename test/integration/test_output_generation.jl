# Test for output file generation
using Test
using Random

# Import the modules we need to test
include("../src/parareal.jl")
include("../src/output_format.jl")

using .Parareal
using .OutputFormat

@testset "Output File Generation Tests" begin
    
    # Test configuration
    Random.seed!(12345)
    T = Float64
    
    # Standard test grid configuration
    mx, my, mz = 4, 4, 4
    grid_size = (mx, my, mz)
    
    # Test temperature data
    test_temperature = rand(T, mx, my, mz) .* 100.0 .+ 300.0  # 300-400K range
    
    # Test parareal result
    test_result = Parareal.PararealResult{T}(
        test_temperature,           # final_solution
        true,                      # converged
        3,                         # iterations
        [T(1e-2), T(1e-4), T(1e-6)], # residual_history
        T(2.5),                    # computation_time
        T(0.1),                    # communication_time
        nothing                    # performance_metrics
    )
    
    @testset "Basic File Generation" begin
        # Create temporary directory for test outputs
        test_dir = mktempdir()
        
        try
            # Create output manager
            manager = OutputFormat.create_output_manager(
                base_filename = "test_output",
                computation_mode = "parareal",
                grid_size = grid_size,
                output_directory = test_dir
            )
            
            # Generate parareal outputs
            generated_files = OutputFormat.generate_parareal_output!(
                manager, test_temperature, test_result, nothing
            )
            
            @test !isempty(generated_files)
            
            # Check that files were actually created
            for filename in generated_files
                filepath = joinpath(test_dir, filename)
                @test isfile(filepath)
                @test filesize(filepath) > 0
            end
            
            # Check file types
            file_extensions = [splitext(f)[2] for f in generated_files]
            @test ".csv" in file_extensions  # Should have CSV files
            @test ".dat" in file_extensions  # Should have data files
            
        finally
            # Cleanup
            rm(test_dir, recursive=true, force=true)
        end
    end
    
    @testset "Output Consistency Check" begin
        # Create temporary directory for test outputs
        test_dir = mktempdir()
        
        try
            # Create output manager
            manager = OutputFormat.create_output_manager(
                base_filename = "consistency_test",
                computation_mode = "parareal",
                grid_size = grid_size,
                output_directory = test_dir
            )
            
            # Generate outputs
            generated_files = OutputFormat.generate_parareal_output!(
                manager, test_temperature, test_result, nothing
            )
            
            # Check consistency
            is_consistent = OutputFormat.ensure_output_consistency!(manager)
            @test is_consistent
            
        finally
            # Cleanup
            rm(test_dir, recursive=true, force=true)
        end
    end
    
    @testset "Export Function" begin
        # Create temporary directory for test outputs
        test_dir = mktempdir()
        
        try
            # Test configuration
            test_config = Parareal.PararealConfig{T}(
                total_time = T(1.0),
                n_time_windows = 2,
                dt_coarse = T(0.1),
                dt_fine = T(0.01)
            )
            
            # Export parareal results
            generated_files, is_consistent = OutputFormat.export_parareal_results(
                test_temperature, test_result, test_config,
                output_directory = test_dir,
                base_filename = "export_test"
            )
            
            @test !isempty(generated_files)
            @test is_consistent
            
            # Verify files exist
            for filename in generated_files
                filepath = joinpath(test_dir, filename)
                @test isfile(filepath)
            end
            
        finally
            # Cleanup
            rm(test_dir, recursive=true, force=true)
        end
    end
end

println("Output file generation tests completed successfully!")