# Simple test for output format functionality
using Test

# Import the modules we need to test
include("../src/output_format.jl")
using .OutputFormat

@testset "Simple Output Format Tests" begin
    
    @testset "OutputManager Creation" begin
        manager = OutputFormat.create_output_manager(
            base_filename = "test",
            computation_mode = "parareal",
            grid_size = (4, 4, 4)
        )
        
        @test manager.config.base_filename == "test"
        @test manager.metadata.computation_mode == "parareal"
        @test manager.metadata.grid_size == (4, 4, 4)
    end
    
    @testset "OutputConfiguration Creation" begin
        config = OutputFormat.OutputConfiguration(
            base_filename = "test_config",
            enable_log_output = true,
            enable_convergence_output = false
        )
        
        @test config.base_filename == "test_config"
        @test config.enable_log_output == true
        @test config.enable_convergence_output == false
    end
    
    @testset "Metadata Creation" begin
        # Create a simple mock config
        mock_config = (n_time_windows = 2, n_mpi_processes = 1)
        mock_result = (iterations = 3, converged = true, computation_time = 1.5)
        
        metadata = OutputFormat.create_output_metadata(mock_config, mock_result)
        
        @test metadata.n_time_windows == 2
        @test metadata.n_mpi_processes == 1
        @test metadata.parareal_iterations == 3
        @test metadata.convergence_achieved == true
        @test metadata.computation_time == 1.5
    end
    
    @testset "Filename Generation" begin
        manager = OutputFormat.create_output_manager(
            base_filename = "test_filename",
            computation_mode = "parareal"
        )
        
        log_filename = OutputFormat.create_output_filename(manager, "log", "txt")
        csv_filename = OutputFormat.create_output_filename(manager, "convergence", "csv")
        
        @test occursin("test_filename", log_filename)
        @test occursin("parareal", log_filename)
        @test endswith(log_filename, ".txt")
        @test endswith(csv_filename, ".csv")
    end
end

println("Simple output format tests completed successfully!")