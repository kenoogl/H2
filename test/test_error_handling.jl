# Minimal error handling tests
using Test

# Import the Parareal module
include("../src/parareal.jl")
using .Parareal

@testset "Minimal Error Handling Tests" begin
    
    @testset "Basic MPI Operations" begin
        config = Parareal.PararealConfig{Float64}()
        manager = Parareal.PararealManager{Float64}(config)
        
        # Initialize MPI
        Parareal.initialize_mpi_parareal!(manager)
        
        # Test convergence broadcast
        result = Parareal.broadcast_convergence_status!(manager.mpi_comm, true)
        @test result == true
        
        # Test synchronization
        @test_nowarn Parareal.synchronize_processes!(manager.mpi_comm)
        
        # Test temperature field exchange (same rank)
        temperature_data = rand(Float64, 5, 5, 5)
        result = Parareal.exchange_temperature_fields!(manager.mpi_comm, temperature_data, 0)
        @test result == temperature_data
        
        # Test invalid rank
        @test_throws Exception Parareal.exchange_temperature_fields!(manager.mpi_comm, temperature_data, -1)
        
        Parareal.finalize_mpi_parareal!(manager)
    end
    
    @testset "Basic Validation" begin
        config = Parareal.PararealConfig{Float64}()
        manager = Parareal.PararealManager{Float64}(config)
        
        # Test basic initialization
        @test_nowarn Parareal.initialize_mpi_parareal!(manager)
        @test manager.is_initialized == true
        
        # Test finalization
        @test_nowarn Parareal.finalize_mpi_parareal!(manager)
        @test manager.is_initialized == false
    end
end

println("Minimal error handling tests completed successfully!")