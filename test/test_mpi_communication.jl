# Test MPI communication functionality for Parareal
using Test
using MPI

# Add src to load path for testing
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

include("../src/parareal.jl")
using .Parareal

@testset "MPI Communication Tests" begin
    
    @testset "MPICommunicator Construction" begin
        # Test with null communicator (for testing)
        comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
        
        @test comm.rank == 0
        @test comm.size == 1
        @test isempty(comm.send_buffers)
        @test isempty(comm.recv_buffers)
        @test isempty(comm.requests)
    end
    
    @testset "Communication Buffer Initialization" begin
        comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
        grid_size = (10, 10, 5)
        
        Parareal.initialize_communication_buffers!(comm, grid_size)
        
        @test length(comm.send_buffers) == comm.size
        @test length(comm.recv_buffers) == comm.size
        @test size(comm.send_buffers[1]) == grid_size
        @test size(comm.recv_buffers[1]) == grid_size
    end
    
    @testset "Temperature Field Exchange (Single Process)" begin
        comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
        
        # Create test temperature field
        temperature_data = rand(Float64, 8, 8, 4)
        
        # Test exchange with same process (should return same data)
        result = Parareal.exchange_temperature_fields!(comm, temperature_data, 0)
        @test result == temperature_data
        
        # Test invalid target rank
        @test_throws ErrorException Parareal.exchange_temperature_fields!(comm, temperature_data, 1)
    end
    
    @testset "Convergence Status Broadcast" begin
        comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
        
        # Test convergence broadcast (single process)
        @test Parareal.broadcast_convergence_status!(comm, true) == true
        @test Parareal.broadcast_convergence_status!(comm, false) == false
    end
    
    @testset "Performance Metrics Gathering" begin
        comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
        
        # Create test metrics
        local_metrics = Dict{String, Float64}(
            "computation_time" => 1.5,
            "communication_time" => 0.3,
            "memory_usage" => 100.0
        )
        
        # Test metrics gathering (single process)
        all_metrics = Parareal.gather_performance_metrics!(comm, local_metrics)
        
        @test haskey(all_metrics, "process_0")
        @test all_metrics["process_0"]["computation_time"] == 1.5
        @test all_metrics["process_0"]["communication_time"] == 0.3
        @test all_metrics["process_0"]["memory_usage"] == 100.0
    end
    
    @testset "Process Synchronization" begin
        comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
        
        # Test synchronization (should not throw in single process)
        @test_nowarn Parareal.synchronize_processes!(comm)
    end
    
    @testset "Data Integrity Checks" begin
        # Test checksum computation
        data1 = ones(Float64, 4, 4, 2)
        checksum1 = Parareal.compute_data_checksum(data1)
        @test checksum1 == 32.0  # 4*4*2 = 32 ones
        
        data2 = zeros(Float64, 3, 3, 3)
        checksum2 = Parareal.compute_data_checksum(data2)
        @test checksum2 == 0.0
        
        # Test data validation
        @test Parareal.validate_temperature_data(data1, 32.0)
        @test !Parareal.validate_temperature_data(data1, 30.0)
        @test Parareal.validate_temperature_data(data2, 0.0)
        
        # Test with tolerance
        @test Parareal.validate_temperature_data(data1, 32.0001, 1e-3)
        @test !Parareal.validate_temperature_data(data1, 32.1, 1e-3)
    end
end

# Property-based test for MPI Communication Reliability
@testset "Property Test: MPI Communication Reliability" begin
    # Property 9: For any temperature field exchange between MPI processes,
    # the data should be transmitted without corruption and within reasonable time bounds
    
    function test_communication_reliability_property(grid_size, data_range)
        comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
        
        # Generate random temperature field
        temperature_data = rand(Float64, grid_size...) * data_range
        
        # Compute original checksum
        original_checksum = Parareal.compute_data_checksum(temperature_data)
        
        # Test self-communication (in single process environment)
        start_time = time()
        result = Parareal.exchange_temperature_fields!(comm, temperature_data, 0)
        end_time = time()
        
        # Property 1: Data integrity preserved
        result_checksum = Parareal.compute_data_checksum(result)
        @test abs(result_checksum - original_checksum) < 1e-12
        
        # Property 2: Data content preserved
        @test result == temperature_data
        
        # Property 3: Communication completes in reasonable time (< 1 second for test)
        communication_time = end_time - start_time
        @test communication_time < 1.0
        
        # Property 4: Data validation works correctly
        @test Parareal.validate_temperature_data(result, original_checksum)
        
        return true
    end
    
    # Test with various grid sizes and data ranges
    test_cases = [
        ((4, 4, 2), 100.0),
        ((8, 8, 4), 500.0),
        ((16, 16, 8), 1000.0),
        ((2, 2, 1), 1.0),
        ((10, 5, 3), 273.15)  # Typical temperature range
    ]
    
    for (grid_size, data_range) in test_cases
        @test test_communication_reliability_property(grid_size, data_range)
    end
end

@testset "Error Handling Tests" begin
    @testset "Invalid Communication Parameters" begin
        comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
        temperature_data = rand(Float64, 4, 4, 2)
        
        # Test invalid target ranks
        @test_throws ErrorException Parareal.exchange_temperature_fields!(comm, temperature_data, -1)
        @test_throws ErrorException Parareal.exchange_temperature_fields!(comm, temperature_data, 2)
    end
    
    @testset "Buffer Management" begin
        comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
        
        # Test multiple buffer initializations
        grid_size1 = (4, 4, 2)
        grid_size2 = (8, 8, 4)
        
        Parareal.initialize_communication_buffers!(comm, grid_size1)
        @test size(comm.send_buffers[1]) == grid_size1
        
        # Reinitialize with different size
        Parareal.initialize_communication_buffers!(comm, grid_size2)
        @test size(comm.send_buffers[1]) == grid_size2
        @test length(comm.send_buffers) == comm.size
    end
end

println("MPI communication tests completed successfully!")