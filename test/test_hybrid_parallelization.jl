using Test
using MPI
using Printf
using FLoops

# Include the Parareal module
include("../src/parareal.jl")
using .Parareal

"""
Property-based test for hybrid parallelization activation
Property 2: Hybrid Parallelization Activation
Validates: Requirements 1.3, 1.4, 1.5
"""

@testset "Hybrid Parallelization Property Tests" begin
    
    @testset "Property 2: Hybrid Parallelization Activation" begin
        
        @testset "Thread pool initialization consistency" begin
            # Test that thread pool initialization is consistent across different configurations
            
            # Test with different thread counts
            thread_counts = [1, 2, 4, min(8, Threads.nthreads())]
            
            for n_threads in thread_counts
                # Create mock MPI communicator for testing
                mock_comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
                
                # Create hybrid coordinator
                coordinator = Parareal.HybridCoordinator{Float64}(mock_comm, n_threads)
                
                # Test initial state
                @test !coordinator.is_thread_pool_initialized
                @test coordinator.thread_pool_size == n_threads
                
                # Initialize thread pool
                Parareal.initialize_thread_pool!(coordinator)
                
                # Test post-initialization state
                @test coordinator.is_thread_pool_initialized
                @test coordinator.thread_pool_size <= Threads.nthreads()  # Should not exceed available threads
                @test coordinator.thread_pool_size > 0
                
                # Test that re-initialization is safe
                Parareal.initialize_thread_pool!(coordinator)
                @test coordinator.is_thread_pool_initialized
            end
        end
        
        @testset "Hybrid configuration validation" begin
            # Test that hybrid configuration validation works correctly
            
            # Create mock MPI communicator
            mock_comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
            
            # Test valid configuration
            coordinator = Parareal.HybridCoordinator{Float64}(mock_comm, 2)
            Parareal.initialize_thread_pool!(coordinator)
            
            is_valid, message = Parareal.validate_hybrid_configuration(coordinator)
            @test is_valid
            @test message == "Hybrid configuration is valid"
            
            # Test invalid configuration - uninitialized thread pool
            coordinator_uninit = Parareal.HybridCoordinator{Float64}(mock_comm, 2)
            is_valid, message = Parareal.validate_hybrid_configuration(coordinator_uninit)
            @test !is_valid
            @test occursin("not initialized", message)
            
            # Test invalid configuration - zero threads
            coordinator_zero = Parareal.HybridCoordinator{Float64}(mock_comm, 0)
            is_valid, message = Parareal.validate_hybrid_configuration(coordinator_zero)
            @test !is_valid
            @test occursin("Invalid thread pool size", message)
        end
        
        @testset "Backend selection consistency" begin
            # Test that backend selection is consistent with thread configuration
            
            mock_comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
            
            # Test single thread - should return SequentialEx
            coordinator_seq = Parareal.HybridCoordinator{Float64}(mock_comm, 1)
            Parareal.initialize_thread_pool!(coordinator_seq)
            
            backend_seq = Parareal.get_hybrid_backend(coordinator_seq)
            @test backend_seq isa SequentialEx
            
            # Test multiple threads - should return ThreadedEx (if available)
            if Threads.nthreads() > 1
                coordinator_par = Parareal.HybridCoordinator{Float64}(mock_comm, 2)
                Parareal.initialize_thread_pool!(coordinator_par)
                
                backend_par = Parareal.get_hybrid_backend(coordinator_par)
                @test backend_par isa ThreadedEx
            end
        end
        
        @testset "Hybrid statistics accuracy" begin
            # Test that hybrid statistics are accurate
            
            mock_comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
            n_threads = 2
            
            coordinator = Parareal.HybridCoordinator{Float64}(mock_comm, n_threads)
            Parareal.initialize_thread_pool!(coordinator)
            
            stats = Parareal.get_hybrid_statistics(coordinator)
            
            # Verify all expected keys are present
            expected_keys = ["mpi_processes", "threads_per_process", "total_parallel_units", 
                           "current_mpi_rank", "thread_pool_initialized", "available_julia_threads"]
            
            for key in expected_keys
                @test haskey(stats, key)
            end
            
            # Verify values are consistent
            @test stats["mpi_processes"] == mock_comm.size
            @test stats["threads_per_process"] == coordinator.thread_pool_size
            @test stats["total_parallel_units"] == stats["mpi_processes"] * stats["threads_per_process"]
            @test stats["current_mpi_rank"] == mock_comm.rank
            @test stats["thread_pool_initialized"] == true
            @test stats["available_julia_threads"] == Threads.nthreads()
        end
        
        @testset "Hybrid execution coordination" begin
            # Test that hybrid execution coordination works correctly
            
            mock_comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
            coordinator = Parareal.HybridCoordinator{Float64}(mock_comm, 2)
            Parareal.initialize_thread_pool!(coordinator)
            
            # Define a simple work function for testing
            function test_work_function(data, backend)
                # Simple computation that can be parallelized
                result = sum(data .^ 2)
                return result
            end
            
            # Test data
            test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
            expected_result = sum(test_data .^ 2)
            
            # Execute work function through coordinator
            result = Parareal.coordinate_hybrid_execution!(coordinator, test_work_function, test_data)
            
            # Verify result is correct
            @test result ≈ expected_result
        end
        
        @testset "Error handling for uninitialized coordinator" begin
            # Test that appropriate errors are thrown for uninitialized coordinators
            
            mock_comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
            coordinator = Parareal.HybridCoordinator{Float64}(mock_comm, 2)
            # Note: Not calling initialize_thread_pool!
            
            # Test that get_hybrid_backend throws error
            @test_throws ErrorException Parareal.get_hybrid_backend(coordinator)
            
            # Test that coordinate_hybrid_execution throws error
            function dummy_work(data, backend)
                return data
            end
            
            @test_throws ErrorException Parareal.coordinate_hybrid_execution!(coordinator, dummy_work, [1, 2, 3])
        end
    end
    
    @testset "Integration with PararealManager" begin
        
        @testset "Hybrid coordinator creation in run_parareal!" begin
            # Test that run_parareal! properly creates and initializes hybrid coordinator
            
            # Create a basic parareal configuration
            config = Parareal.PararealConfig{Float64}(
                total_time = 1.0,
                n_time_windows = 4,
                dt_coarse = 0.1,
                dt_fine = 0.01,
                n_threads_per_process = 2
            )
            
            manager = Parareal.PararealManager{Float64}(config)
            
            # Initialize MPI (mock)
            manager.mpi_comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
            Parareal.create_time_windows!(manager)
            manager.is_initialized = true
            
            # Test that run_parareal! executes without error
            # (This is a placeholder implementation, so we just test it doesn't crash)
            try
                Parareal.run_parareal!(manager, nothing)
                @test true  # If we get here, no exception was thrown
            catch e
                @test false  # run_parareal! threw an exception
                println("run_parareal! threw an exception: $e")
            end
        end
        
        @testset "Thread count validation in manager" begin
            # Test that thread count is properly validated in manager configuration
            
            # Test with valid thread count
            config_valid = Parareal.PararealConfig{Float64}(n_threads_per_process = 2)
            manager_valid = Parareal.PararealManager{Float64}(config_valid)
            @test manager_valid.config.n_threads_per_process == 2
            
            # Test with thread count exceeding available threads
            excessive_threads = Threads.nthreads() + 10
            config_excessive = Parareal.PararealConfig{Float64}(n_threads_per_process = excessive_threads)
            manager_excessive = Parareal.PararealManager{Float64}(config_excessive)
            
            # The configuration should accept the value (validation happens at runtime)
            @test manager_excessive.config.n_threads_per_process == excessive_threads
        end
    end
end

# Property verification summary
println("\n=== Property 2: Hybrid Parallelization Activation - Verification Summary ===")
println("✓ Thread pool initialization consistency across different configurations")
println("✓ Hybrid configuration validation for various scenarios")
println("✓ Backend selection consistency with thread configuration")
println("✓ Hybrid statistics accuracy and completeness")
println("✓ Hybrid execution coordination functionality")
println("✓ Error handling for uninitialized coordinators")
println("✓ Integration with PararealManager")
println("✓ Thread count validation in manager configuration")
println("================================================================================")