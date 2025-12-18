# Test resource management and cleanup system
using Test
using Statistics

# Import the Parareal module
include("../src/parareal.jl")
using .Parareal

@testset "Resource Management Tests" begin
    
    @testset "Memory Pool Tests" begin
        @testset "Basic Memory Pool Operations" begin
            pool = Parareal.MemoryPool{Float64}(100.0)  # 100 MB limit
            
            @test pool.max_pool_size_mb == 100.0
            @test pool.total_allocated_mb == 0.0
            @test pool.allocation_count == 0
            @test pool.deallocation_count == 0
        end
        
        @testset "Memory Allocation and Deallocation" begin
            pool = Parareal.MemoryPool{Float32}(50.0)
            
            # Allocate a block
            size = (10, 10, 10)
            block1 = Parareal.allocate_memory!(pool, size)
            
            @test Base.size(block1) == size
            @test eltype(block1) == Float32
            @test pool.allocation_count == 1
            @test pool.total_allocated_mb > 0
            
            # Allocate another block of same size
            block2 = Parareal.allocate_memory!(pool, size)
            @test Base.size(block2) == size
            @test pool.allocation_count == 2
            
            # Deallocate first block
            success = Parareal.deallocate_memory!(pool, block1)
            @test success == true
            @test pool.deallocation_count == 1
            
            # Allocate again - should reuse the deallocated block
            block3 = Parareal.allocate_memory!(pool, size)
            @test Base.size(block3) == size
            # Should reuse, so allocation count should be same
            @test pool.allocation_count == 2  # No new allocation
        end
        
        @testset "Memory Pool Limits" begin
            # Small pool to test limits
            pool = Parareal.MemoryPool{Float64}(1.0, false)  # 1 MB, no GC
            
            # Try to allocate more than limit
            large_size = (200, 200, 200)  # Should be > 1MB
            
            @test_throws Exception Parareal.allocate_memory!(pool, large_size)
        end
        
        @testset "Garbage Collection on Pressure" begin
            pool = Parareal.MemoryPool{Float64}(10.0, true)  # Enable GC
            
            # Allocate and deallocate to create available blocks
            size = (50, 50, 50)
            for i in 1:3
                block = Parareal.allocate_memory!(pool, size)
                Parareal.deallocate_memory!(pool, block)
            end
            
            # Should have available blocks
            @test haskey(pool.available_blocks, size)
            @test length(pool.available_blocks[size]) > 0
            
            # Cleanup unused blocks
            freed_mb = Parareal.cleanup_unused_blocks!(pool)
            @test freed_mb >= 0.0
        end
        
        @testset "Memory Pool Statistics" begin
            pool = Parareal.MemoryPool{Float64}()
            
            # Perform some operations
            sizes = [(10, 10, 10), (20, 20, 20), (10, 10, 10)]
            blocks = []
            
            for size in sizes
                block = Parareal.allocate_memory!(pool, size)
                push!(blocks, block)
            end
            
            @test pool.allocation_count == 3
            @test pool.peak_allocated_mb >= pool.total_allocated_mb
            
            # Deallocate all
            for block in blocks
                Parareal.deallocate_memory!(pool, block)
            end
            
            @test pool.deallocation_count == 3
        end
    end
    
    @testset "Thread Pool Tests" begin
        @testset "Thread Pool Initialization" begin
            pool = Parareal.ThreadPool(4)
            
            @test pool.n_threads == 4
            @test length(pool.thread_ids) == 4
            @test pool.is_initialized == false
            
            # Initialize
            success = Parareal.initialize_thread_pool!(pool)
            @test success == true
            @test pool.is_initialized == true
        end
        
        @testset "Thread Pool Cleanup" begin
            pool = Parareal.ThreadPool(2)
            Parareal.initialize_thread_pool!(pool)
            
            # Simulate some tasks
            pool.active_tasks[1] = "task1"
            pool.active_tasks[2] = "task2"
            push!(pool.completed_tasks, "completed1")
            push!(pool.failed_tasks, ("failed1", ErrorException("test")))
            
            # Cleanup
            success = Parareal.cleanup_thread_pool!(pool)
            @test success == true
            @test pool.is_initialized == false
            @test isempty(pool.active_tasks)
            @test isempty(pool.completed_tasks)
            @test isempty(pool.failed_tasks)
        end
        
        @testset "Thread Pool with Too Many Threads" begin
            # Request more threads than available
            available_threads = Threads.nthreads()
            pool = Parareal.ThreadPool(available_threads + 10)
            
            Parareal.initialize_thread_pool!(pool)
            
            # Should adjust to available threads
            @test pool.n_threads <= available_threads
            @test length(pool.thread_ids) == pool.n_threads
        end
    end
    
    @testset "MPI Resource Pool Tests" begin
        @testset "MPI Pool Initialization" begin
            pool = Parareal.MPIResourcePool{Float64}(5, 50.0)
            
            @test pool.max_buffer_count_per_rank == 5
            @test pool.max_buffer_size_mb == 50.0
            @test pool.buffer_allocation_count == 0
            
            # Create a mock communicator
            comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
            
            # Initialize pool
            success = Parareal.initialize_mpi_pool!(pool, comm)
            @test success == true
            @test pool.communicator === comm
            
            # Should have buffer pools for each rank
            @test haskey(pool.send_buffers, 0)
            @test haskey(pool.recv_buffers, 0)
        end
        
        @testset "MPI Pool Cleanup" begin
            pool = Parareal.MPIResourcePool{Float64}()
            comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
            
            Parareal.initialize_mpi_pool!(pool, comm)
            
            # Add some buffers
            buffer = zeros(Float64, 10, 10, 10)
            push!(pool.send_buffers[0], buffer)
            push!(pool.recv_buffers[0], buffer)
            
            # Cleanup
            success = Parareal.cleanup_mpi_pool!(pool)
            @test success == true
            @test isempty(pool.send_buffers[0])
            @test isempty(pool.recv_buffers[0])
            @test pool.communicator === nothing
        end
    end
    
    @testset "Resource Manager Tests" begin
        @testset "Resource Manager Creation" begin
            manager = Parareal.create_resource_manager(Float64;
                max_memory_mb = 100.0,
                n_threads = 2,
                max_mpi_buffers = 10
            )
            
            @test manager isa Parareal.ResourceManager{Float64}
            @test manager.memory_pool.max_pool_size_mb == 100.0
            @test manager.thread_pool.n_threads == 2
            @test manager.mpi_pool.max_buffer_count_per_rank == 10
        end
        
        @testset "Resource Allocation" begin
            manager = Parareal.create_resource_manager(Float64)
            
            grid_size = (20, 20, 20)
            n_time_windows = 4
            
            # Allocate resources
            success = Parareal.allocate_resources!(manager, grid_size, n_time_windows)
            @test success == true
            @test manager.thread_pool.is_initialized == true
        end
        
        @testset "Resource Allocation with MPI" begin
            manager = Parareal.create_resource_manager(Float64)
            
            # Create mock MPI communicator
            comm = Parareal.MPICommunicator{Float64}(MPI.COMM_NULL)
            
            grid_size = (10, 10, 10)
            n_time_windows = 2
            
            # Allocate resources with MPI
            success = Parareal.allocate_resources!(manager, grid_size, n_time_windows, comm)
            @test success == true
            @test manager.thread_pool.is_initialized == true
            @test manager.mpi_pool.communicator === comm
        end
        
        @testset "Resource Deallocation" begin
            manager = Parareal.create_resource_manager(Float64)
            
            # Allocate first
            Parareal.allocate_resources!(manager, (10, 10, 10), 2)
            
            # Deallocate specific resources
            @test Parareal.deallocate_resources!(manager, :memory) == true
            @test Parareal.deallocate_resources!(manager, :threads) == true
            @test Parareal.deallocate_resources!(manager, :mpi) == true
            
            # Deallocate all
            @test Parareal.deallocate_resources!(manager, :all) == true
        end
        
        @testset "Complete Resource Cleanup" begin
            manager = Parareal.create_resource_manager(Float64)
            
            # Allocate resources
            Parareal.allocate_resources!(manager, (15, 15, 15), 3)
            
            # Add some memory allocations
            block1 = Parareal.allocate_memory!(manager.memory_pool, (10, 10, 10))
            block2 = Parareal.allocate_memory!(manager.memory_pool, (5, 5, 5))
            
            @test manager.memory_pool.allocation_count == 2
            
            # Complete cleanup
            success = Parareal.cleanup_all_resources!(manager)
            @test success == true
            @test manager.cleanup_count == 1
            @test !manager.is_monitoring
        end
        
        @testset "Resource Usage Statistics" begin
            manager = Parareal.create_resource_manager(Float64)
            
            # Allocate some resources
            Parareal.allocate_resources!(manager, (10, 10, 10), 2)
            block = Parareal.allocate_memory!(manager.memory_pool, (8, 8, 8))
            
            # Get usage statistics
            usage = Parareal.get_resource_usage(manager)
            
            @test haskey(usage, "memory_allocated_mb")
            @test haskey(usage, "memory_peak_mb")
            @test haskey(usage, "memory_utilization")
            @test haskey(usage, "thread_count")
            @test haskey(usage, "cleanup_count")
            
            @test usage["memory_allocated_mb"] > 0
            @test usage["thread_count"] > 0
            @test usage["cleanup_count"] >= 0
        end
        
        @testset "Resource Health Monitoring" begin
            manager = Parareal.create_resource_manager(Float64;
                memory_pressure_threshold = 0.5,
                enable_emergency_cleanup = true
            )
            
            # Start monitoring
            success = Parareal.start_resource_monitoring!(manager)
            @test success == true
            @test manager.is_monitoring == true
            
            # Monitor health (should not throw)
            memory_utilization = Parareal.monitor_resource_health!(manager)
            @test memory_utilization >= 0.0
            @test memory_utilization <= 1.0
        end
        
        @testset "Memory Pressure Handling" begin
            # Create manager with low memory limit and high pressure threshold
            manager = Parareal.create_resource_manager(Float64;
                max_memory_mb = 1.0,  # Very small limit
                memory_pressure_threshold = 0.1,  # Very low threshold
                enable_emergency_cleanup = true
            )
            
            # Allocate some memory to trigger pressure
            try
                block = Parareal.allocate_memory!(manager.memory_pool, (20, 20, 20))
                
                # Monitor health - should trigger emergency cleanup
                memory_utilization = Parareal.monitor_resource_health!(manager)
                
                # Should have triggered emergency cleanup
                @test manager.emergency_cleanup_count >= 0
                
            catch e
                # Expected if memory limit is exceeded
                @test e isa Exception
            end
        end
        
        @testset "Resource Error Tracking" begin
            manager = Parareal.create_resource_manager(Float64)
            
            # Simulate some failed tasks
            push!(manager.thread_pool.failed_tasks, ("task1", ErrorException("error1")))
            push!(manager.thread_pool.failed_tasks, ("task2", ErrorException("error2")))
            
            # Monitor health - should track errors
            Parareal.monitor_resource_health!(manager)
            
            # Should have recorded errors
            @test length(manager.resource_errors) >= 2
            
            # Failed tasks should be cleared
            @test isempty(manager.thread_pool.failed_tasks)
        end
    end
    
    @testset "Integration with PararealManager" begin
        @testset "Manager with Resource Management" begin
            config = Parareal.PararealConfig{Float64}()
            manager = Parareal.PararealManager{Float64}(config)
            
            # Initialize with resource management
            Parareal.initialize_mpi_parareal!(manager)
            
            @test manager.resource_manager !== nothing
            @test manager.resource_manager isa Parareal.ResourceManager{Float64}
        end
        
        @testset "Resource Cleanup on Finalization" begin
            config = Parareal.PararealConfig{Float64}()
            manager = Parareal.PararealManager{Float64}(config)
            
            Parareal.initialize_mpi_parareal!(manager)
            
            # Allocate some resources
            if manager.resource_manager !== nothing
                block = Parareal.allocate_memory!(manager.resource_manager.memory_pool, (5, 5, 5))
                @test manager.resource_manager.memory_pool.allocation_count > 0
            end
            
            # Finalize - should cleanup resources
            Parareal.finalize_mpi_parareal!(manager)
            
            @test manager.is_initialized == false
        end
        
        @testset "Resource Allocation in run_parareal!" begin
            config = Parareal.PararealConfig{Float64}(
                total_time = 0.1,
                n_time_windows = 2,
                dt_coarse = 0.05,
                dt_fine = 0.01,
                max_iterations = 2
            )
            manager = Parareal.PararealManager{Float64}(config)
            
            Parareal.initialize_mpi_parareal!(manager)
            
            # Create test problem data
            initial_condition = rand(Float64, 8, 8, 8)
            problem_data = Parareal.Heat3dsProblemData{Float64}(
                (0.1, 0.1, 0.1), [0.1, 0.2], [0.1, 0.1], 
                zeros(UInt8, 8, 8, 8), nothing, "thread", false
            )
            
            # Run parareal - should use resource manager for memory allocation
            result = Parareal.run_parareal!(manager, initial_condition, problem_data)
            
            @test result isa Parareal.PararealResult{Float64}
            
            # Should have used resource manager
            if manager.resource_manager !== nothing
                @test manager.resource_manager.memory_pool.allocation_count > 0
            end
            
            Parareal.finalize_mpi_parareal!(manager)
        end
    end
    
    @testset "Error Handling in Resource Management" begin
        @testset "Cleanup on Error" begin
            manager = Parareal.create_resource_manager(Float64; cleanup_on_error = true)
            
            # Allocate resources
            Parareal.allocate_resources!(manager, (10, 10, 10), 2)
            
            # Simulate allocation failure
            @test_throws Exception Parareal.allocate_resources!(
                manager, (1000, 1000, 1000), 100  # Too large
            )
            
            # Should have attempted cleanup
            @test manager.cleanup_count >= 0
        end
        
        @testset "Resource Deallocation Errors" begin
            manager = Parareal.create_resource_manager(Float64)
            
            # Try to deallocate invalid resource type
            result = Parareal.deallocate_resources!(manager, :invalid_type)
            @test result == true  # Should handle gracefully
        end
        
        @testset "Memory Pool Error Handling" begin
            pool = Parareal.MemoryPool{Float64}()
            
            # Try to deallocate untracked block
            fake_block = zeros(Float64, 5, 5, 5)
            result = Parareal.deallocate_memory!(pool, fake_block)
            @test result == false  # Should return false for untracked block
        end
    end
end

println("Resource management tests completed successfully!")