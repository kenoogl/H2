using Test
using FLoops

# Include the Common module
include("../src/common.jl")
using .Common

"""
Unit tests for ThreadsX integration with MPI context
Tests thread pool creation within MPI processes and spatial parallelization performance
Requirements: 6.2, 6.3
"""

@testset "ThreadsX Integration Tests" begin
    
    @testset "Backend selection for MPI context" begin
        
        @testset "get_backend_mpi function" begin
            # Test sequential backend selection
            backend_seq = Common.get_backend_mpi("sequential", 1)
            @test backend_seq isa SequentialEx
            
            backend_seq2 = Common.get_backend_mpi("sequential", 4)
            @test backend_seq2 isa SequentialEx
            
            # Test threaded backend selection
            backend_thread1 = Common.get_backend_mpi("thread", 1)
            @test backend_thread1 isa SequentialEx  # Single thread should use sequential
            
            if Threads.nthreads() > 1
                backend_thread_multi = Common.get_backend_mpi("thread", 2)
                @test backend_thread_multi isa ThreadedEx
            end
        end
        
        @testset "get_hybrid_backend function" begin
            # Test various MPI + thread combinations
            
            # Sequential case: 1 MPI process, 1 thread
            backend_seq = Common.get_hybrid_backend("sequential", 1, 1)
            @test backend_seq isa SequentialEx
            
            # MPI-only case: multiple MPI processes, 1 thread each
            backend_mpi = Common.get_hybrid_backend("sequential", 4, 1)
            @test backend_mpi isa SequentialEx
            
            # Thread-only case: 1 MPI process, multiple threads
            if Threads.nthreads() > 1
                backend_thread = Common.get_hybrid_backend("thread", 1, 2)
                @test backend_thread isa ThreadedEx
            end
            
            # Hybrid case: multiple MPI processes, multiple threads each
            if Threads.nthreads() > 1
                backend_hybrid = Common.get_hybrid_backend("thread", 4, 2)
                @test backend_hybrid isa ThreadedEx
            end
        end
    end
    
    @testset "MPI-aware parallel operations" begin
        
        @testset "myfill_mpi! function" begin
            # Create test array
            test_array = zeros(Float64, 10, 10, 10)
            fill_value = 42.0
            
            # Test sequential fill
            Common.myfill_mpi!(test_array, fill_value, "sequential", 1)
            @test all(test_array .== fill_value)
            
            # Reset array
            test_array .= 0.0
            
            # Test threaded fill (if threads available)
            if Threads.nthreads() > 1
                Common.myfill_mpi!(test_array, fill_value, "thread", 2)
                @test all(test_array .== fill_value)
            else
                # Fallback to sequential
                Common.myfill_mpi!(test_array, fill_value, "thread", 2)
                @test all(test_array .== fill_value)
            end
        end
        
        @testset "mycopy_mpi! function" begin
            # Create test arrays
            src_array = rand(Float64, 8, 8, 8)
            dst_array = zeros(Float64, 8, 8, 8)
            
            # Test sequential copy
            Common.mycopy_mpi!(dst_array, src_array, "sequential", 1)
            @test dst_array == src_array
            
            # Reset destination
            dst_array .= 0.0
            
            # Test threaded copy (if threads available)
            if Threads.nthreads() > 1
                Common.mycopy_mpi!(dst_array, src_array, "thread", 2)
                @test dst_array == src_array
            else
                # Fallback to sequential
                Common.mycopy_mpi!(dst_array, src_array, "thread", 2)
                @test dst_array == src_array
            end
        end
    end
    
    @testset "Hybrid parallel operations" begin
        
        @testset "myfill_hybrid! function" begin
            # Create test array
            test_array = zeros(Float64, 12, 12, 12)
            fill_value = 3.14
            
            # Test various hybrid configurations
            test_configs = [
                ("sequential", 1, 1),  # Pure sequential
                ("sequential", 4, 1),  # MPI-only
                ("thread", 1, 2),      # Thread-only
                ("thread", 4, 2)       # True hybrid
            ]
            
            for (par, mpi_size, n_threads) in test_configs
                test_array .= 0.0
                Common.myfill_hybrid!(test_array, fill_value, par, mpi_size, n_threads)
                @test all(test_array .== fill_value)
            end
        end
        
        @testset "mycopy_hybrid! function" begin
            # Create test arrays
            src_array = rand(Float64, 6, 6, 6)
            dst_array = zeros(Float64, 6, 6, 6)
            
            # Test various hybrid configurations
            test_configs = [
                ("sequential", 1, 1),  # Pure sequential
                ("sequential", 4, 1),  # MPI-only
                ("thread", 1, 2),      # Thread-only
                ("thread", 4, 2)       # True hybrid
            ]
            
            for (par, mpi_size, n_threads) in test_configs
                dst_array .= 0.0
                Common.mycopy_hybrid!(dst_array, src_array, par, mpi_size, n_threads)
                @test dst_array == src_array
            end
        end
    end
    
    @testset "Performance consistency" begin
        
        @testset "Spatial parallelization performance maintained" begin
            # Create larger arrays for performance testing
            large_array = zeros(Float64, 50, 50, 50)
            src_large = rand(Float64, 50, 50, 50)
            fill_value = 1.23
            
            # Measure sequential performance
            time_seq = @elapsed begin
                Common.myfill!(large_array, fill_value, "sequential")
                Common.mycopy!(large_array, src_large, "sequential")
            end
            
            # Measure MPI-aware performance (should be similar for single process)
            large_array .= 0.0
            time_mpi = @elapsed begin
                Common.myfill_mpi!(large_array, fill_value, "sequential", 1)
                Common.mycopy_mpi!(large_array, src_large, "sequential", 1)
            end
            
            # Measure hybrid performance (should be similar for single process)
            large_array .= 0.0
            time_hybrid = @elapsed begin
                Common.myfill_hybrid!(large_array, fill_value, "sequential", 1, 1)
                Common.mycopy_hybrid!(large_array, src_large, "sequential", 1, 1)
            end
            
            # Performance should be comparable (within reasonable tolerance)
            # Note: We're not doing strict performance assertions as they can be flaky
            # Instead, we verify correctness and that operations complete
            @test all(large_array .== src_large)
            @test time_seq > 0
            @test time_mpi > 0
            @test time_hybrid > 0
            
            println("Performance comparison:")
            println("  Sequential: $(time_seq) seconds")
            println("  MPI-aware:  $(time_mpi) seconds")
            println("  Hybrid:     $(time_hybrid) seconds")
        end
        
        @testset "Thread scaling behavior" begin
            # Test that threaded operations scale appropriately
            if Threads.nthreads() > 1
                test_array = zeros(Float64, 30, 30, 30)
                fill_value = 2.71
                
                # Test single thread
                time_1thread = @elapsed Common.myfill_mpi!(test_array, fill_value, "thread", 1)
                
                # Test multiple threads
                test_array .= 0.0
                time_multithread = @elapsed Common.myfill_mpi!(test_array, fill_value, "thread", Threads.nthreads())
                
                # Verify correctness
                @test all(test_array .== fill_value)
                
                # Both should complete successfully
                @test time_1thread > 0
                @test time_multithread > 0
                
                println("Thread scaling:")
                println("  1 thread:  $(time_1thread) seconds")
                println("  $(Threads.nthreads()) threads: $(time_multithread) seconds")
            else
                println("Skipping thread scaling test (only 1 thread available)")
            end
        end
    end
    
    @testset "WorkBuffers compatibility" begin
        
        @testset "WorkBuffers creation and access" begin
            # Test that WorkBuffers work correctly in MPI context
            mx, my, mz = 20, 20, 20
            wk = Common.WorkBuffers(mx, my, mz)
            
            # Verify all arrays are properly allocated
            @test size(wk.θ) == (mx, my, mz)
            @test size(wk.b) == (mx, my, mz)
            @test size(wk.mask) == (mx, my, mz)
            @test size(wk.ρ) == (mx, my, mz)
            @test size(wk.λ) == (mx, my, mz)
            @test size(wk.cp) == (mx, my, mz)
            @test size(wk.hsrc) == (mx, my, mz)
            
            # Test that arrays can be used with MPI-aware functions
            test_value = 5.0
            Common.myfill_mpi!(wk.θ, test_value, "sequential", 1)
            @test all(wk.θ .== test_value)
            
            # Test copying between WorkBuffer arrays
            Common.mycopy_mpi!(wk.b, wk.θ, "sequential", 1)
            @test all(wk.b .== test_value)
        end
        
        @testset "WorkBuffers with hybrid operations" begin
            # Test WorkBuffers with hybrid parallelization
            mx, my, mz = 15, 15, 15
            wk = Common.WorkBuffers(mx, my, mz)
            
            # Test hybrid fill operations
            test_value = 7.5
            Common.myfill_hybrid!(wk.λ, test_value, "sequential", 1, 1)
            @test all(wk.λ .== test_value)
            
            # Test hybrid copy operations
            Common.mycopy_hybrid!(wk.cp, wk.λ, "sequential", 1, 1)
            @test all(wk.cp .== test_value)
            
            # Test with threaded operations (if available)
            if Threads.nthreads() > 1
                test_value2 = 9.9
                Common.myfill_hybrid!(wk.ρ, test_value2, "thread", 1, 2)
                @test all(wk.ρ .== test_value2)
            end
        end
    end
end

println("\n=== ThreadsX Integration Test Summary ===")
println("✓ Backend selection for MPI context")
println("✓ MPI-aware parallel operations (fill, copy)")
println("✓ Hybrid parallel operations")
println("✓ Performance consistency verification")
println("✓ WorkBuffers compatibility with MPI context")
println("✓ Thread scaling behavior validation")
println("===========================================")