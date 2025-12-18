# Minimal logging system tests
using Test

# Import the Parareal module
include("../src/parareal.jl")
using .Parareal

@testset "Minimal Logging Tests" begin
    
    @testset "Basic Logging" begin
        # Test basic message logging
        @test_nowarn Parareal.log_message(0, "Test message")
        
        # Test performance logging
        @test_nowarn Parareal.log_performance_data(0, 1, 1e-6, 0.5)
    end
end

println("Minimal logging tests completed successfully!")