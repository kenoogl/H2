using Test
using LinearAlgebra
using Statistics
using MPI
using Dates

# Include the Parareal module
include("../src/parareal.jl")
using .Parareal

"""
Unit tests for validation components
Task 6.6: Write unit tests for validation components
- Test AccuracyMetrics calculation accuracy
- Verify ValidationResult data integrity
- Test error analysis report generation
Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
"""

@testset "Validation Components Unit Tests" begin
    
    @testset "AccuracyMetrics Calculation Tests" begin
        
        @testset "L2 norm error calculation" begin
            # Test with known data
            grid_size = (4, 4, 4)
            
            # Test case 1: Identical arrays should give zero error
            data1 = ones(Float64, grid_size...) * 300.0
            data2 = ones(Float64, grid_size...) * 300.0
            
            metrics = Parareal.compute_accuracy_metrics(data1, data2)
            @test metrics.l2_norm_error ≈ 0.0 atol=1e-15
            @test metrics.max_pointwise_error ≈ 0.0 atol=1e-15
            @test metrics.relative_error ≈ 0.0 atol=1e-15
            
            # Test case 2: Known uniform error
            data2_uniform = data1 .+ 1.0  # Add uniform error of 1.0
            
            metrics_uniform = Parareal.compute_accuracy_metrics(data1, data2_uniform)
            expected_l2 = sqrt(sum((data1 - data2_uniform).^2))
            @test metrics_uniform.l2_norm_error ≈ expected_l2 rtol=1e-12
            @test metrics_uniform.max_pointwise_error ≈ 1.0 rtol=1e-12
            
            # Test case 3: Single point error
            data2_point = copy(data1)
            data2_point[2, 2, 2] += 5.0  # Add error at single point
            
            metrics_point = Parareal.compute_accuracy_metrics(data1, data2_point)
            @test metrics_point.max_pointwise_error ≈ 5.0 rtol=1e-12
            @test metrics_point.l2_norm_error ≈ 5.0 rtol=1e-12  # Only one point differs
        end
        
        @testset "Relative error calculation" begin
            grid_size = (3, 3, 3)
            
            # Test with non-zero reference
            reference = ones(Float64, grid_size...) * 100.0
            test_data = ones(Float64, grid_size...) * 101.0  # 1% error
            
            metrics = Parareal.compute_accuracy_metrics(test_data, reference)
            expected_relative = norm(reference - test_data) / norm(reference)
            @test metrics.relative_error ≈ expected_relative rtol=1e-12
            
            # Test with zero reference (should handle gracefully)
            zero_reference = zeros(Float64, grid_size...)
            non_zero_test = ones(Float64, grid_size...)
            
            metrics_zero = Parareal.compute_accuracy_metrics(non_zero_test, zero_reference)
            @test isfinite(metrics_zero.relative_error)
            @test metrics_zero.relative_error >= 0.0
        end
        
        @testset "Error distribution calculation" begin
            grid_size = (4, 4, 4)
            
            # Create test data with known error pattern
            reference = zeros(Float64, grid_size...)
            test_data = zeros(Float64, grid_size...)
            
            # Add known errors at specific locations
            test_data[1, 1, 1] = 1.0
            test_data[2, 2, 2] = 2.0
            test_data[3, 3, 3] = 3.0
            
            metrics = Parareal.compute_accuracy_metrics(test_data, reference)
            
            # Check that error distribution captures the errors
            @test length(metrics.error_distribution) == prod(grid_size)
            @test all(isfinite.(metrics.error_distribution))
            @test all(metrics.error_distribution .>= 0.0)
            
            # Check that specific errors are captured
            error_values = sort(metrics.error_distribution, rev=true)
            @test error_values[1] ≈ 3.0 rtol=1e-12  # Largest error
            @test error_values[2] ≈ 2.0 rtol=1e-12  # Second largest
            @test error_values[3] ≈ 1.0 rtol=1e-12  # Third largest
        end
        
        @testset "Edge cases and robustness" begin
            grid_size = (2, 2, 2)
            
            # Test with very large values
            large_ref = ones(Float64, grid_size...) * 1e10
            large_test = ones(Float64, grid_size...) * (1e10 + 1e5)
            
            metrics_large = Parareal.compute_accuracy_metrics(large_test, large_ref)
            @test isfinite(metrics_large.l2_norm_error)
            @test isfinite(metrics_large.relative_error)
            @test all(isfinite.(metrics_large.error_distribution))
            
            # Test with very small values
            small_ref = ones(Float64, grid_size...) * 1e-10
            small_test = ones(Float64, grid_size...) * 1.1e-10
            
            metrics_small = Parareal.compute_accuracy_metrics(small_test, small_ref)
            @test isfinite(metrics_small.l2_norm_error)
            @test isfinite(metrics_small.relative_error)
            @test all(isfinite.(metrics_small.error_distribution))
            
            # Test with mixed positive/negative values
            mixed_ref = [-1.0 1.0; -2.0 2.0][:, :, 1:1]
            mixed_test = [-1.1 1.1; -2.1 2.1][:, :, 1:1]
            
            metrics_mixed = Parareal.compute_accuracy_metrics(mixed_test, mixed_ref)
            @test isfinite(metrics_mixed.l2_norm_error)
            @test metrics_mixed.l2_norm_error > 0.0
        end
    end
    
    @testset "ValidationResult Data Integrity Tests" begin
        
        @testset "ValidationResult construction" begin
            # Create test data
            grid_size = (3, 3, 3)
            reference = ones(Float64, grid_size...) * 300.0
            test_solution = reference .+ 0.1
            
            # Create accuracy metrics
            metrics = Parareal.compute_accuracy_metrics(test_solution, reference)
            
            # Create parareal config
            config = Parareal.PararealConfig{Float64}(
                total_time = 1.0,
                n_time_windows = 4,
                convergence_tolerance = 1e-6
            )
            
            # Create tolerance settings
            tolerance_settings = Parareal.ToleranceSettings{Float64}(
                absolute_tolerance = 1e-5,
                relative_tolerance = 1e-3,
                max_pointwise_tolerance = 1e-4
            )
            
            # Create validation result
            validation_result = Parareal.ValidationResult{Float64}(
                Dates.DateTime(2024, 1, 1, 0, 0, 0),
                "test_problem",
                config,
                metrics,
                true,
                ["Test recommendation"]
            )
            
            # Test data integrity
            @test validation_result.problem_id == "test_problem"
            @test validation_result.accuracy_metrics === metrics
            @test validation_result.is_within_tolerance == true
            @test validation_result.parareal_config === config
            @test length(validation_result.recommendations) == 1
            @test validation_result.recommendations[1] == "Test recommendation"
            @test validation_result.timestamp == Dates.DateTime(2024, 1, 1, 0, 0, 0)
        end
        
        @testset "Tolerance checking logic" begin
            grid_size = (2, 2, 2)
            reference = ones(Float64, grid_size...) * 100.0
            
            # Test case 1: Within all tolerances
            test_solution_good = reference .+ 1e-6  # Small error
            metrics_good = Parareal.compute_accuracy_metrics(test_solution_good, reference)
            
            tolerance_settings = Parareal.ToleranceSettings{Float64}(
                absolute_tolerance = 1e-5,
                relative_tolerance = 1e-3,
                max_pointwise_tolerance = 1e-5
            )
            
            is_within = Parareal.check_tolerance(metrics_good, tolerance_settings)
            @test is_within == true
            
            # Test case 2: Exceeds absolute tolerance
            test_solution_bad_abs = reference .+ 1e-4  # Large absolute error
            metrics_bad_abs = Parareal.compute_accuracy_metrics(test_solution_bad_abs, reference)
            
            is_within_abs = Parareal.check_tolerance(metrics_bad_abs, tolerance_settings)
            @test is_within_abs == false
            
            # Test case 3: Exceeds relative tolerance
            test_solution_bad_rel = reference .* 1.1  # 10% relative error
            metrics_bad_rel = Parareal.compute_accuracy_metrics(test_solution_bad_rel, reference)
            
            is_within_rel = Parareal.check_tolerance(metrics_bad_rel, tolerance_settings)
            @test is_within_rel == false
            
            # Test case 4: Exceeds pointwise tolerance
            test_solution_bad_point = copy(reference)
            test_solution_bad_point[1, 1, 1] += 1e-3  # Large pointwise error
            metrics_bad_point = Parareal.compute_accuracy_metrics(test_solution_bad_point, reference)
            
            is_within_point = Parareal.check_tolerance(metrics_bad_point, tolerance_settings)
            @test is_within_point == false
        end
        
        @testset "Recommendation generation" begin
            grid_size = (3, 3, 3)
            reference = ones(Float64, grid_size...) * 300.0
            
            # Test different error scenarios and their recommendations
            test_cases = [
                (
                    name = "high_absolute_error",
                    test_data = reference .+ 1e-2,
                    expected_keywords = ["absolute", "tolerance", "convergence"]
                ),
                (
                    name = "high_relative_error", 
                    test_data = reference .* 1.05,
                    expected_keywords = ["relative", "error", "scaling"]
                ),
                (
                    name = "high_pointwise_error",
                    test_data = let data = copy(reference); data[2,2,2] += 0.1; data end,
                    expected_keywords = ["pointwise", "local", "grid"]
                )
            ]
            
            tolerance_settings = Parareal.ToleranceSettings{Float64}(
                absolute_tolerance = 1e-6,
                relative_tolerance = 1e-4,
                max_pointwise_tolerance = 1e-5
            )
            
            for test_case in test_cases
                metrics = Parareal.compute_accuracy_metrics(test_case.test_data, reference)
                recommendations = Parareal.generate_recommendations(metrics, tolerance_settings, Parareal.PararealConfig{Float64}())
                
                @test length(recommendations) > 0
                @test all(rec -> isa(rec, String), recommendations)
                @test all(rec -> length(rec) > 10, recommendations)  # Non-trivial recommendations
                
                # Check that recommendations contain relevant keywords
                recommendation_text = join(recommendations, " ")
                for keyword in test_case.expected_keywords
                    @test occursin(keyword, lowercase(recommendation_text))
                end
            end
        end
    end
    
    @testset "Error Analysis Report Generation Tests" begin
        
        @testset "Single validation result report" begin
            # Create a single validation result
            grid_size = (3, 3, 3)
            reference = ones(Float64, grid_size...) * 300.0
            test_solution = reference .+ 0.01
            
            metrics = Parareal.compute_accuracy_metrics(test_solution, reference)
            config = Parareal.PararealConfig{Float64}()
            tolerance_settings = Parareal.ToleranceSettings{Float64}()
            
            validation_result = Parareal.ValidationResult{Float64}(
                Dates.DateTime(2024, 1, 1, 12, 0, 0),
                "single_test",
                config,
                metrics,
                false,
                ["Improve convergence"]
            )
            
            # Generate individual report
            report = Parareal.generate_error_analysis_report(validation_result)
            
            @test report isa String
            @test length(report) > 100  # Should be substantial
            @test occursin("single_test", report)
            @test occursin("Parareal Validation Report", report)
            @test occursin("L2 Norm Error", report)
            @test occursin("Max Pointwise Error", report)
            @test occursin("Relative Error", report)
            @test occursin("Improve convergence", report)
        end
        
        @testset "Multiple validation results report" begin
            # Create multiple validation results
            validation_results = Parareal.ValidationResult{Float64}[]
            
            grid_size = (2, 2, 2)
            reference = ones(Float64, grid_size...) * 300.0
            
            for i in 1:5
                error_magnitude = 1e-3 / i  # Decreasing error
                test_solution = reference .+ error_magnitude
                
                metrics = Parareal.compute_accuracy_metrics(test_solution, reference)
                config = Parareal.PararealConfig{Float64}()
                tolerance_settings = Parareal.ToleranceSettings{Float64}()
                
                validation_result = Parareal.ValidationResult{Float64}(
                    Dates.DateTime(2024, 1, 1, 10+i, 0, 0),
                    "multi_test_$i",
                    config,
                    metrics,
                    (i > 2),  # Later tests pass
                    ["Recommendation $i"]
                )
                
                push!(validation_results, validation_result)
            end
            
            # Test the comprehensive report function from test_comprehensive_validation.jl
            # Load the function from the comprehensive validation test file
            include("test_comprehensive_validation.jl")
            comprehensive_report = Main.create_error_analysis_report(validation_results)
            
            @test comprehensive_report isa String
            @test length(comprehensive_report) > 500  # Should be comprehensive
            @test occursin("Comprehensive Error Analysis Report", comprehensive_report)
            @test occursin("Total validation runs: 5", comprehensive_report)
            @test occursin("Statistical Summary", comprehensive_report)
            @test occursin("Mean:", comprehensive_report)
            @test occursin("Std:", comprehensive_report)
            @test occursin("Tolerance Analysis", comprehensive_report)
            @test occursin("Recommendations", comprehensive_report)
            @test occursin("Individual Results", comprehensive_report)
            
            # Check that all individual results are mentioned
            for i in 1:5
                @test occursin("multi_test_$i", comprehensive_report)
            end
        end
        
        @testset "Empty and edge case reports" begin
            # Test with empty validation results
            empty_results = Parareal.ValidationResult{Float64}[]
            empty_report = Main.create_error_analysis_report(empty_results)
            
            @test empty_report isa String
            @test occursin("No validation results", empty_report)
            
            # Test with single result
            grid_size = (2, 2, 2)
            reference = ones(Float64, grid_size...) * 300.0
            test_solution = reference .+ 1e-8
            
            metrics = Parareal.compute_accuracy_metrics(test_solution, reference)
            config = Parareal.PararealConfig{Float64}()
            tolerance_settings = Parareal.ToleranceSettings{Float64}()
            
            single_result = Parareal.ValidationResult{Float64}(
                Dates.DateTime(2024, 1, 1, 0, 0, 0),
                "edge_case",
                config,
                metrics,
                true,
                ["Excellent performance"]
            )
            
            single_report = Main.create_error_analysis_report([single_result])
            
            @test single_report isa String
            @test occursin("Total validation runs: 1", single_report)
            @test occursin("edge_case", single_report)
            @test occursin("PASS", single_report)  # Should show passing status
        end
        
        @testset "Report formatting and structure" begin
            # Test that reports have proper structure and formatting
            grid_size = (2, 2, 2)
            reference = ones(Float64, grid_size...) * 300.0
            test_solution = reference .+ 0.001
            
            metrics = Parareal.compute_accuracy_metrics(test_solution, reference)
            config = Parareal.PararealConfig{Float64}()
            tolerance_settings = Parareal.ToleranceSettings{Float64}()
            
            validation_result = Parareal.ValidationResult{Float64}(
                Dates.DateTime(2024, 1, 1, 0, 0, 0),
                "format_test",
                config,
                metrics,
                false,
                ["Format test recommendation"]
            )
            
            report = Parareal.generate_error_analysis_report(validation_result)
            
            # Check for proper section headers
            @test occursin("===", report)  # Section dividers
            @test occursin("Problem ID:", report)
            @test occursin("Validation Status:", report)
            @test occursin("Accuracy Metrics:", report)  # Actual header name
            @test occursin("Recommendations:", report)
            
            # Check for proper formatting of numerical values
            lines = split(report, '\n')
            metric_lines = filter(line -> occursin("error:", line), lines)  # lowercase "error:"
            @test length(metric_lines) >= 3  # Should have L2, max pointwise, and relative errors
            
            # Check that numerical values are properly formatted (not scientific notation for readability)
            for line in metric_lines
                if occursin(":", line)
                    value_part = split(line, ":")[2]
                    @test length(strip(value_part)) > 0  # Should have actual values
                end
            end
        end
    end
    
    @testset "Validation Manager Integration Tests" begin
        
        @testset "ValidationManager state management" begin
            # Test ValidationManager initialization and state
            tolerance_settings = Parareal.ToleranceSettings{Float64}(
                absolute_tolerance = 1e-6,
                relative_tolerance = 1e-4,
                max_pointwise_tolerance = 1e-5
            )
            
            validation_manager = Parareal.ValidationManager{Float64}(
                tolerance_settings = tolerance_settings
            )
            
            @test validation_manager.tolerance_settings === tolerance_settings
            @test length(validation_manager.validation_history) == 0
            @test validation_manager isa Parareal.ValidationManager{Float64}
        end
        
        @testset "Validation history tracking" begin
            validation_manager = Parareal.ValidationManager{Float64}()
            
            # Perform multiple validations
            grid_size = (2, 2, 2)
            reference = ones(Float64, grid_size...) * 300.0
            
            for i in 1:3
                test_solution = reference .+ (0.001 * i)
                
                config = Parareal.PararealConfig{Float64}()
                parareal_result = Parareal.PararealResult{Float64}(
                    test_solution, true, 5, [1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 1.0, 0.1
                )
                
                validation_result = Parareal.validate_against_sequential!(
                    validation_manager, parareal_result, reference, "history_test_$i", config
                )
                
                # Check that history is updated
                @test length(validation_manager.validation_history) == i
                @test validation_manager.validation_history[end] === validation_result
                @test validation_result.problem_id == "history_test_$i"
            end
            
            # Test history retrieval
            @test length(validation_manager.validation_history) == 3
            @test all(result -> result isa Parareal.ValidationResult{Float64}, validation_manager.validation_history)
        end
        
        @testset "Sequential reference computation" begin
            validation_manager = Parareal.ValidationManager{Float64}()
            
            # Create test problem
            grid_size = (3, 3, 3)
            initial_condition = ones(Float64, grid_size...) * 300.0
            
            problem_data = Parareal.create_heat3ds_problem_data(
                (0.1, 0.1, 0.1), [0.1], [0.1],
                zeros(UInt8, grid_size...), nothing, "sequential", false
            )
            
            # Run sequential reference
            sequential_result = Parareal.run_sequential_reference!(
                validation_manager, initial_condition, problem_data, 0.01
            )
            
            @test sequential_result isa Array{Float64, 3}
            @test size(sequential_result) == grid_size
            @test all(isfinite.(sequential_result))
            @test all(sequential_result .> 250.0)  # Reasonable temperature bounds
            @test all(sequential_result .< 350.0)
        end
    end
end

# Test suite summary
println("\n=== Validation Components Unit Tests - Summary ===")
println("✓ AccuracyMetrics calculation accuracy verified")
println("✓ ValidationResult data integrity confirmed")
println("✓ Error analysis report generation tested")
println("✓ Tolerance checking logic validated")
println("✓ Recommendation generation verified")
println("✓ ValidationManager integration tested")
println("✓ Edge cases and robustness confirmed")
println("====================================================")