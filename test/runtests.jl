#!/usr/bin/env julia

"""
Heat3ds Parareal時間並列化 - 統合テストスイート

このスクリプトは、Parareal時間並列化機能の全テストを実行します。
テストはカテゴリ別に分類されており、段階的に実行できます。

実行方法:
    julia test/runtests.jl [category]
    
カテゴリ:
    unit        - 単体テスト
    integration - 統合テスト  
    performance - 性能テスト
    validation  - 検証テスト
    all         - 全テスト (デフォルト)
"""

using Test
using Pkg

# テストカテゴリの定義
const TEST_CATEGORIES = Dict(
    "unit" => [
        "unit/test_mpi_initialization.jl",
        "unit/test_mpi_communication.jl", 
        "unit/test_time_windows.jl",
        "unit/test_parameter_validation.jl",
        "unit/test_solver_compatibility.jl",
        "unit/test_threadsx_integration.jl",
        "unit/test_error_handling.jl",
        "unit/test_resource_management.jl",
        "unit/test_logging_minimal.jl"
    ],
    "integration" => [
        "integration/test_heat3ds_integration.jl",
        "integration/test_hybrid_parallelization.jl",
        "integration/test_boundary_condition_integration.jl",
        "integration/test_boundary_condition_mpi_compatibility.jl",
        "integration/test_backward_compatibility.jl",
        "integration/test_output_format_consistency.jl",
        "integration/test_output_format_comprehensive.jl",
        "integration/test_output_format_simple.jl",
        "integration/test_output_generation.jl",
        "integration/test_example_configurations.jl"
    ],
    "performance" => [
        "performance/test_performance_monitoring.jl",
        "performance/test_performance_monitoring_accuracy.jl",
        "performance/test_performance_analysis.jl",
        "performance/test_performance_metrics.jl",
        "performance/test_performance_integration.jl",
        "performance/test_performance_claims.jl",
        "performance/test_parameter_space_exploration.jl",
        "performance/test_time_step_ratio_optimization.jl"
    ],
    "validation" => [
        "validation/test_parareal_convergence.jl",
        "validation/test_sequential_consistency.jl",
        "validation/test_numerical_precision_preservation.jl",
        "validation/test_graceful_degradation.jl",
        "validation/test_comprehensive_validation.jl",
        "validation/test_validation_components.jl",
        "validation/test_boundary_condition_compatibility.jl",
        "validation/test_benchmark_accuracy.jl"
    ]
)

function print_header(category)
    """テストカテゴリのヘッダー表示"""
    println("="^60)
    println("Running $(uppercase(category)) Tests")
    println("="^60)
end

function run_test_file(test_file)
    """単一テストファイルの実行"""
    test_path = joinpath(@__DIR__, test_file)
    
    if !isfile(test_path)
        @warn "Test file not found: $test_path"
        return false
    end
    
    println("Running: $test_file")
    
    try
        include(test_path)
        println("✅ $test_file - PASSED")
        return true
    catch e
        println("❌ $test_file - FAILED: $e")
        return false
    end
end

function run_test_category(category)
    """テストカテゴリの実行"""
    if !haskey(TEST_CATEGORIES, category)
        @error "Unknown test category: $category"
        return false
    end
    
    print_header(category)
    
    test_files = TEST_CATEGORIES[category]
    passed = 0
    total = length(test_files)
    
    for test_file in test_files
        if run_test_file(test_file)
            passed += 1
        end
        println()
    end
    
    println("$category Tests Summary: $passed/$total passed")
    return passed == total
end

function main()
    """メイン実行関数"""
    
    # コマンドライン引数の処理
    category = length(ARGS) > 0 ? ARGS[1] : "all"
    
    println("Heat3ds Parareal Time Parallelization Test Suite")
    println("Category: $category")
    println("Julia version: $(VERSION)")
    println()
    
    # 必要なパッケージの確認
    required_packages = ["Test", "MPI"]  # Heat3dsは実際の環境でのみ利用可能
    
    for pkg in required_packages
        try
            eval(:(using $(Symbol(pkg))))
        catch e
            @warn "Package $pkg not available: $e"
        end
    end
    
    # テスト実行
    if category == "all"
        # 全カテゴリを順次実行
        all_passed = true
        for cat in ["unit", "integration", "performance", "validation"]
            if !run_test_category(cat)
                all_passed = false
            end
            println()
        end
        
        println("="^60)
        println("OVERALL TEST RESULT: $(all_passed ? "✅ ALL PASSED" : "❌ SOME FAILED")")
        println("="^60)
        
        return all_passed ? 0 : 1
        
    elseif haskey(TEST_CATEGORIES, category)
        # 指定カテゴリのみ実行
        success = run_test_category(category)
        return success ? 0 : 1
        
    else
        @error "Invalid category: $category"
        println("Available categories: $(join(keys(TEST_CATEGORIES), ", ")), all")
        return 1
    end
end

# スクリプトとして実行された場合
if abspath(PROGRAM_FILE) == @__FILE__
    exit_code = main()
    exit(exit_code)
end