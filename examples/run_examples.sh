#!/bin/bash

# Parareal実行例スクリプト集
# 
# このスクリプトは、様々なParareal実行例を提供します。
# 使用方法: ./run_examples.sh [example_name]
#
# 利用可能な例:
#   basic       - 基本的なParareal実行
#   ic_thermal  - IC熱解析例
#   optimization - パラメータ最適化
#   benchmark   - ベンチマーク実行
#   all         - 全ての例を実行

set -e  # エラー時に停止

# デフォルト設定
DEFAULT_PROCESSES=4
DEFAULT_THREADS=4

# 色付き出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ヘルプ表示
show_help() {
    echo "Parareal Examples Runner"
    echo ""
    echo "Usage: $0 [example_name] [options]"
    echo ""
    echo "Examples:"
    echo "  basic       - Basic Parareal execution"
    echo "  ic_thermal  - IC thermal analysis example"
    echo "  optimization - Parameter optimization"
    echo "  benchmark   - Benchmark problems"
    echo "  all         - Run all examples"
    echo ""
    echo "Options:"
    echo "  -p, --processes N    Number of MPI processes (default: $DEFAULT_PROCESSES)"
    echo "  -t, --threads N      Number of threads per process (default: $DEFAULT_THREADS)"
    echo "  -h, --help          Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 basic -p 8 -t 2"
    echo "  $0 benchmark --processes 16"
    echo "  $0 all"
}

# ログ出力関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 環境チェック
check_environment() {
    log_info "Checking environment..."
    
    # Julia確認
    if ! command -v julia &> /dev/null; then
        log_error "Julia not found. Please install Julia."
        exit 1
    fi
    
    # MPI確認
    if ! command -v mpirun &> /dev/null; then
        log_error "MPI not found. Please install MPI (OpenMPI or MPICH)."
        exit 1
    fi
    
    # Julia MPIパッケージ確認
    if ! julia -e "using MPI" &> /dev/null; then
        log_warning "MPI.jl package not found. Installing..."
        julia -e "using Pkg; Pkg.add(\"MPI\"); using MPI; MPI.install_mpiexec()"
    fi
    
    # Heat3ds確認
    if ! julia -e "using Heat3ds" &> /dev/null; then
        log_error "Heat3ds package not found. Please install Heat3ds."
        exit 1
    fi
    
    log_success "Environment check passed"
}

# 基本例の実行
run_basic_example() {
    local processes=$1
    local threads=$2
    
    log_info "Running basic Parareal example..."
    log_info "Processes: $processes, Threads per process: $threads"
    
    export JULIA_NUM_THREADS=$threads
    
    mpirun -np $processes julia basic_parareal_example.jl
    
    if [ $? -eq 0 ]; then
        log_success "Basic example completed successfully"
    else
        log_error "Basic example failed"
        return 1
    fi
}

# IC熱解析例の実行
run_ic_thermal_example() {
    local processes=$1
    local threads=$2
    
    log_info "Running IC thermal analysis example..."
    log_info "Processes: $processes, Threads per process: $threads"
    
    if [ $processes -lt 4 ]; then
        log_warning "IC thermal analysis works best with 4+ processes"
    fi
    
    export JULIA_NUM_THREADS=$threads
    
    mpirun -np $processes julia ic_thermal_analysis_example.jl
    
    if [ $? -eq 0 ]; then
        log_success "IC thermal analysis completed successfully"
    else
        log_error "IC thermal analysis failed"
        return 1
    fi
}

# パラメータ最適化例の実行
run_optimization_example() {
    local processes=$1
    local threads=$2
    
    log_info "Running parameter optimization example..."
    log_info "Processes: $processes, Threads per process: $threads"
    
    if [ $processes -lt 8 ]; then
        log_warning "Parameter optimization works best with 8+ processes"
    fi
    
    export JULIA_NUM_THREADS=$threads
    
    # 最適化は時間がかかるので、タイムアウトを設定
    timeout 1800 mpirun -np $processes julia parameter_optimization_example.jl
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_success "Parameter optimization completed successfully"
    elif [ $exit_code -eq 124 ]; then
        log_warning "Parameter optimization timed out (30 minutes)"
    else
        log_error "Parameter optimization failed"
        return 1
    fi
}

# ベンチマーク例の実行
run_benchmark_example() {
    local processes=$1
    local threads=$2
    local problem=${3:-"all"}
    
    log_info "Running benchmark problems..."
    log_info "Processes: $processes, Threads per process: $threads, Problem: $problem"
    
    export JULIA_NUM_THREADS=$threads
    
    mpirun -np $processes julia benchmark_problems.jl $problem
    
    if [ $? -eq 0 ]; then
        log_success "Benchmark completed successfully"
    else
        log_error "Benchmark failed"
        return 1
    fi
}

# 全例の実行
run_all_examples() {
    local processes=$1
    local threads=$2
    
    log_info "Running all Parareal examples..."
    
    # 基本例
    run_basic_example $processes $threads
    if [ $? -ne 0 ]; then
        log_error "Stopping due to basic example failure"
        return 1
    fi
    
    sleep 5
    
    # IC熱解析例
    run_ic_thermal_example $processes $threads
    if [ $? -ne 0 ]; then
        log_warning "IC thermal example failed, continuing..."
    fi
    
    sleep 5
    
    # ベンチマーク（小規模問題のみ）
    run_benchmark_example $processes $threads "small"
    if [ $? -ne 0 ]; then
        log_warning "Benchmark failed, continuing..."
    fi
    
    # パラメータ最適化は時間がかかるのでスキップ
    log_info "Skipping parameter optimization (too time-consuming for 'all' mode)"
    log_info "To run optimization: $0 optimization -p $processes -t $threads"
    
    log_success "All examples completed"
}

# システム情報の表示
show_system_info() {
    log_info "System Information:"
    echo "  OS: $(uname -s) $(uname -r)"
    echo "  CPU cores: $(nproc)"
    echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"
    echo "  Julia version: $(julia --version)"
    echo "  MPI version: $(mpirun --version | head -n1)"
    
    # Julia パッケージ情報
    echo "  Julia packages:"
    julia -e "using Pkg; Pkg.status([\"MPI\", \"Heat3ds\", \"ThreadsX\"])" 2>/dev/null || echo "    Package info unavailable"
}

# メイン処理
main() {
    local example_name=""
    local processes=$DEFAULT_PROCESSES
    local threads=$DEFAULT_THREADS
    
    # 引数解析
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--processes)
                processes="$2"
                shift 2
                ;;
            -t|--threads)
                threads="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                if [ -z "$example_name" ]; then
                    example_name="$1"
                else
                    log_error "Multiple example names specified"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # デフォルト例の設定
    if [ -z "$example_name" ]; then
        example_name="basic"
    fi
    
    # 引数検証
    if ! [[ "$processes" =~ ^[0-9]+$ ]] || [ "$processes" -lt 1 ]; then
        log_error "Invalid number of processes: $processes"
        exit 1
    fi
    
    if ! [[ "$threads" =~ ^[0-9]+$ ]] || [ "$threads" -lt 1 ]; then
        log_error "Invalid number of threads: $threads"
        exit 1
    fi
    
    # システム情報表示
    show_system_info
    echo ""
    
    # 環境チェック
    check_environment
    echo ""
    
    # 例の実行
    case $example_name in
        basic)
            run_basic_example $processes $threads
            ;;
        ic_thermal)
            run_ic_thermal_example $processes $threads
            ;;
        optimization)
            run_optimization_example $processes $threads
            ;;
        benchmark)
            run_benchmark_example $processes $threads
            ;;
        all)
            run_all_examples $processes $threads
            ;;
        *)
            log_error "Unknown example: $example_name"
            show_help
            exit 1
            ;;
    esac
    
    echo ""
    log_success "Script completed"
}

# スクリプトとして実行された場合のみmain()を呼び出し
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi