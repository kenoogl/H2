# Heat3ds - 3D非一様格子熱伝導解析 with Parareal時間並列化

IC（集積回路）の多層構造における温度分布を計算するJuliaシミュレーションコード。
**新機能**: MPI+ThreadsXハイブリッド並列化によるParareal時間並列化対応。

## 🚀 新機能: Parareal時間並列化

### 主要特徴
- **ハイブリッドMPI+Threads並列化**: MPI時間並列 + ThreadsX空間並列
- **自動パラメータ最適化**: 文献ベースガイドライン + 自動調整
- **堅牢な検証システム**: 機械精度レベルの精度確認
- **包括的性能監視**: リアルタイム効率測定
- **グレースフルデグラデーション**: 収束失敗時の自動フォールバック

### 期待性能
- **小規模問題**: 1.2-2.5倍高速化
- **中規模問題**: 2.0-4.0倍高速化  
- **大規模問題**: 3.0-8.0倍高速化
- **精度**: 機械精度レベル (< 1e-6相対誤差)

## プロジェクト概要

3次元非一様格子を用いた熱伝導解析ソルバー。TSV、はんだバンプ、シリコン層、基板、ヒートシンクを含むICパッケージの熱解析に対応。

## 主要機能

### 従来機能
- **非一様格子**: Z方向に最適化された非等間隔格子
- **反復ソルバー**: PBiCGSTAB、CG、SORをサポート
- **前処理**: Gauss-Seidelスムーザーによる収束加速
- **並列化**: ThreadsXによるマルチスレッド対応
- **境界条件**: 等温、熱流束、対流の3種類
- **定常/非定常解析**: is_steadyフラグで切り替え可能

### 新機能: Parareal時間並列化
- **時間方向並列化**: 複数時間窓の並列処理
- **MPI通信**: プロセス間での温度場データ交換
- **自動最適化**: 問題特性に応じたパラメータ調整
- **精度保証**: 逐次計算と同等の数値精度
- **性能監視**: 詳細なスピードアップ・効率測定

## ディレクトリ構成

```
H2/
├── README.md                    # このファイル
├── CLAUDE.md                    # Claude Code用プロジェクト説明
├── MIGRATION_PLAN.md            # CommonSolver準拠への移植計画
├── .gitignore                   # Git除外設定
│
├── run.jl                       # 実行スクリプト（プロジェクトルートから実行）
├── test.jl                      # テスト実行スクリプト
│
├── src/                         # ソースコード
│   ├── heat3ds.jl               # メインプログラム
│   ├── parareal.jl              # Parareal時間並列化（新規）
│   ├── parameter_optimization.jl # パラメータ最適化（新規）
│   ├── output_format.jl         # 出力形式管理（新規）
│   ├── common.jl                # 共通定数・データ構造
│   ├── NonUniform.jl            # 線形ソルバー（PBiCGSTAB, CG, SOR）
│   ├── boundary_conditions.jl   # 境界条件設定
│   ├── RHS.jl                   # 右辺ベクトル計算
│   ├── Zcoord.jl                # Z方向格子生成
│   ├── modelA.jl                # ICジオメトリモデル
│   ├── plotter.jl               # 結果可視化
│   ├── convergence_history.jl   # 収束履歴記録
│   ├── parse_log_residuals.jl   # ログ解析
│   ├── test_symmetry.jl         # 係数行列対称性テスト
│   └── test_steady_state.jl     # 定常解析テスト
│
├── docs/                        # ドキュメント（新規）
│   ├── parareal_user_guide.md   # Pararealユーザーガイド
│   ├── mpi_setup_guide.md       # MPI設定ガイド
│   └── troubleshooting_faq.md   # トラブルシューティング
│
├── examples/                    # 実行例・サンプル（新規）
│   ├── README.md                # サンプル説明
│   ├── run_examples.sh          # 自動実行スクリプト
│   ├── basic_parareal_example.jl # 基本実行例
│   ├── ic_thermal_analysis_example.jl # IC熱解析例
│   ├── parameter_optimization_example.jl # パラメータ最適化例
│   └── benchmark_problems.jl    # ベンチマーク問題集
│
├── test/                        # テストスイート（拡張）
│   ├── runtests.jl              # メインテスト実行
│   ├── README.md                # テスト説明
│   ├── unit/                    # 単体テスト
│   ├── integration/             # 統合テスト
│   ├── performance/             # 性能テスト
│   ├── validation/              # 検証テスト
│   └── summaries/               # テスト結果サマリー
│
├── .kiro/                       # Kiro仕様（開発用）
│   └── specs/parareal-time-parallelization/
│       ├── requirements.md      # 要件定義
│       ├── design.md            # 設計文書
│       └── tasks.md             # 実装計画
│
├── archive/                     # アーカイブ（新規）
│   └── temp_files/              # 一時ファイル
│
├── output/                      # 実行結果（.gitignore）
├── results/                     # 収束履歴（.gitignore）
└── benchmarks/                  # ベンチマーク結果
```

## 実行方法

### 🚀 Parareal時間並列化実行（新機能）

#### 基本的なParareal実行
```bash
# 4プロセスでParareal実行
mpirun -np 4 julia examples/basic_parareal_example.jl

# 8プロセス、各プロセス4スレッドで実行
export JULIA_NUM_THREADS=4
mpirun -np 8 julia examples/ic_thermal_analysis_example.jl
```

#### 自動実行スクリプト
```bash
# 基本例の実行
./examples/run_examples.sh basic -p 4 -t 2

# IC熱解析例の実行
./examples/run_examples.sh ic_thermal -p 8 -t 4

# パラメータ最適化
./examples/run_examples.sh optimization -p 16 -t 2

# ベンチマーク実行
./examples/run_examples.sh benchmark -p 8
```

#### Julia REPLからのParareal実行
```julia
using MPI
using Heat3ds

# Parareal設定
config = PararealConfig(
    total_time=1.0,
    n_time_windows=4,
    dt_coarse=0.01,
    dt_fine=0.001,
    time_step_ratio=10.0,
    max_iterations=15,
    convergence_tolerance=1.0e-6,
    n_mpi_processes=4,
    n_threads_per_process=2
)

# Parareal実行
result = q3d(64, 64, 32,
            solver="pbicgstab",
            parareal=true,
            parareal_config=config)
```

### 従来の逐次実行

### 基本実行（プロジェクトルートから）

```bash
# デフォルトパラメータで実行
julia run.jl
```

デフォルト: 240x240x30グリッド、PBiCGSTAB、Gauss-Seidel、定常解析

### パラメータ指定実行

```bash
# コマンドライン引数で指定
julia run.jl NX NY NZ solver smoother epsilon par [is_steady]

# 例: 240x240x30グリッド、PBiCGSTAB、GS、ε=1e-4、逐次、定常解析
julia run.jl 240 240 30 pbicgstab gs 1e-4 sequential true

# 例: 小グリッド、CG、スムーザーなし、ε=1e-6、逐次、非定常解析
julia run.jl 40 40 30 cg "" 1e-6 sequential false
```

### 並列実行

```bash
# 4スレッドで並列実行
julia -t 4 run.jl 240 240 30 pbicgstab gs 1e-4 thread true

# 8スレッドで並列実行
julia -t 8 run.jl 240 240 30 pbicgstab gs 1e-4 thread true

# 注意: par="thread"の場合、julia -t N で起動してください
```

### Julia REPLから実行

```julia
# srcディレクトリをロードパスに追加
push!(LOAD_PATH, "src")
include("src/heat3ds.jl")

# 関数呼び出し
q3d(240, 240, 30, "pbicgstab", "gs", epsilon=1.0e-4, par="thread", is_steady=true)
q3d(240, 240, 30, "cg", "gs", epsilon=1.0e-6, par="sequential", is_steady=true)
```

### パラメータ

- `NX, NY, NZ`: 内部セル数（例: 240, 240, 30）
- `solver`: "pbicgstab", "cg", "sor"
- `smoother`: "gs" (Gauss-Seidel), "" (なし)
- `epsilon`: 収束判定基準（デフォルト: 1.0e-6）
- `par`: "thread" (並列), "sequential" (逐次)
- `is_steady`: true (定常), false (非定常)

### テスト実行

```bash
# すべてのテストを実行
julia test.jl

# 個別テスト
julia test.jl symmetry   # 係数行列の対称性テスト
julia test.jl steady     # 定常熱伝導の既知解テスト
```

## 技術仕様

### グリッド

- **X, Y方向**: 一様格子（1.2mm / NX）
- **Z方向**: 非一様格子（13層、0〜0.6mm）
  - zm0-zm12の座標で定義（modelA.jl）
  - Zcase2!で生成（Zcoord.jl）

### 境界条件（Mode 3）

- **X, Y側面**: 熱伝達（h=5 W/(m²⋅K), θ∞=300K）
- **Z下面**: 等温（PCB温度 300K）
- **Z上面**: 熱伝達（h=5 W/(m²⋅K), θ∞=300K）

### ソルバー

#### PBiCGSTAB（推奨）
- 非対称行列用
- 前処理付き双共役勾配安定化法
- 高速収束

#### CG
- 対称行列用
- 前処理付き共役勾配法

#### SOR
- 逐次過緩和法
- デバッグ用

### 前処理

- **Gauss-Seidel** (smoother="gs")
  - Red-Black SOR、5回反復
  - 収束を大幅に改善

### 型パラメータ化

全関数がAbstractFloatで型パラメータ化され、Float32/Float64の両方に対応。

## 実行例

### 240x240x30グリッド、定常解析

```
Grid  : 242 242 32
Solver: pbicgstab with smoother gs
ItrMax : 8000
ε      : 0.0001

Initial residual = 0.002771608776496581
         1     6.58308938608891E-01
         2     8.89700596995947E-01
         ...
       208     9.72579214884977E-05
Converged at 208

θmin=300.6K  θmax=349.4K
計算時間: 183秒
```

## 依存パッケージ

### 基本パッケージ
```julia
using Printf
using LinearAlgebra
using FLoops
using ThreadsX
```

### Parareal時間並列化用（新規）
```julia
using MPI          # MPI並列化
using JSON         # 設定・結果保存
using Dates        # タイムスタンプ
using Statistics   # 統計解析
```

### インストール方法
```bash
# 基本パッケージ
julia -e "using Pkg; Pkg.add([\"FLoops\", \"ThreadsX\"])"

# Parareal用パッケージ
julia -e "using Pkg; Pkg.add([\"MPI\", \"JSON\", \"Dates\", \"Statistics\"])"

# MPI設定
julia -e "using MPI; MPI.install_mpiexec()"
```

## 開発履歴

- **2024-12-30**: Parareal時間並列化機能完全実装完了
  - MPI+ThreadsXハイブリッド並列化
  - 自動パラメータ最適化システム
  - 包括的ドキュメント・サンプル・テストスイート
  - 期待性能: 2-8倍高速化、機械精度レベル精度
- **2024-10-19**: 型パラメータ化とHC配列対応完了（ステップ5.9-6）
- **2024-10-16**: CommonSolver準拠への移植開始（ステップ5.1-5.8）
- **2024-09-15**: 初期バージョン

## ライセンス

MIT License

## 著者

- 開発: kenoogl
- 支援: Claude Code (Anthropic)

## 関連文書

### 基本文書
- [CLAUDE.md](CLAUDE.md) - Claude Code用の詳細仕様
- [MIGRATION_PLAN.md](MIGRATION_PLAN.md) - CommonSolver準拠への移植計画

### Parareal時間並列化文書
- [docs/parareal_user_guide.md](docs/parareal_user_guide.md) - Pararealユーザーガイド
- [docs/mpi_setup_guide.md](docs/mpi_setup_guide.md) - MPI設定・実行ガイド
- [docs/troubleshooting_faq.md](docs/troubleshooting_faq.md) - トラブルシューティング・FAQ
- [examples/README.md](examples/README.md) - 実行例・サンプル説明
- [test/README.md](test/README.md) - テストスイート説明

### 開発仕様書
- [.kiro/specs/parareal-time-parallelization/requirements.md](.kiro/specs/parareal-time-parallelization/requirements.md) - 要件定義
- [.kiro/specs/parareal-time-parallelization/design.md](.kiro/specs/parareal-time-parallelization/design.md) - 設計文書
- [.kiro/specs/parareal-time-parallelization/tasks.md](.kiro/specs/parareal-time-parallelization/tasks.md) - 実装計画
