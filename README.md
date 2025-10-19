# Heat3ds - 3D非一様格子熱伝導解析

IC（集積回路）の多層構造における温度分布を計算するJuliaシミュレーションコード。

## プロジェクト概要

3次元非一様格子を用いた熱伝導解析ソルバー。TSV、はんだバンプ、シリコン層、基板、ヒートシンクを含むICパッケージの熱解析に対応。

## 主要機能

- **非一様格子**: Z方向に最適化された非等間隔格子
- **反復ソルバー**: PBiCGSTAB、CG、SORをサポート
- **前処理**: Gauss-Seidelスムーザーによる収束加速
- **並列化**: ThreadsXによるマルチスレッド対応
- **境界条件**: 等温、熱流束、対流の3種類
- **定常/非定常解析**: is_steadyフラグで切り替え可能

## ディレクトリ構成

```
H2/
├── README.md                    # このファイル
├── CLAUDE.md                    # Claude Code用プロジェクト説明
├── MIGRATION_PLAN.md            # CommonSolver準拠への移植計画
├── TODO_NEXT_SESSION.md         # 次回セッション用メモ
├── .gitignore                   # Git除外設定
│
├── run.jl                       # 実行スクリプト（プロジェクトルートから実行）
├── test.jl                      # テスト実行スクリプト
│
├── src/                         # ソースコード
│   ├── heat3ds.jl               # メインプログラム
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
├── output/                      # 実行結果（.gitignore）
│   ├── log.txt                  # ソルバーログ
│   ├── temp3*.png               # 温度分布図
│   ├── temp3Z_*.csv             # Z方向温度プロファイル
│   └── alpha3.png               # 熱拡散率分布
│
└── results/                     # 収束履歴（.gitignore）
    └── convergence/
        └── convergence_*.png/csv
```

## 実行方法

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

```julia
using Printf
using LinearAlgebra
using FLoops
using ThreadsX
```

## 開発履歴

- **2024-10-19**: 型パラメータ化とHC配列対応完了（ステップ5.9-6）
- **2024-10-16**: CommonSolver準拠への移植開始（ステップ5.1-5.8）
- **2024-09-15**: 初期バージョン

## ライセンス

MIT License

## 著者

- 開発: kenoogl
- 支援: Claude Code (Anthropic)

## 関連文書

- [CLAUDE.md](CLAUDE.md) - Claude Code用の詳細仕様
- [MIGRATION_PLAN.md](MIGRATION_PLAN.md) - CommonSolver準拠への移植計画
