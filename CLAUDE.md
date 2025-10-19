# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

3次元非一様格子における熱伝導解析のためのJuliaシミュレーションコード。
IC (Integrated Circuit) の多層構造における温度分布を計算する。

## 実行方法

### メインシミュレーションの実行

```bash
julia heat3ds.jl
```

デフォルトパラメータ: 240x240x31グリッド、PBiCGSTABソルバー、Gauss-Seidelスムーザー

### パラメータ付き実行

```julia
# Julia REPLから
include("heat3ds.jl")
q3d(NX, NY, NZ, "solver", "smoother", epsilon=1.0e-6, par="thread")

# 例:
q3d(240, 240, 31, "pbicgstab", "gs", epsilon=1.0e-4, par="sequential")
```

パラメータ:
- `NX, NY, NZ`: 内部セル数
- `solver`: "pbicgstab", "cg", "sor"
- `smoother`: "gs" (Gauss-Seidel), "" (なし)
- `epsilon`: 収束判定基準 (デフォルト: 1.0e-6)
- `par`: "thread" (並列), "sequential" (逐次)
- `is_steady`: 定常解析フラグ (デフォルト: false)。trueの場合、時間微分項を除外した定常熱伝導方程式を解く

### テスト実行

個別モジュールのテスト:

```bash
julia modelA.jl           # ジオメトリモデルのテスト
julia test_symmetry.jl    # 係数行列の対称性テスト
julia test_steady_state.jl # 定常熱伝導の既知解テスト
```

テスト種別:
- **test_symmetry.jl**: 係数行列Aの対称性検証。ランダムベクトルx, yに対して (Ax, y) と (x, Ay) を比較
- **test_steady_state.jl**: 1次元定常熱伝導問題の解析解との比較

## アーキテクチャ

### モジュール構成

- **heat3ds.jl**: メインエントリーポイント。シミュレーション全体を統括
- **common.jl**: 共通定数、データ構造、境界条件タイプの定義
- **modelA.jl**: IC幾何モデル構築。TSV、はんだバンプ、シリコン層、基板、ヒートシンクのモデル化
- **NonUniform.jl**: 非一様格子用の線形ソルバー (PBiCGSTAB, CG, SOR)
- **boundary_conditions.jl**: 境界条件の設定・適用 (等温、熱流束、対流)
- **RHS.jl**: 右辺ベクトル(RHS)の計算、境界条件の適用
- **Zcoord.jl**: Z方向の非一様格子座標生成
- **plotter.jl**: 結果の可視化
- **convergence_history.jl**: 収束履歴の記録・可視化
- **parse_log_residuals.jl**: ログファイルからの残差データ解析

### データフロー

1. **前処理** (`preprocess!`):
   - Z方向非一様格子の生成 (`Zcoordinate.genZ!`)
   - ジオメトリモデルの構築 (`modelA.fillID!`)
   - 物性値の設定 (`modelA.setProperties!`)

2. **境界条件設定**:
   - `set_mode3_bc_parameters()` で各面の境界条件を定義
   - `BoundaryConditions.apply_boundary_conditions!` で適用

3. **反復計算** (`main`):
   - RHS計算 (`calRHS!`)
   - 線形ソルバー (`PBiCGSTAB!`, `CG!`)
   - 収束判定

4. **後処理**:
   - 温度分布の可視化 (`plot_slice_xz_nu`, `plot_slice_xy_nu`, `plot_line_z_nu`)
   - 収束履歴の出力 (`plot_convergence_curve`, `export_convergence_csv`)

### 重要な構造体

- `WorkBuffers`: 全計算用配列を保持
  - `θ`: 温度配列
  - `λ`, `ρ`, `cp`: 物性値配列 (熱拡散率、密度、比熱)
  - `mask`: マスク配列 (境界条件で値を固定する場合は0)
  - `b`, `hsrc`: RHS配列、熱源項
  - `pcg_*`: PBiCGSTABソルバー用の作業配列
- `BoundaryCondition`: 単一境界面の境界条件
- `BoundaryConditionSet`: 6面の境界条件セット (x_minus, x_plus, y_minus, y_plus, z_minus, z_plus)
- `BoundaryType`: 境界条件タイプの列挙型 (ISOTHERMAL, HEAT_FLUX, CONVECTION)

### 境界条件システム

境界条件は3種類:
- **ISOTHERMAL**: 等温条件 (Dirichlet)。mask=0で固定温度
- **HEAT_FLUX**: 熱流束条件 (Neumann)。λを調整して実装
- **CONVECTION**: 熱伝達条件 (Robin)。RHS項に追加項を適用

Z方向の計算範囲は境界条件によって自動調整される (`compute_z_range`)。

### 並列化

FLoops.jl を使用した並列化:
- `par="thread"`: ThreadedEx() バックエンド
- `par="sequential"`: SequentialEx() バックエンド

並列化対象: RHS計算、ソルバーの内積・AXPY演算、前処理

## 開発時の注意点

### インデント

常に2スペースインデントを使用する。

### 定常解析と非定常解析

- **定常解析** (`is_steady=true`): 時間微分項を除外。支配方程式が ∇·(λ∇T) + Q = 0 となる
- **非定常解析** (`is_steady=false`): 時間微分項を含む。支配方程式が ρcp∂T/∂t = ∇·(λ∇T) + Q となる
- 定常解析では大きなΔtの非定常解析の極限に相当するが、数値的により安定で高速

### 収束判定

- 反復ソルバー (PBiCGSTAB, CG) の収束判定は相対残差ノルム `||r||/||b||` で行う
- `epsilon` パラメータで収束基準を指定 (デフォルト: 1.0e-6)
- 最大反復回数は `ItrMax = 8000` (common.jl で定義)

### ジオメトリモデル

modelA.jl の座標系:
- Z軸: 0.0 (基板下面) ～ 0.6e-3 m (ヒートシンク上面)
- 主要Z座標: zm0～zm12 (13層の非一様格子)
- TSV、はんだバンプは円柱・球体の体積占有率50%以上でセルに割り当て

### 格子インデックス規則

配列サイズ: `(MX, MY, MZ) = (NX+2, NY+2, NZ+2)`
- インデックス 1, SZ[*]: ガイドセル (境界条件用)
- インデックス 2～SZ[*]-1: 計算内点
- Z方向は境界条件により計算範囲が変動 (`z_range`)

### ソルバーの選択

- **PBiCGSTAB**: 非対称行列用、収束が速い。推奨
- **CG**: 対称行列用
- **SOR**: 直接法、デバッグ用

スムーザー "gs" (Gauss-Seidel) を使用すると前処理の効果で収束が改善される。
