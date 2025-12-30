# Heat3ds Parareal時間並列化 - テストスイート

このディレクトリには、Parareal時間並列化機能の包括的なテストスイートが含まれています。

## 📁 ディレクトリ構造

```
test/
├── runtests.jl              # メインテスト実行スクリプト
├── README.md                # このファイル
├── unit/                    # 単体テスト
├── integration/             # 統合テスト
├── performance/             # 性能テスト
├── validation/              # 検証テスト
└── summaries/               # テスト結果サマリー
```

## 🚀 テスト実行方法

### 全テストの実行
```bash
julia test/runtests.jl
# または
julia test/runtests.jl all
```

### カテゴリ別実行
```bash
# 単体テスト
julia test/runtests.jl unit

# 統合テスト
julia test/runtests.jl integration

# 性能テスト
julia test/runtests.jl performance

# 検証テスト
julia test/runtests.jl validation
```

### 個別テストファイルの実行
```bash
julia test/unit/test_mpi_initialization.jl
```

## 📋 テストカテゴリ

### 🔧 Unit Tests (単体テスト)
基本的なコンポーネントの動作を検証

- **test_mpi_initialization.jl** - MPI環境初期化
- **test_mpi_communication.jl** - MPI通信機能
- **test_time_windows.jl** - 時間窓管理
- **test_parameter_validation.jl** - パラメータ検証
- **test_solver_compatibility.jl** - ソルバー互換性
- **test_threadsx_integration.jl** - ThreadsX統合
- **test_error_handling.jl** - エラーハンドリング
- **test_resource_management.jl** - リソース管理
- **test_logging_minimal.jl** - ログ機能

### 🔗 Integration Tests (統合テスト)
コンポーネント間の連携を検証

- **test_heat3ds_integration.jl** - Heat3ds統合
- **test_hybrid_parallelization.jl** - ハイブリッド並列化
- **test_boundary_condition_integration.jl** - 境界条件統合
- **test_boundary_condition_mpi_compatibility.jl** - 境界条件MPI互換性
- **test_backward_compatibility.jl** - 後方互換性
- **test_output_format_consistency.jl** - 出力形式一貫性
- **test_output_format_comprehensive.jl** - 包括的出力形式
- **test_output_format_simple.jl** - 簡単出力形式
- **test_output_generation.jl** - 出力生成
- **test_example_configurations.jl** - サンプル設定

### ⚡ Performance Tests (性能テスト)
性能とスケーラビリティを検証

- **test_performance_monitoring.jl** - 性能監視
- **test_performance_monitoring_accuracy.jl** - 性能監視精度
- **test_performance_analysis.jl** - 性能解析
- **test_performance_metrics.jl** - 性能メトリクス
- **test_performance_integration.jl** - 性能統合
- **test_performance_claims.jl** - 性能主張検証
- **test_parameter_space_exploration.jl** - パラメータ空間探索
- **test_time_step_ratio_optimization.jl** - 時間ステップ比最適化

### ✅ Validation Tests (検証テスト)
数値精度と正確性を検証

- **test_parareal_convergence.jl** - Parareal収束
- **test_sequential_consistency.jl** - 逐次一貫性
- **test_numerical_precision_preservation.jl** - 数値精度保持
- **test_graceful_degradation.jl** - グレースフルデグラデーション
- **test_comprehensive_validation.jl** - 包括的検証
- **test_validation_components.jl** - 検証コンポーネント
- **test_boundary_condition_compatibility.jl** - 境界条件互換性
- **test_benchmark_accuracy.jl** - ベンチマーク精度

## 🎯 テスト実行の推奨順序

1. **Unit Tests** - 基本機能の確認
2. **Integration Tests** - 統合動作の確認
3. **Validation Tests** - 数値精度の確認
4. **Performance Tests** - 性能評価（時間がかかる）

## 📊 期待される結果

### 成功基準
- **Unit Tests**: 100% パス
- **Integration Tests**: 95%以上 パス
- **Validation Tests**: 90%以上 パス（数値精度依存）
- **Performance Tests**: 80%以上 パス（ハードウェア依存）

### 一般的な失敗原因
1. **MPI環境未設定** - MPI.jlパッケージ未インストール
2. **Heat3ds未インストール** - 実際のHeat3dsパッケージが必要
3. **リソース不足** - メモリ不足、プロセス数不足
4. **数値精度** - ハードウェア・コンパイラ依存の精度差

## 🔧 トラブルシューティング

### MPI関連エラー
```bash
# MPI.jlの再インストール
julia -e "using Pkg; Pkg.rm(\"MPI\"); Pkg.add(\"MPI\"); using MPI; MPI.install_mpiexec()"
```

### メモリ不足エラー
```bash
# Julia起動時のメモリ制限
julia --heap-size-hint=4G test/runtests.jl unit
```

### 個別テストのデバッグ
```bash
# デバッグモードで実行
julia --track-allocation=user test/unit/test_mpi_initialization.jl
```

## 📈 継続的インテグレーション

### GitHub Actions設定例
```yaml
name: Parareal Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: julia-actions/setup-julia@v1
      - run: julia test/runtests.jl unit integration
```

### ローカル開発での推奨実行
```bash
# 開発中の基本チェック
julia test/runtests.jl unit

# プルリクエスト前の完全チェック
julia test/runtests.jl all
```

## 📚 参考資料

- [Julia Test.jl Documentation](https://docs.julialang.org/en/v1/stdlib/Test/)
- [MPI.jl Documentation](https://juliaparallel.github.io/MPI.jl/stable/)
- [Property-Based Testing in Julia](https://github.com/ssfrr/TestSetExtensions.jl)

## 🤝 貢献ガイドライン

新しいテストを追加する場合：

1. 適切なカテゴリディレクトリに配置
2. `test_*.jl`の命名規則に従う
3. `@testset`を使用してテストをグループ化
4. `runtests.jl`の該当カテゴリに追加
5. 十分なドキュメントとコメントを含める

テストの品質基準：
- 明確なテスト名と説明
- 独立性（他のテストに依存しない）
- 再現性（同じ結果を毎回生成）
- 適切なアサーション
- エラーケースのテスト