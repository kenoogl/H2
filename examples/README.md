# Parareal実行例・サンプル集

このディレクトリには、Heat3dsのParareal時間並列化機能を使用するための実行例とサンプルコードが含まれています。

## 📁 ファイル構成

### 実行例スクリプト
- **`basic_parareal_example.jl`** - 基本的なParareal実行例
- **`ic_thermal_analysis_example.jl`** - IC熱解析用の実用的な例
- **`parameter_optimization_example.jl`** - パラメータ最適化のデモ
- **`benchmark_problems.jl`** - 性能評価用ベンチマーク問題集

### 実行支援ツール
- **`run_examples.sh`** - 全ての例を簡単に実行するためのシェルスクリプト
- **`README.md`** - このファイル

## 🚀 クイックスタート

### 1. 基本的な実行
```bash
# 最も簡単な実行方法
./run_examples.sh basic

# プロセス数とスレッド数を指定
./run_examples.sh basic -p 8 -t 2
```

### 2. IC熱解析例
```bash
# IC熱解析の実行（推奨: 4プロセス以上）
./run_examples.sh ic_thermal -p 8 -t 4
```

### 3. パラメータ最適化
```bash
# 最適パラメータの探索（推奨: 8プロセス以上）
./run_examples.sh optimization -p 16 -t 2
```

### 4. ベンチマーク実行
```bash
# 全ベンチマーク問題の実行
./run_examples.sh benchmark -p 8

# 特定の問題のみ実行
mpirun -np 4 julia benchmark_problems.jl small
```

## 📋 詳細な実行例

### 基本的なParareal実行

**ファイル**: `basic_parareal_example.jl`

最もシンプルなParareal実行例です。中規模問題（64³格子）を使用して、Pararealの基本的な動作を確認できます。

```bash
# 4プロセスで実行
mpirun -np 4 julia basic_parareal_example.jl

# 8プロセス、各プロセス4スレッドで実行
export JULIA_NUM_THREADS=4
mpirun -np 8 julia basic_parareal_example.jl
```

**期待される出力**:
```
=== Basic Parareal Example ===
MPI processes: 4
Julia threads per process: 4
Problem size: 64 × 64 × 32 = 131072 grid points
...
=== Results ===
Parareal computation completed successfully!
Total execution time: 45.23 seconds
Parareal iterations: 8
Overall speedup: 2.34x
```

### IC熱解析例

**ファイル**: `ic_thermal_analysis_example.jl`

実際のIC（集積回路）パッケージの熱解析を模擬した実用的な例です。高解像度格子（100×100×20）と長時間解析（10秒）を使用します。

```bash
# IC熱解析の実行
mpirun -np 8 julia ic_thermal_analysis_example.jl
```

**特徴**:
- IC特有の熱特性（シリコンの熱拡散率）を考慮
- 高精度要求（収束判定基準: 1e-7）
- 自動パラメータ最適化
- 詳細な性能・精度レポート

### パラメータ最適化例

**ファイル**: `parameter_optimization_example.jl`

異なる時間ステップ比率、時間窓数の組み合わせを系統的にテストし、最適な設定を見つけます。

```bash
# パラメータ最適化の実行（時間がかかります）
mpirun -np 16 julia parameter_optimization_example.jl
```

**テスト対象**:
- 時間ステップ比率: 5, 10, 25, 50, 100
- 時間窓数: 2, 4, 8
- 各組み合わせでの性能・収束性評価

**出力ファイル**:
- `parareal_optimization_results_*.json` - 詳細な結果データ
- `optimal_parareal_config.jl` - 最適設定のJuliaコード

### ベンチマーク問題集

**ファイル**: `benchmark_problems.jl`

標準的なベンチマーク問題を使用してPararealの性能を評価します。

```bash
# 全ベンチマーク問題の実行
mpirun -np 8 julia benchmark_problems.jl all

# 特定の問題のみ実行
mpirun -np 4 julia benchmark_problems.jl small
mpirun -np 8 julia benchmark_problems.jl medium
mpirun -np 16 julia benchmark_problems.jl large
```

**利用可能な問題**:
- **small**: 小規模問題（32³格子）- アルゴリズム検証用
- **medium**: 中規模問題（64×64×32格子）- 性能評価用
- **large**: 大規模問題（128×64×32格子）- スケーラビリティテスト用
- **ic_package**: ICパッケージ熱解析（100×100×20格子）
- **high_aspect**: 高アスペクト比問題（128×128×8格子）

## ⚙️ 設定とカスタマイズ

### 環境変数

```bash
# Juliaスレッド数の設定
export JULIA_NUM_THREADS=8

# デバッグモードの有効化
export PARAREAL_DEBUG=1
export PARAREAL_LOG_LEVEL=DEBUG

# メモリ使用量の制限
export JULIA_GC_ALLOC_POOL=3145728  # 3MB
```

### MPI設定

```bash
# Open MPI設定
export OMPI_MCA_btl_tcp_if_include=eth0
export OMPI_MCA_oob_tcp_if_include=eth0

# Intel MPI設定
export I_MPI_FABRICS=tcp
export I_MPI_TCP_NETMASK=192.168.1.0/24
```

### カスタム問題の作成

基本テンプレート:
```julia
using MPI
using Heat3ds

# MPI初期化
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

# Parareal設定
config = PararealConfig(
    total_time=1.0,
    n_time_windows=size,
    dt_coarse=0.01,
    dt_fine=0.001,
    time_step_ratio=10.0,
    max_iterations=15,
    convergence_tolerance=1.0e-6,
    n_mpi_processes=size,
    n_threads_per_process=Threads.nthreads()
)

# 実行
result = q3d(64, 64, 32,
            solver="pbicgstab",
            parareal=true,
            parareal_config=config)

# MPI終了
MPI.Finalize()
```

## 📊 性能の目安

### 期待されるスピードアップ

| 問題サイズ | プロセス数 | 期待スピードアップ | 実際の範囲 |
|-----------|-----------|------------------|-----------|
| 小規模 (32³) | 2-4 | 1.5-2.5x | 1.2-2.0x |
| 中規模 (64³) | 4-8 | 2.5-5.0x | 2.0-4.0x |
| 大規模 (128³) | 8-16 | 4.0-10.0x | 3.0-8.0x |

### 推奨ハードウェア構成

**最小構成**:
- CPU: 4コア以上
- メモリ: 8GB以上
- ネットワーク: Gigabit Ethernet

**推奨構成**:
- CPU: 16-32コア（複数ノード）
- メモリ: 32-64GB per ノード
- ネットワーク: InfiniBand または 10GbE

## 🔧 トラブルシューティング

### よくある問題

1. **収束しない**
   ```julia
   # 時間ステップ比率を小さくする
   config.time_step_ratio = 5.0  # 10.0から変更
   ```

2. **性能が出ない**
   ```julia
   # プロセス数を減らす
   config.n_mpi_processes = 2  # 4から変更
   ```

3. **メモリ不足**
   ```julia
   # 粗解法の解像度を削減
   config.coarse_spatial_resolution_factor = 0.5
   ```

### デバッグ実行

```bash
# デバッグ情報付きで実行
export PARAREAL_DEBUG=1
mpirun -np 4 julia basic_parareal_example.jl

# Valgrindでメモリチェック
mpirun -np 2 valgrind --tool=memcheck julia basic_parareal_example.jl
```

## 📈 結果の解釈

### 性能メトリクス

- **Speedup**: 逐次実行に対する高速化倍率
- **Efficiency**: Speedup / プロセス数（理想値: 1.0）
- **Iterations**: Parareal収束に要した反復回数

### 精度メトリクス

- **L2 norm error**: 逐次計算との全体的な誤差
- **Max pointwise error**: 最大点別誤差
- **Relative error**: 相対誤差

### 判定基準

- **優秀**: Efficiency > 70%, L2 error < 1e-6
- **良好**: Efficiency > 50%, L2 error < 1e-5
- **要改善**: Efficiency < 50% または L2 error > 1e-4

## 📚 参考資料

- [Pararealユーザーガイド](../docs/parareal_user_guide.md)
- [MPI設定ガイド](../docs/mpi_setup_guide.md)
- [トラブルシューティング](../docs/troubleshooting_faq.md)
- Heat3ds公式ドキュメント

## 🤝 サポート

問題が発生した場合は、以下の情報を含めてお問い合わせください：

1. 使用したスクリプト名
2. 実行コマンド
3. エラーメッセージの全文
4. システム情報（OS、Julia版、MPI版）
5. 使用したParareal設定

```bash
# システム情報の収集
julia -e "using InteractiveUtils; versioninfo()"
mpirun --version
```