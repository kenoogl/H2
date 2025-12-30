# トラブルシューティング・FAQ

## よくある問題と解決策

### 1. 収束関連の問題

#### 問題: Pararealが収束しない
**症状**:
```
Warning: Parareal failed to converge after 20 iterations
Residual norm: 1.234e-3 (tolerance: 1.0e-6)
```

**原因と解決策**:

1. **時間ステップ比率が大きすぎる**
   ```julia
   # 悪い例
   config.time_step_ratio = 200.0
   
   # 良い例
   config.time_step_ratio = 10.0  # まず小さい値から試す
   ```

2. **収束判定基準が厳しすぎる**
   ```julia
   # 緩い基準から始める
   config.convergence_tolerance = 1.0e-4  # 1.0e-6から変更
   config.max_iterations = 30             # 反復回数を増加
   ```

3. **問題の性質が時間並列化に不適**
   ```julia
   # 時間結合が強い問題の場合、逐次計算を使用
   result = q3d(NX, NY, NZ, parareal=false)
   ```

#### 問題: 収束が非常に遅い
**症状**: 反復回数が20回を超えても収束しない

**解決策**:
```julia
# 粗解法の精度を向上
config.coarse_spatial_resolution_factor = 0.8  # 0.5から0.8に変更
config.dt_coarse = config.dt_fine * 5.0        # 比率を小さく

# または適応的時間窓サイズを使用
config.adaptive_time_windows = true
```

### 2. 性能関連の問題

#### 問題: 逐次計算より遅い
**症状**: 
```
Sequential time: 120.5 seconds
Parareal time: 180.3 seconds
Speedup: 0.67
```

**原因と解決策**:

1. **問題サイズが小さすぎる**
   ```julia
   # 小規模問題では並列化オーバーヘッドが支配的
   # 解決策: より大きな問題サイズを使用するか、プロセス数を減らす
   config.n_mpi_processes = 2  # 4から2に変更
   ```

2. **通信オーバーヘッドが大きい**
   ```julia
   # 時間窓数を減らして通信頻度を削減
   config.n_time_windows = 2
   
   # または非同期通信を有効化
   config.enable_async_communication = true
   ```

3. **負荷分散が悪い**
   ```julia
   # 動的負荷分散を有効化
   config.enable_dynamic_load_balancing = true
   ```

#### 問題: スケーラビリティが悪い
**症状**: プロセス数を増やしても性能が向上しない

**解決策**:
```julia
# 強スケーリング限界の確認
function check_scalability_limit(problem_size, n_processes)
    work_per_process = problem_size / n_processes
    if work_per_process < 10000  # 経験的閾値
        println("Warning: Work per process too small for efficient scaling")
        return false
    end
    return true
end

# 弱スケーリングテストの実行
for np in [1, 2, 4, 8]
    grid_size = 32 * np  # プロセス数に比例してサイズ増加
    # テスト実行...
end
```

### 3. メモリ関連の問題

#### 問題: OutOfMemoryError
**症状**:
```
ERROR: OutOfMemoryError()
Stacktrace:
 [1] Array{Float64, 3}(::UndefInitializer, ::Tuple{Int64, Int64, Int64})
```

**解決策**:

1. **メモリ使用量の削減**
   ```julia
   # 粗解法の空間解像度を削減
   config.coarse_spatial_resolution_factor = 0.25  # より積極的に削減
   
   # データ型の変更
   config.precision = Float32  # Float64からFloat32に変更
   ```

2. **ガベージコレクションの調整**
   ```julia
   # Julia起動時の設定
   ENV["JULIA_GC_ALLOC_POOL"] = "3145728"  # 3MB
   ENV["JULIA_GC_ALLOC_OTHER"] = "1048576" # 1MB
   
   # 実行中の強制GC
   GC.gc()
   GC.gc(true)  # フルGC
   ```

3. **メモリ使用量の監視**
   ```julia
   using Pkg
   Pkg.add("MemPool")
   using MemPool
   
   # メモリ使用量の確認
   println("Memory usage: ", Base.gc_live_bytes() / 1024^3, " GB")
   ```

### 4. MPI関連の問題

#### 問題: MPI初期化エラー
**症状**:
```
ERROR: MPI not properly initialized
```

**解決策**:
```julia
# MPI.jlの再インストール
using Pkg
Pkg.rm("MPI")
Pkg.add("MPI")
using MPI
MPI.install_mpiexec()

# システムMPIとの互換性確認
MPI.versioninfo()
```

#### 問題: プロセス間通信エラー
**症状**:
```
MPI_ERR_COMM: Invalid communicator
```

**解決策**:
```bash
# ネットワーク設定の確認
ping node2  # 他ノードへの接続確認

# ファイアウォール設定
sudo ufw allow from 192.168.1.0/24

# MPI設定の調整
export OMPI_MCA_btl_tcp_if_include=eth0
export OMPI_MCA_oob_tcp_if_include=eth0
```

#### 問題: デッドロック
**症状**: プログラムが無限に待機状態

**デバッグ方法**:
```bash
# デバッグ情報の有効化
export OMPI_MCA_btl_base_verbose=10

# タイムアウト設定
mpirun --timeout 300 -np 4 julia parareal_example.jl

# プロセス状態の確認
ps aux | grep julia
kill -QUIT <pid>  # スタックトレースの出力
```

### 5. 数値精度の問題

#### 問題: 精度が期待値より悪い
**症状**:
```
L2 norm error: 1.234e-3 (expected: < 1.0e-6)
Max pointwise error: 5.678e-2
```

**解決策**:

1. **時間ステップサイズの調整**
   ```julia
   # より細かい時間ステップを使用
   config.dt_fine = config.dt_fine / 2.0
   config.dt_coarse = config.dt_coarse / 2.0
   ```

2. **ソルバー精度の向上**
   ```julia
   # より厳しい収束基準
   solver_tolerance = 1.0e-8  # 1.0e-6から変更
   
   # より高精度なソルバーを使用
   solver = "pbicgstab"  # "sor"から変更
   ```

3. **数値安定性の確認**
   ```julia
   # CFL条件の確認
   function check_cfl_condition(dt, dx, thermal_diffusivity)
       cfl = thermal_diffusivity * dt / dx^2
       if cfl > 0.5
           @warn "CFL condition violated: $cfl > 0.5"
           return false
       end
       return true
   end
   ```

## FAQ（よくある質問）

### Q1: Pararealはどのような問題に適していますか？

**A**: 以下の条件を満たす問題に適しています：

✅ **適している問題**:
- 長時間の非定常解析（時間ステップ数 > 1000）
- 拡散支配的な問題（熱伝導、拡散方程式）
- 時間方向の結合が比較的弱い問題
- 空間並列化だけでは不十分な大規模問題

❌ **適していない問題**:
- 短時間の解析（時間ステップ数 < 100）
- 対流支配的な問題（高レイノルズ数流体）
- 強い時間結合を持つ問題（カオス系）
- 小規模問題（格子点数 < 10万）

### Q2: 最適なパラメータ設定はどのように決めますか？

**A**: 段階的なアプローチを推奨します：

1. **初期設定**:
   ```julia
   config = PararealConfig(
       time_step_ratio=10.0,      # 保守的な値から開始
       n_time_windows=2,          # 少ないプロセス数から
       max_iterations=15,
       convergence_tolerance=1.0e-6
   )
   ```

2. **自動最適化の使用**:
   ```julia
   config.auto_optimize_parameters = true
   config.parameter_exploration_mode = true
   ```

3. **手動調整**:
   - 収束しない場合: `time_step_ratio`を小さく
   - 性能が出ない場合: `n_time_windows`を調整
   - 精度が悪い場合: `dt_fine`を小さく

### Q3: どの程度の高速化が期待できますか？

**A**: 問題とハードウェアに依存しますが、一般的な目安：

| プロセス数 | 理想的高速化 | 実際の高速化 | 効率 |
|-----------|-------------|-------------|------|
| 2         | 2.0x        | 1.4-1.8x    | 70-90% |
| 4         | 4.0x        | 2.4-3.2x    | 60-80% |
| 8         | 8.0x        | 3.2-5.6x    | 40-70% |
| 16        | 16.0x       | 4.8-9.6x    | 30-60% |

**影響要因**:
- 通信オーバーヘッド（10-30%）
- 負荷分散の不均衡（5-15%）
- Parareal収束特性（問題依存）

### Q4: 既存のコードをどの程度変更する必要がありますか？

**A**: 最小限の変更で済みます：

**変更前**:
```julia
result = q3d(64, 64, 32, 
            solver="pbicgstab", 
            epsilon=1.0e-6)
```

**変更後**:
```julia
config = PararealConfig(...)  # 設定追加
result = q3d(64, 64, 32, 
            solver="pbicgstab", 
            epsilon=1.0e-6,
            parareal=true,           # 追加
            parareal_config=config)  # 追加
```

### Q5: メモリ使用量はどの程度増加しますか？

**A**: 概算式：

```
追加メモリ = プロセス数 × 格子点数 × 8バイト × 2（粗解法・精密解法）
```

**例**: 64×64×32格子、4プロセスの場合
```
追加メモリ = 4 × (64×64×32) × 8 × 2 = 2.1 GB
```

**削減方法**:
- 粗解法の解像度削減: `coarse_spatial_resolution_factor = 0.5`
- データ型変更: `Float32`使用で半減
- 時間窓数の削減

### Q6: どのようなハードウェアが推奨されますか？

**A**: 推奨構成：

**最小構成**:
- CPU: 4コア以上
- メモリ: 8GB以上
- ネットワーク: Gigabit Ethernet

**推奨構成**:
- CPU: 16-32コア（複数ノード）
- メモリ: 32-64GB per ノード
- ネットワーク: InfiniBand または 10GbE
- ストレージ: 並列ファイルシステム

**HPC環境**:
- 複数ノード（2-8ノード）
- 高速インターコネクト
- 大容量メモリ（128GB+ per ノード）

### Q7: 他の並列化手法との比較は？

**A**: 各手法の特徴：

| 手法 | 適用範囲 | 実装難易度 | スケーラビリティ |
|------|----------|------------|------------------|
| 空間並列化 | 大規模格子 | 低 | 高（格子サイズ依存） |
| 時間並列化（Parareal） | 長時間計算 | 中 | 中（時間ステップ数依存） |
| ハイブリッド | 大規模・長時間 | 高 | 非常に高 |

**組み合わせの効果**:
- 空間のみ: 8-16倍高速化（限界あり）
- 時間のみ: 2-8倍高速化
- ハイブリッド: 16-128倍高速化（理論値）

### Q8: 精度の検証はどのように行いますか？

**A**: 段階的検証アプローチ：

1. **単体テスト**:
   ```julia
   # 既知解との比較
   analytical_solution = heat_analytical(x, y, z, t)
   numerical_solution = q3d_parareal(...)
   error = norm(analytical_solution - numerical_solution)
   ```

2. **逐次計算との比較**:
   ```julia
   config.validation_mode = true
   result = q3d(..., parareal_config=config)
   println("Relative error: ", result.validation_metrics.relative_error)
   ```

3. **収束性テスト**:
   ```julia
   # 時間ステップを半分にして結果を比較
   dt_values = [0.01, 0.005, 0.0025]
   errors = []
   for dt in dt_values
       result = q3d_parareal(dt_fine=dt)
       push!(errors, compute_error(result))
   end
   # 収束次数の確認
   ```

### Q9: トラブル時のサポートはどこで受けられますか？

**A**: サポートリソース：

1. **ドキュメント**:
   - ユーザーガイド
   - API リファレンス
   - サンプルコード

2. **コミュニティ**:
   - GitHub Issues
   - Julia Discourse
   - Stack Overflow

3. **問題報告時の情報**:
   ```julia
   # システム情報の収集
   using InteractiveUtils
   versioninfo()
   
   using MPI
   MPI.versioninfo()
   
   # エラー情報
   # - 完全なエラーメッセージ
   # - 使用した設定（PararealConfig）
   # - 実行環境（OS、MPI、Julia版）
   ```

### Q10: 将来の機能拡張予定は？

**A**: 開発ロードマップ：

**短期（3-6ヶ月）**:
- GPU対応（CUDA.jl統合）
- 適応的時間窓サイズ
- より多くのソルバー対応

**中期（6-12ヶ月）**:
- 多物理連成問題対応
- 機械学習による最適化
- 可視化ツールの強化

**長期（1年以上）**:
- 異種ハードウェア対応
- クラウド環境最適化
- 自動チューニング機能