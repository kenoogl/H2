#!/bin/bash

# 並列化性能ベンチマークを実行するスクリプト
# Sequential実行と1-8スレッド並列実行を測定

echo "================================================================================"
echo "並列化性能ベンチマーク測定開始"
echo "================================================================================"
echo ""

# 既存の結果ファイルを削除
if [ -f benchmark_results.csv ]; then
  rm benchmark_results.csv
  echo "既存のbenchmark_results.csvを削除しました"
fi

# Sequential実行
echo "◆ Sequential実行"
echo "--------------------------------------------------------------------------------"
julia benchmark_single.jl sequential
echo ""

# 1-8スレッド並列実行
for nthreads in 1 2 3 4 5 6 7 8; do
  echo "◆ ${nthreads}スレッド並列実行"
  echo "--------------------------------------------------------------------------------"
  julia -t ${nthreads} benchmark_single.jl thread
  echo ""
done

echo "================================================================================"
echo "ベンチマーク測定完了"
echo "================================================================================"
echo ""

# 結果を表示
if [ -f benchmark_results.csv ]; then
  echo "測定結果:"
  echo ""
  cat benchmark_results.csv
  echo ""

  # スピードアップを計算して表示（Pythonが利用可能な場合）
  if command -v python3 &> /dev/null; then
    python3 - << 'EOF'
import csv

# 結果を読み込む
results = []
with open('benchmark_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append({
            'mode': row['Mode'],
            'threads': int(row['Threads']),
            'time': float(row['Time(s)'])
        })

# Sequential実行時間を基準とする
sequential_time = next(r['time'] for r in results if r['mode'] == 'Sequential')

print("=" * 80)
print("性能比較（Sequential実行を基準としたスピードアップ）")
print("=" * 80)
print()
print(f"{'モード':<12} | {'スレッド数':<10} | {'実行時間(秒)':<12} | {'スピードアップ':<12}")
print("-" * 80)

for r in results:
    speedup = sequential_time / r['time']
    mode_str = r['mode']
    threads_str = str(r['threads']) if r['mode'] == 'Parallel' else '-'
    print(f"{mode_str:<12} | {threads_str:<10} | {r['time']:<12.3f} | {speedup:<12.2f}x")

print()
print("=" * 80)

# スピードアップの効率を計算（並列実行のみ）
print()
print("並列化効率（理想的なスピードアップに対する割合）")
print("=" * 80)
print()
print(f"{'スレッド数':<10} | {'スピードアップ':<12} | {'効率(%)':<12}")
print("-" * 80)

for r in results:
    if r['mode'] == 'Parallel':
        speedup = sequential_time / r['time']
        efficiency = (speedup / r['threads']) * 100
        print(f"{r['threads']:<10} | {speedup:<12.2f}x | {efficiency:<12.1f}%")

print()
print("=" * 80)
EOF
  fi
fi
