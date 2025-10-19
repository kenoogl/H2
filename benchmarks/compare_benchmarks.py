#!/usr/bin/env python3
"""
2回のベンチマーク結果を比較分析するスクリプト
"""

import csv

def load_results(filename):
  results = []
  with open(filename, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      results.append({
        'mode': row['Mode'],
        'threads': int(row['Threads']),
        'time': float(row['Time(s)'])
      })
  return results

def compare_benchmarks():
  print("=" * 80)
  print("並列化性能ベンチマーク結果比較（2回の測定）")
  print("=" * 80)
  print()

  # 2回の結果を読み込む
  run1 = load_results('benchmark_results_run1.csv')
  run2 = load_results('benchmark_results.csv')

  # Sequential時間を取得
  seq1 = next(r['time'] for r in run1 if r['mode'] == 'Sequential')
  seq2 = next(r['time'] for r in run2 if r['mode'] == 'Sequential')

  print("測定条件:")
  print("  グリッドサイズ: 240x240x30")
  print("  ソルバー: PBiCGSTAB + Gauss-Seidel")
  print("  収束判定: epsilon=1.0e-4")
  print("  解析種別: 定常解析")
  print()
  print("=" * 80)
  print()

  # 比較表
  print(f"{'モード':<12} | {'スレ':<4} | {'Run1 時間':<10} | {'Run2 時間':<10} | {'差分':<8} | {'Run2 速度':<10}")
  print("-" * 80)

  for i, r1 in enumerate(run1):
    r2 = run2[i]
    diff = r2['time'] - r1['time']
    speedup2 = seq2 / r2['time']
    mode_str = r1['mode']
    threads_str = str(r1['threads'])

    print(f"{mode_str:<12} | {threads_str:<4} | {r1['time']:>10.3f}s | {r2['time']:>10.3f}s | {diff:>+7.2f}s | {speedup2:>9.2f}x")

  print()
  print("=" * 80)
  print()

  # 統計分析
  print("統計分析:")
  print()

  # Sequential時間の比較
  seq_diff = seq2 - seq1
  seq_diff_pct = (seq_diff / seq1) * 100
  print(f"1. Sequential実行時間:")
  print(f"   Run1: {seq1:.3f}秒")
  print(f"   Run2: {seq2:.3f}秒")
  print(f"   差分: {seq_diff:+.3f}秒 ({seq_diff_pct:+.1f}%)")
  print()

  # 並列実行の再現性
  parallel1 = [r for r in run1 if r['mode'] == 'Parallel']
  parallel2 = [r for r in run2 if r['mode'] == 'Parallel']

  print("2. 並列実行の再現性:")
  print()
  max_diff_threads = 0
  max_diff_value = 0

  for i, (p1, p2) in enumerate(zip(parallel1, parallel2)):
    diff = abs(p2['time'] - p1['time'])
    diff_pct = (diff / p1['time']) * 100

    if diff > max_diff_value:
      max_diff_value = diff
      max_diff_threads = p1['threads']

    print(f"   {p1['threads']}スレッド: {p1['time']:.3f}s → {p2['time']:.3f}s (差分: {diff:.3f}s, {diff_pct:.1f}%)")

  print()
  print(f"   最大差分: {max_diff_threads}スレッドで{max_diff_value:.3f}秒")
  print()

  # 平均スピードアップ
  print("3. 平均スピードアップ（2回の平均）:")
  print()

  for i, (p1, p2) in enumerate(zip(parallel1, parallel2)):
    speedup1 = seq1 / p1['time']
    speedup2 = seq2 / p2['time']
    avg_speedup = (speedup1 + speedup2) / 2

    print(f"   {p1['threads']}スレッド: {avg_speedup:.2f}x")

  print()
  print("=" * 80)
  print()

  # 最終分析
  print("総合分析:")
  print()

  # 8スレッドの平均スピードアップ
  p8_1 = next(r for r in parallel1 if r['threads'] == 8)
  p8_2 = next(r for r in parallel2 if r['threads'] == 8)
  speedup8_1 = seq1 / p8_1['time']
  speedup8_2 = seq2 / p8_2['time']
  avg_speedup8 = (speedup8_1 + speedup8_2) / 2

  print(f"1. 最大スピードアップ（8スレッド）: {avg_speedup8:.2f}x")
  print()

  # 2スレッドの効率
  p2_1 = next(r for r in parallel1 if r['threads'] == 2)
  p2_2 = next(r for r in parallel2 if r['threads'] == 2)
  speedup2_1 = seq1 / p2_1['time']
  speedup2_2 = seq2 / p2_2['time']
  avg_speedup2 = (speedup2_1 + speedup2_2) / 2
  efficiency2 = (avg_speedup2 / 2) * 100

  print(f"2. 2スレッド並列化効率: {efficiency2:.1f}%")
  print()

  # 3-6スレッドの挙動
  p3_2 = next(r for r in parallel2 if r['threads'] == 3)
  p6_2 = next(r for r in parallel2 if r['threads'] == 6)

  print(f"3. 3-6スレッドの挙動:")
  print(f"   Run2での時間: {p3_2['time']:.1f}秒 → {p6_2['time']:.1f}秒")
  print(f"   3スレッドでほぼピークに達し、その後は緩やかな改善")
  print()

  # 推奨設定
  print("4. 推奨スレッド数:")
  print(f"   - コストパフォーマンス重視: 2-3スレッド（効率{efficiency2:.0f}%）")
  print(f"   - 最大性能重視: 8スレッド（{avg_speedup8:.1f}倍高速）")
  print()

  print("=" * 80)

if __name__ == '__main__':
  compare_benchmarks()
