#!/usr/bin/env python3
"""
並列化性能ベンチマーク結果の分析スクリプト
"""

import csv
import sys

def analyze_benchmark():
  # CSVファイルから結果を読み込む
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
  print("並列化性能ベンチマーク結果")
  print("=" * 80)
  print()
  print("測定条件:")
  print("  グリッドサイズ: 240x240x30")
  print("  ソルバー: PBiCGSTAB")
  print("  スムーザー: Gauss-Seidel")
  print("  収束判定: epsilon=1.0e-4")
  print("  解析種別: 定常解析 (is_steady=true)")
  print()
  print("=" * 80)
  print()

  # 結果テーブル
  print(f"{'モード':<12} | {'スレッド数':>10} | {'実行時間(秒)':>12} | {'スピードアップ':>12} | {'効率(%)':>10}")
  print("-" * 80)

  for r in results:
    speedup = sequential_time / r['time']
    efficiency = (speedup / r['threads']) * 100 if r['threads'] > 0 else 0
    mode_str = r['mode']
    threads_str = str(r['threads']) if r['mode'] == 'Parallel' else str(r['threads'])
    print(f"{mode_str:<12} | {threads_str:>10} | {r['time']:>12.3f} | {speedup:>12.2f}x | {efficiency:>9.1f}%")

  print()
  print("=" * 80)
  print()

  # 分析結果
  print("分析結果:")
  print()

  # 並列実行のみを抽出
  parallel_results = [r for r in results if r['mode'] == 'Parallel']

  # 最大スピードアップ
  max_speedup_result = max(parallel_results, key=lambda r: sequential_time / r['time'])
  max_speedup = sequential_time / max_speedup_result['time']
  print(f"1. 最大スピードアップ: {max_speedup:.2f}x ({max_speedup_result['threads']}スレッド)")

  # 2スレッドの効率
  thread2_result = next(r for r in parallel_results if r['threads'] == 2)
  thread2_speedup = sequential_time / thread2_result['time']
  thread2_efficiency = (thread2_speedup / 2) * 100
  print(f"2. 2スレッド並列化効率: {thread2_efficiency:.1f}%")

  # 3-6スレッドの飽和
  thread3_result = next(r for r in parallel_results if r['threads'] == 3)
  thread6_result = next(r for r in parallel_results if r['threads'] == 6)
  thread3_time = thread3_result['time']
  thread6_time = thread6_result['time']
  print(f"3. 3-6スレッド実行時間: {thread3_time:.1f}秒 → {thread6_time:.1f}秒 (ほぼ横ばい)")

  # 7-8スレッドでの改善
  thread7_result = next(r for r in parallel_results if r['threads'] == 7)
  thread8_result = next(r for r in parallel_results if r['threads'] == 8)
  thread7_speedup = sequential_time / thread7_result['time']
  thread8_speedup = sequential_time / thread8_result['time']
  print(f"4. 7-8スレッドでの追加改善: {thread7_speedup:.2f}x → {thread8_speedup:.2f}x")

  print()
  print("=" * 80)
  print()

  # 考察
  print("考察:")
  print()
  print("1. 優れた2スレッド並列化効率:")
  print(f"   2スレッドで{thread2_efficiency:.1f}%の効率を達成しており、")
  print("   並列化のオーバーヘッドが非常に小さいことを示しています。")
  print()
  print("2. 3-6スレッドでのスケーラビリティの飽和:")
  print("   3スレッド以降、実行時間が約60秒で横ばいとなっています。")
  print("   これは以下の要因が考えられます:")
  print("   - メモリバンド幅の制約")
  print("   - キャッシュ競合")
  print("   - 並列化されていない部分の影響（Amdahlの法則）")
  print()
  print("3. 7-8スレッドでの改善:")
  print("   7-8スレッドで再び性能が向上しており、")
  print("   一部の処理でさらに並列化の余地があることを示唆しています。")
  print()
  print("4. 推奨スレッド数:")
  print("   コストパフォーマンスの観点から、2-3スレッドが最適です。")
  print("   最大性能を求める場合は8スレッドを使用してください。")
  print()
  print("=" * 80)

if __name__ == '__main__':
  analyze_benchmark()
