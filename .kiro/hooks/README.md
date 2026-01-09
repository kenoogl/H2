# Kiro Git Hooks for Parareal Project

このディレクトリには、Parareal時間並列化プロジェクトのタスク完了時に自動的にGitコミット・プッシュを行うhookが含まれています。

## ファイル構成

- `hooks.json` - Kiro IDE用のhook設定ファイル
- `task-completion-git-hook.json` - タスク完了トリガー用の設定
- `auto-commit-task.sh` - メインの自動コミットスクリプト
- `generate-commit-message.sh` - 詳細なコミットメッセージ生成スクリプト

## 機能

### 自動コミット・プッシュ
タスクが完了状態にマークされると、以下の処理が自動実行されます：

1. **変更のステージング**: `git add .`
2. **コミット**: タスク情報を含む詳細なコミットメッセージで実行
3. **プッシュ**: 現在のブランチにプッシュ

### コミットメッセージ形式
```
Complete Task X.Y: タスク名

タスクの詳細説明
- 実装内容1
- 実装内容2

Changed files:
  - src/parareal.jl
  - test/test_*.jl

Auto-committed by Kiro hook
Spec: parareal-time-parallelization
Branch: feature/parareal-time-parallelization
Timestamp: 2024-12-16 15:30:45
```

## 使用方法

### 自動実行
Kiro IDEでタスクを完了状態にマークすると自動的に実行されます。

### 手動実行
```bash
# 基本的な使用方法
.kiro/hooks/auto-commit-task.sh "タスク名" "2.1" "parareal-time-parallelization"

# コミットメッセージのプレビュー
.kiro/hooks/generate-commit-message.sh "2.1" "Create hybrid parallelization coordinator"
```

## 設定

### hookの有効/無効
`hooks.json`の`enabled`フィールドで制御：
```json
{
  "settings": {
    "enabled": true  // false にすると無効化
  }
}
```

### タイムアウト設定
デフォルト60秒、必要に応じて調整可能：
```json
{
  "action": {
    "timeout": 60
  }
}
```

## トラブルシューティング

### よくある問題

1. **権限エラー**
   ```bash
   chmod +x .kiro/hooks/*.sh
   ```

2. **Gitリモートへのプッシュ失敗**
   - SSH鍵の設定を確認
   - リモートリポジトリへのアクセス権限を確認

3. **ログの確認**
   ```bash
   tail -f .kiro/hooks/hooks.log
   ```

### デバッグモード
スクリプトの先頭に`set -x`を追加すると詳細なログが出力されます。

## カスタマイズ

### コミットメッセージのカスタマイズ
`generate-commit-message.sh`を編集してメッセージ形式を変更できます。

### 追加のhook
新しいhookを追加する場合は`hooks.json`に設定を追加してください。

## セキュリティ注意事項

- hookスクリプトは実行権限を持つため、内容を十分確認してから使用してください
- 自動プッシュ機能により、意図しない変更がリモートに送信される可能性があります
- 機密情報が含まれる場合は、hookを無効化することを推奨します