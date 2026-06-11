# pipeline/

## 役割

パイプラインを構成する2つの処理ステップをモジュールとして格納するパッケージディレクトリ。

## 構成

```
pipeline/
├── __init__.py             # パッケージ初期化（空ファイルで可）
├── stem_separation/        # Step 1: Demucs によるドラム stem 分離
└── transcription/          # Step 2: MT3 によるドラム自動採譜
```

## 設計原則

- 各サブモジュールは **単一責務**：stem_separation はドラム wav を返すだけ、transcription は MIDI を書き出すだけ
- `main.py` がオーケストレーション（パスの受け渡し、エラーハンドリング、ログ出力）を担う
- 各モジュールは `main.py` 経由だけでなく、**単体でも import・実行できる**インターフェースを持つ

## モジュール間インターフェース

| モジュール | 入力 | 出力 |
|-----------|------|------|
| `stem_separation` | 入力音声ファイルパス（str/Path）, tmp 出力ディレクトリ | ドラム stem wav のパス（Path） |
| `transcription` | ドラム stem wav のパス（Path）, output ディレクトリ | 生成した MIDI ファイルのパス（Path） |
