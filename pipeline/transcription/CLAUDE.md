# pipeline/transcription/

## 役割

**Step 2**: omnizart の drum モジュールを使ってドラム stem wav を解析し、MIDI ファイルとして `output/` に保存する。

## 構成ファイル（実装時に作成）

```
transcription/
├── __init__.py
└── transcriber.py    # メイン処理: transcribe(drums_wav_path, output_dir) -> Path
```

## 主要関数インターフェース

```python
def transcribe(drums_wav_path: Path, output_dir: Path) -> Path:
    """
    Returns:
        midi_path: output_dir 以下に保存された MIDI ファイルのパス
    """
```

## 実装詳細

### omnizart 呼び出し方法

omnizart は Python API を提供している。drum モジュールを直接呼び出す。

```python
from omnizart.drum import app as drum_app

midi_path = drum_app.transcribe(str(drums_wav_path), output=str(output_dir))
```

### 出力ファイル

- omnizart が生成した MIDI データを **そのまま** `output/` に保存する
- ノートの丸め・量子化、特定打楽器へのフィルタリング、ベロシティ補正などの後処理は一切行わない
- ファイル名は `{入力楽曲名}.mid` とする（`main.py` でリネームを担当）

### モデルダウンロード

- omnizart は初回実行時に自動的にモデルファイルを `~/.omnizart/` にダウンロードする
- オフライン環境では事前に `omnizart application download-models drum` を実行しておく必要がある

## 依存ライブラリ

```
omnizart
```

omnizart は内部で以下を使用するが、通常は omnizart インストール時に解決される:

```
tensorflow (>=2.x)
librosa
pretty_midi
```

## 注意事項

- omnizart の依存する TensorFlow と Demucs の依存する PyTorch は同一環境に共存できるが、
  CUDA バージョンの整合性に注意すること
- omnizart の出力パスが引数の `output_dir` と異なる場合（内部でリネームが発生する場合）は、
  `transcriber.py` 内で実際の出力ファイルを glob で特定して返す

## エラーケース

| 状況 | 対処 |
|------|------|
| ドラム wav が存在しない | `FileNotFoundError` を raise（`main.py` でキャッチ） |
| モデルファイル未ダウンロード | omnizart の例外をそのまま伝播させ、ダウンロードコマンドをログに出力 |
| MIDI ファイルが生成されなかった | `RuntimeError` を raise |
