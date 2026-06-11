# pipeline/transcription/

## 役割

**Step 2**: Google MT3 (Multi-Task Music Transcription) を使ってドラム stem wav を解析し、MIDI ファイルとして `output/` に保存する。

## 構成ファイル

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

### MT3 推論フロー

```python
import librosa
import note_seq
from mt3 import metrics_utils, note_sequences, spectrograms, vocabularies

# 1. 16 kHz でリサンプリング
audio, _ = librosa.load(wav_path, sr=16000, mono=True)

# 2. メルスペクトログラムを 512 フレーム (~4 秒) 単位に分割
spectrogram = spectrograms.compute_spectrogram(audio, spectrogram_config)

# 3. T5X モデル（JAX）でトークン列を推論
tokens, _ = predict_fn(params, {"encoder_input_tokens": batch, ...}, None)

# 4. トークン → NoteSequence → MIDI
result = metrics_utils.event_predictions_to_ns(predictions, codec=codec,
    encoding_spec=note_sequences.NoteEncodingWithTiesSpec)
note_seq.sequence_proto_to_midi_file(result["est_ns"], str(midi_path))
```

### モデルキャッシュ

- モジュールレベルの `_cached_model` / `_cached_ckpt` によりプロセス内での再ロードを回避する
- チェックポイントパスは `MT3_CHECKPOINT` 環境変数で変更可能（デフォルト: `/tmp/mt3/mt3/`）

### 出力ファイル

- MT3 が生成した MIDI データを **そのまま** `output/` に保存する
- ノートの丸め・量子化、特定打楽器へのフィルタリング、ベロシティ補正などの後処理は一切行わない
- ファイル名は `{入力楽曲名}.mid` とする（`main.py` でリネームを担当）

### チェックポイントのダウンロード

```bash
gsutil -q -m cp -r gs://mt3/checkpoints/mt3/ /tmp/mt3/
```

Colab ノートブックのセル 4 が自動で実行する。ランタイムを再起動するたびに再実行が必要（`/tmp/` は揮発性）。

## 依存ライブラリ

```
mt3          # git+https://github.com/magenta/mt3
t5x          # MT3 の T5X フレームワーク（mt3 依存として自動インストール）
seqio        # データパイプライン（mt3 依存として自動インストール）
note-seq     # NoteSequence ↔ MIDI 変換（mt3 依存として自動インストール）
jax[cuda12_pip]   # MT3 の JAX バックエンド
```

## 注意事項

- MT3 は JAX ベース、Demucs は PyTorch ベース。同一環境に共存できるが CUDA バージョンの整合性に注意
- チェックポイントは約 400 MB。Colab の `/tmp/` は揮発性なため、ランタイム再起動後に再ダウンロードが必要

## エラーケース

| 状況 | 対処 |
|------|------|
| ドラム wav が存在しない | `FileNotFoundError` を raise（`main.py` でキャッチ） |
| MT3 依存未インストール | `ImportError` にインストールコマンドを付けて raise |
| チェックポイント未ダウンロード | T5X がパスを見つけられず例外を送出（セル 4 を再実行） |
