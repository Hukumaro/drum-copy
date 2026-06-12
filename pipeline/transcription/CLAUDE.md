# pipeline/transcription/

## 役割

**Step 2**: YourMT3+ (mimbres/YourMT3) を使ってドラム stem wav を解析し、MIDI ファイルとして `output/` に保存する。

## 構成ファイル

```
transcription/
├── __init__.py
├── transcriber.py       # ドラム採譜: transcribe(wav_path, output_dir, stem_name) -> Path
└── bass_transcriber.py  # ベース採譜: transcribe_bass(wav_path, output_dir, stem_name) -> Path
```

## 主要関数インターフェース

```python
def transcribe(wav_path: Path, output_dir: Path, stem_name: str = "") -> Path:
    """
    Returns:
        midi_path: output_dir 以下に保存された MIDI ファイルのパス
    """
```

## 実装詳細

### YourMT3+ 推論フロー

```
ドラム stem wav
  → torchaudio.info() でメタデータ取得
  → model_helper.transcribe(model, audio_info)  # 全楽器 MIDI を生成
  → _extract_drum_channel()                      # MIDI channel 9 のみ抽出
  → output/{stem_name}.mid
```

### YourMT3+ のセットアップ

Colab セットアップセルが `/tmp/ymt3/` に HuggingFace Spaces リポジトリをクローンする。
モデルウェイト（~600 MB）は初回 `load_model_checkpoint()` 呼び出し時に HuggingFace から自動ダウンロード。

```bash
git clone --depth=1 https://huggingface.co/spaces/mimbres/YourMT3 /tmp/ymt3
pip install -r /tmp/ymt3/requirements.txt
```

クローン先のオーバーライド: `export YMT3_DIR=/path/to/ymt3`

### 使用チェックポイント

`YPTF.MoE+ Multi (noPS)` — 13チャンネル Mixture-of-Experts モデル。
AMT ベンチマーク（ENST Drums を含む）で最高精度。

```
mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt
```

### ドラムチャンネル抽出

YourMT3+ は全楽器を含む多チャンネル MIDI を出力する。
`_extract_drum_channel()` が MIDI channel 9（GM パーカッション標準）のノートのみを保持し、
他のノートイベントを除去する。channel 9 のノートが見つからない場合はフル MIDI をそのまま保存する。

### モデルキャッシュ

モジュールレベルの `_cached_model` によりプロセス内での再ロードを回避する。

### 出力ファイル

- ファイル名: `{wav_stem}_{stem_name}.mid`（stem_name が空の場合は `{wav_stem}.mid`）
- YourMT3+ が生成した MIDI の channel 9 を抽出して保存（後処理なし）

## ベース採譜（bass_transcriber.py）

Basic Pitch（Spotify）を使用。v0.3.0+ は ONNX Runtime ベースで TensorFlow 不要。

```python
def transcribe_bass(wav_path: Path, output_dir: Path, stem_name: str = "bass") -> Path:
```

- `minimum_frequency=30.0 Hz`（B0以下のサブベースノイズを除去）
- `maximum_frequency`は無制限（Demucs でステム分離済みのため）
- 出力: `pretty_midi.PrettyMIDI.write()` でファイル保存

インストール: `pip install 'basic-pitch[onnx]'`

## stem → transcriber 対応（ノートブック cell 15 の _dispatch）

| stem | transcriber |
|------|------------|
| `"drums"` | `transcriber.transcribe()` — YourMT3+ |
| `"bass"` | `bass_transcriber.transcribe_bass()` — Basic Pitch |

## 依存ライブラリ

```
torch / torchaudio    # YourMT3+ 実行基盤（Demucs と共有、同一 venv で動作）
mido                  # MIDI channel フィルタリング（YourMT3+ requirements.txt に含まれる）
basic-pitch[onnx]     # ベース採譜（ONNX Runtime ベース、TF 不要）
pretty_midi           # basic-pitch 依存、MIDI 保存に使用
```

**注意**: すべて PyTorch / ONNX ベース。Demucs と同一 venv で動作し、TF ABI ハックは不要。

## エラーケース

| 状況 | 対処 |
|------|------|
| ドラム wav が存在しない | `FileNotFoundError` を raise |
| YourMT3+ リポジトリ未クローン | `_ensure_ymt3()` が自動クローンを試みる |
| モデルウェイト未ダウンロード | `load_model_checkpoint()` が HuggingFace から自動取得 |
| channel 9 ノートなし | フル MIDI を保存（ワーニングログを出力）|
