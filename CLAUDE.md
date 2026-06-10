# drum-copy — Drum Extraction & Transcription Pipeline

## 概要

楽曲ファイル（wav / mp3）からドラムパートを自動的に分離し、MIDIファイルとして出力するPythonパイプライン。

```
入力音声 → [stem_separation] → ドラムwav → [transcription] → MIDI出力
```

## 実行環境

| 環境 | 方法 | 備考 |
|------|------|------|
| **Google Colab（推奨）** | `drum_copy_colab.ipynb` を開く | GPU (T4) が使える。入出力は Google Drive |
| ローカル PC | `python main.py` | GPU なしの場合は処理が遅い |

### Google Colab での使い方

1. このプロジェクトフォルダを `マイドライブ/drum-copy/` にアップロード
2. `drum_copy_colab.ipynb` を Google Colab で開く
3. ランタイムを **GPU（T4）** に設定
4. 上から順にセルを実行

## ディレクトリ構成

```
drum-copy/
├── drum_copy_colab.ipynb   # Google Colab 実行ノートブック（主要エントリーポイント）
├── main.py                 # ローカル CLI エントリーポイント
├── input/                  # 入力楽曲ファイルの置き場（wav / mp3）
├── output/                 # 最終出力 MIDI ファイルの保存先
├── tmp/                    # 中間ファイル（Demucs が生成するドラム stem wav）
├── backends/               # 実行環境の抽象化レイヤー
│   ├── base.py             #   PipelineBackend 抽象基底クラス
│   ├── local.py            #   ローカル実行（main.py が使用）
│   └── colab.py            #   Google Colab + Google Drive 実行
└── pipeline/
    ├── stem_separation/    # Step 1: Demucs によるドラム stem 分離モジュール
    └── transcription/      # Step 2: omnizart によるドラム自動採譜モジュール
```

## 処理フロー

1. `input/` に配置した楽曲ファイルを受け取る
2. **Step 1 – Stem 分離**: `pipeline/stem_separation/` の処理で Demucs を呼び出し、ドラムトラックを `tmp/` に一時保存
3. **Step 2 – 自動採譜**: `pipeline/transcription/` の処理で omnizart drum モジュールを呼び出し、`output/` に MIDI を保存
4. `tmp/` の中間ファイルは任意でクリーンアップ

## 設計方針

- omnizart の出力結果は加工せず、生データをそのまま MIDI として保存する
- 各ステップは独立したモジュールとして実装し、単体でも呼び出し可能にする
- パスはすべてエントリーポイントから引数として渡し、各モジュール内にハードコードしない
- **`backends/` レイヤー**で実行環境の差異（ローカル / Colab / 将来のサーバーレスGPU）を吸収する

### 将来のサーバーレスGPU対応（Modal / RunPod / Replicate など）

`backends/base.py` の `PipelineBackend` を継承して新バックエンドを実装する。
主な拡張ポイント:
- `setup()` でリモートプロバイダーへの認証・設定
- `get_paths()` でリモート環境からアクセス可能なストレージパスを返す
- 必要に応じて `separate()` / `transcribe()` の呼び出し自体をリモートジョブとしてディスパッチ

## 主要依存ライブラリ

| ライブラリ | 用途 |
|-----------|------|
| `demucs` | ドラム stem 分離 |
| `omnizart` | ドラム自動採譜 → MIDI 出力 |
| `torch` / `torchaudio` | Demucs の実行基盤 |
| `soundfile` | ドラム stem wav の書き込み |
| `librosa` | 音声ファイルの読み込み・前処理補助 |

## 前提確認事項

- Python 3.9 以上を推奨（omnizart の依存関係に注意）
- **Google Colab では GPU ランタイムを必ず有効化**（T4 で十分）
- ローカル実行で CUDA 対応 GPU がない場合、処理に数十分かかることがある
- omnizart は初回実行時にモデルファイルを `~/.omnizart/` にダウンロードする
- Colab では omnizart の `vamp` 依存を除去したパッチインストールが必要（ノートブックが自動で処理）
