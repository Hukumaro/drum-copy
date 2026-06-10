# pipeline/stem_separation/

## 役割

**Step 1**: Demucs を使って入力楽曲からドラムトラック（stem）を分離し、wav ファイルとして `tmp/` に保存する。

## 構成ファイル（実装時に作成）

```
stem_separation/
├── __init__.py
└── separator.py    # メイン処理: separate(input_path, tmp_dir) -> Path
```

## 主要関数インターフェース

```python
def separate(input_path: Path, tmp_dir: Path) -> Path:
    """
    Returns:
        drums_wav_path: tmp_dir 以下に保存されたドラム stem wav のパス
    """
```

## 実装詳細

### 使用モデル

- デフォルト: `htdemucs`（Hybrid Transformer Demucs、精度と速度のバランスが良い）
- 変更する場合は `main.py` 側から引数として渡す

### Demucs 呼び出し方法

Demucs は Python API と CLI の両方を提供する。このモジュールでは **Python API** を使用する。

```python
from demucs.api import Separator

separator = Separator(model="htdemucs")
origin, separated = separator.separate_audio_file(input_path)
# separated["drums"] が torch.Tensor として取得できる
```

### ドラム wav の保存

- `torchaudio.save()` で `tmp/{入力ファイル名}/drums.wav` に書き出す
- サンプルレートは Demucs モデルのデフォルト（44100 Hz）をそのまま使用する

### GPU / CPU 切り替え

- `torch.cuda.is_available()` で自動判定する
- CPU フォールバック時は処理時間が大幅に増加するため、ログにその旨を出力する

## 依存ライブラリ

```
demucs
torch
torchaudio
```

## エラーケース

| 状況 | 対処 |
|------|------|
| 入力ファイルが存在しない | `FileNotFoundError` を raise（`main.py` でキャッチ） |
| 対応外の音声フォーマット | Demucs の例外をそのまま伝播させる |
| CUDA OOM | CPU にフォールバックして再試行する |
