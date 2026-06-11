"""
Google Colab 環境向けパッケージインストールスクリプト。
drum_copy_colab.ipynb のインストールセルから呼び出される。

このファイルをリポジトリで管理することで、インストールロジックの変更を
GitHub push だけで Colab に反映できる（ノートブック自体を差し替え不要）。
"""
import subprocess, sys, os


def pip_run(*pkgs, flags=None):
    cmd = [sys.executable, "-m", "pip", "install"] + list(pkgs) + (flags or [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        out = r.stdout + r.stderr
        print(out[:3000])
        if len(out) > 3000:
            print("...[中略]...")
            print(out[-2000:])
        raise subprocess.CalledProcessError(r.returncode, cmd)


print("1/4  JAX (CUDA 12)...")
pip_run(
    "jax[cuda12_pip]",
    "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    flags=["-q"],
)

print("2/4  MT3 (t5x / seqio / note-seq を含む)...")
pip_run("git+https://github.com/magenta/mt3", flags=["-q"])

print("3/4  demucs, librosa, soundfile...")
pip_run("demucs", "librosa", "soundfile", flags=["-q"])

print("4/4  MT3 チェックポイントをダウンロード...")
os.makedirs("/tmp/mt3", exist_ok=True)
r = subprocess.run(
    ["gsutil", "-q", "-m", "cp", "-r", "gs://mt3/checkpoints/mt3/", "/tmp/mt3/"],
    capture_output=True, text=True,
)
if r.returncode != 0:
    print(f"⚠️  gsutil エラー: {r.stderr.strip()}")
    print("  手動でダウンロードするか MT3_CHECKPOINT 環境変数でパスを指定してください。")
else:
    print("  ✓ チェックポイント保存先: /tmp/mt3/mt3/")

print("✅ インストール完了")
