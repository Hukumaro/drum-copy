"""
Google Colab 環境向けパッケージインストールスクリプト。
drum_copy_colab.ipynb のインストールセルから呼び出される。

このファイルをリポジトリで管理することで、インストールロジックの変更を
GitHub push だけで Colab に反映できる（ノートブック自体を差し替え不要）。
"""
import subprocess, sys, shutil
from pathlib import Path


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


def git_clone(url, dest):
    r = subprocess.run(["git", "clone", "--depth=1", url, str(dest)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError(
            f"git clone 失敗 (exit {r.returncode}): {url}\n"
            "リポジトリが存在しないか、ネットワークエラーの可能性があります。"
        )


print("1/6  numpy をピン留め（2.x 非互換回避）...")
pip_run("numpy<2.0", flags=["-q"])

print("2/6  demucs...")
pip_run("demucs", flags=["-q"])

print("3/6  setuptools / wheel（Python 3.12 の distutils 削除対策）...")
pip_run("setuptools<70", "wheel", flags=["-q"])

print("4/6  Cython（madmom ビルドに必要）...")
pip_run("Cython<3", flags=["-q"])

print("5/6  madmom をソースからビルド（Python 3.12 / Colab 互換パッチ適用）...")
_madmom_src = Path("/content/_madmom_src")
if _madmom_src.exists():
    shutil.rmtree(_madmom_src)
git_clone("https://github.com/CPJKU/madmom.git", _madmom_src)
# Python 3.12 で削除された distutils を setuptools で置換
_setup_py = _madmom_src / "setup.py"
_txt = _setup_py.read_text()
_txt = _txt.replace("from distutils.core import",               "from setuptools import")
_txt = _txt.replace("from distutils.extension import",          "from setuptools.extension import")
_txt = _txt.replace("from distutils.command.build_ext import",  "from setuptools.command.build_ext import")
_setup_py.write_text(_txt)
pip_run(str(_madmom_src), flags=["-q", "--no-build-isolation"])

print("6/7  omnizart をクローン...")
src_dir = Path("/content/_omnizart_src")
if src_dir.exists():
    print("  既存ディレクトリを削除:", str(src_dir))
    shutil.rmtree(src_dir)
git_clone("https://github.com/Music-and-Culture-Technology-Lab/omnizart.git", src_dir)

print("7/7  omnizart の依存パッケージをインストール...")
# omnizart 本体は pip install（ビルド）せず sys.path に追加する。
# pyproject.toml / setup.py のビルドシステム互換性問題を完全に回避できる。
# omnizart はピュア Python なのでビルド不要でインポート可能。
pip_run(
    "click>=7.1.2", "jsonschema>=3.2.0", "mir_eval>=0.6",
    "pillow>=8.3.2", "pretty_midi>=0.2.9", "pyyaml>=5.3.1",
    "urllib3>=1.26.4",
    flags=["-q"],
)
sys.path.insert(0, str(src_dir))

print("✅ インストール完了")
