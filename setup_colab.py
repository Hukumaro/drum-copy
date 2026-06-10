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


print("1/7  numpy をピン留め（2.x 非互換回避）...")
pip_run("numpy<2.0", flags=["-q"])

print("2/7  demucs / soundfile...")
pip_run("demucs", "soundfile", flags=["-q"])

print("3/7  setuptools / wheel（Python 3.12 の distutils 削除対策）...")
pip_run("setuptools<70", "wheel", flags=["-q"])

print("4/7  Cython（madmom ビルドに必要）...")
pip_run("Cython<3", flags=["-q"])

print("5/7  madmom をソースからビルド（Python 3.12 / Colab 互換パッチ適用）...")
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

print("6/7  omnizart をクローン & パッチ...")
src_dir = Path("/content/_omnizart_src")
if src_dir.exists():
    print("  既存ディレクトリを削除:", str(src_dir))
    shutil.rmtree(src_dir)
git_clone("https://github.com/Music-and-Culture-Technology-Lab/omnizart.git", src_dir)

# setup.cfg を削除（vamp/pyaudio 等の問題依存が含まれるため）
setup_cfg = src_dir / "setup.cfg"
if setup_cfg.exists():
    setup_cfg.unlink()

NL = chr(10)

# pyproject.toml を最小版に置換する。
# 元の pyproject.toml は setuptools-scm でバージョン解決しようとして
# --depth=1 clone 環境では失敗する。
# build-backend を明示しつつ [project] でメタデータを完結させることで
# --no-build-isolation でも setuptools 69.x が正常にメタデータを生成できる。
pyproject_lines = [
    "[build-system]",
    'requires = ["setuptools>=61", "wheel"]',
    'build-backend = "setuptools.build_meta"',
    "",
    "[project]",
    'name = "omnizart"',
    'version = "0.6.0"',
    "dependencies = [",
    '    "click>=7.1.2",',
    '    "jsonschema>=3.2.0",',
    '    "librosa>=0.8.0",',
    '    "madmom>=0.16.1",',
    '    "mir_eval>=0.6",',
    '    "pillow>=8.3.2",',
    '    "pretty_midi>=0.2.9",',
    '    "pyyaml>=5.3.1",',
    '    "urllib3>=1.26.4",',
    "]",
    "",
    "[project.scripts]",
    'omnizart = "omnizart.cli.cli:entry"',
]
(src_dir / "pyproject.toml").write_text(NL.join(pyproject_lines) + NL)

# setup.py はメタデータを pyproject.toml に一本化したため最小化
(src_dir / "setup.py").write_text("from setuptools import setup" + NL + "setup()" + NL)

print("7/7  omnizart をインストール...")
pip_run(str(src_dir), flags=["-q", "--no-build-isolation"])

print("✅ インストール完了")
