"""StemSplitter 安裝腳本 (Windows / macOS / Linux)

用法:
    python install.py          自動偵測 NVIDIA 顯示卡
    python install.py --cpu    強制安裝 CPU 版 (沒有 NVIDIA 顯示卡時)

會在專案資料夾建立 .venv 並安裝所有套件。
"""
import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
CUDA_INDEX = "https://download.pytorch.org/whl/cu126"


def venv_python():
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(*args):
    print("\n> " + " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True)


def main():
    ap = argparse.ArgumentParser(description="安裝 StemSplitter")
    ap.add_argument("--cpu", action="store_true", help="強制安裝 CPU 版 PyTorch")
    args = ap.parse_args()

    if sys.version_info < (3, 10):
        sys.exit(f"需要 Python 3.10 以上，目前是 {sys.version.split()[0]}")

    if not venv_python().exists():
        print(f"建立虛擬環境: {VENV}")
        venv.create(VENV, with_pip=True)
    py = venv_python()
    pip = [py, "-m", "pip", "install"]
    run(*pip, "--upgrade", "pip")

    # macOS 用 PyTorch 預設版本 (Apple Silicon 會自動使用 MPS 加速)
    cuda = not args.cpu and sys.platform != "darwin" and shutil.which("nvidia-smi") is not None
    if cuda:
        print("\n偵測到 NVIDIA 顯示卡，安裝 CUDA 版 PyTorch")
        run(*pip, "torch", "torchaudio", "--index-url", CUDA_INDEX)
    else:
        if sys.platform != "darwin":
            print("\n沒有偵測到 NVIDIA 顯示卡，安裝 CPU 版 PyTorch (會比較慢)")
        run(*pip, "torch", "torchaudio")

    run(*pip, f"audio-separator[{'gpu' if cuda else 'cpu'}]>=0.47")
    run(*pip, "-r", ROOT / "requirements.txt")

    print("\n安裝完成！使用方式:")
    print(f"  python {Path('split_stems.py')} 歌曲.mp3")


if __name__ == "__main__":
    main()
