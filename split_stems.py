"""音檔樂器分軌工具

用法:
    python split_stems.py 歌曲.mp3 [更多檔案...] [-o 輸出資料夾] [--model demucs] [--guitars electric]

流程:
  1. BS-Roformer-SW (或 Demucs htdemucs_6s) 拆成 鼓 / Bass / 吉他 / 鋼琴 / 人聲 / 其他
  2. 吉他軌依左右定位拆成多個聲部，再用 CLAP 判斷每段是木吉他還是電吉他
  3. 「其他」軌同樣依定位拆開，用 CLAP 分出合成器，剩下歸為其他
  4. 偵測每軌是否真的有聲音，沒聲音的不輸出
輸出: <原檔名> - Stems/<原檔名> - <樂器>.wav
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _use_venv():
    """用系統 Python 執行時，自動改用專案的 .venv 重新執行，使用者不需要先 activate。"""
    venv = ROOT / ".venv"
    py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if py.exists() and Path(sys.prefix).resolve() != venv.resolve():
        sys.exit(subprocess.call([str(py), *sys.argv]))


_use_venv()

# 模型、快取與暫存檔的位置，可用環境變數 STEMSPLIT_DATA 改到其他磁碟
DATA_DIR = Path(os.environ.get("STEMSPLIT_DATA", ROOT / ".cache"))
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))
# CLAP 模型已下載過就直接用本機快取，不再連線 Hugging Face
if (DATA_DIR / "hf" / "hub" / "models--laion--clap-htsat-unfused").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import static_ffmpeg  # noqa: E402

static_ffmpeg.add_paths(weak=True)  # 系統已安裝 ffmpeg 就用系統的，沒有才下載
from audio_separator.separator import Separator  # noqa: E402  (需要先有 ffmpeg)

import clap_route  # noqa: E402
import pan_split  # noqa: E402

MODELS = {
    "roformer": "BS-Roformer-SW.ckpt",  # 品質較好
    "demucs": "htdemucs_6s.yaml",       # 較快
}

SILENCE_DB = -40.0       # 相對原曲音量低於此值視為沒有這個樂器
MIN_ACTIVE_RATIO = 0.03  # 至少 3% 時間有聲音


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)


def is_present(stem, mix, sr):
    """以音量和有聲時間比例判斷這軌是否真的有樂器。"""
    if rms_db(stem) - rms_db(mix) < SILENCE_DB:
        return False
    frame = sr // 10
    mono = stem.mean(axis=0)
    n = len(mono) // frame
    if n == 0:
        return False
    frames = mono[: n * frame].reshape(n, frame)
    frame_db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-12)
    return (frame_db > rms_db(mix) - 35).mean() >= MIN_ACTIVE_RATIO


def load(path):
    data, sr = sf.read(path, always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data.T.astype(np.float32), sr


def separate(sep, model, path, workdir):
    """跑分軌模型，回傳 {stem 名稱(小寫): 音訊}。"""
    workdir.mkdir(parents=True, exist_ok=True)
    sep.output_dir = str(workdir)
    sep.load_model(model_filename=model)
    result = {}
    for f in sep.separate(str(path)):
        f = Path(f) if Path(f).is_absolute() else workdir / f
        # audio-separator 檔名格式: <輸入>_(Stem)_<model>.wav
        stem = f.stem.split("_(")[-1].split(")")[0].lower()
        result[stem] = load(f)[0]
    return result


def pick(stems, *keys):
    for k in keys:
        for name, audio in stems.items():
            if k in name:
                return audio
    raise KeyError(f"模型輸出中找不到 {keys}，實際有: {list(stems)}")


def split_sources(audio, sr, mix):
    """依左右定位拆聲部，回傳 [(定位, 音訊), ...]。"""
    positions, cache = pan_split.find_sources(audio, sr)
    if len(positions) <= 1:
        return [(positions[0] if positions else 0.0, audio)]
    parts = pan_split.split_by_pan(audio, sr, positions, cache)
    keep = [is_present(a, mix, sr) for a in parts]
    if not any(keep):
        return [(0.0, audio)]
    kept = [(p, a.copy()) for p, a, k in zip(positions, parts, keep) if k]
    # 太小聲而被捨棄的聲部併入定位最接近的聲部，避免聲音憑空消失
    for p, a, k in zip(positions, parts, keep):
        if not k:
            nearest = min(range(len(kept)), key=lambda i: abs(kept[i][0] - p))
            kept[nearest][1][:] += a
    return kept


def process(src: Path, out_root: Path, sep, model: str, guitars: str = "auto"):
    name = src.stem
    print(f"\n=== {src.name} ===")
    (DATA_DIR / "tmp").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="stems_", dir=DATA_DIR / "tmp"))
    try:
        return _process(src, out_root / f"{name} - Stems", name, work, sep, model, guitars)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _process(src, out_dir, name, work, sep, model, guitars):
    # 統一轉成 44.1kHz 立體聲 WAV
    mix_wav = work / "mix.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-ac", "2", "-ar", "44100", str(mix_wav)], check=True)
    mix, sr = load(mix_wav)

    print(f"[1/3] 主分軌 ({MODELS[model]})...")
    stems = separate(sep, MODELS[model], mix_wav, work / "main")
    tracks = {
        "Drums": pick(stems, "drum"),
        "Bass": pick(stems, "bass"),
        "Piano": pick(stems, "piano"),
        "Vocals": pick(stems, "vocal"),
    }
    guitar = pick(stems, "guitar")
    other = pick(stems, "other")

    print("[2/3] 吉他: 拆聲部 + 判斷木吉他/電吉他...")
    acoustic = np.zeros_like(mix)
    electrics = []
    if is_present(guitar, mix, sr):
        for pos, part in split_sources(guitar, sr, mix):
            if guitars == "auto":
                ac, el, ratio = clap_route.route(part, sr, clap_route.GUITAR_LABELS, "acoustic")
            elif guitars == "acoustic":
                ac, el, ratio = part, np.zeros_like(part), 1.0
            else:
                ac, el, ratio = np.zeros_like(part), part, 0.0
            print(f"      聲部({pan_split.position_label(pos)}): 木吉他 {ratio:.0%} / 電吉他 {1 - ratio:.0%}")
            acoustic += ac
            if is_present(el, mix, sr):
                electrics.append((pos, el))
    tracks["Acoustic Guitar"] = acoustic
    if len(electrics) == 1:
        tracks["Electric Guitar"] = electrics[0][1]
    for i, (pos, el) in enumerate(electrics if len(electrics) > 1 else [], 1):
        tracks[f"Electric Guitar {i} ({pan_split.position_label(pos)})"] = el

    print("[3/3] 其他: 分出合成器...")
    synth = np.zeros_like(mix)
    rest = np.zeros_like(mix)
    if is_present(other, mix, sr):
        for pos, part in split_sources(other, sr, mix):
            sy, ot, ratio = clap_route.route(part, sr, clap_route.OTHER_LABELS, "synth")
            print(f"      聲部({pan_split.position_label(pos)}): 合成器 {ratio:.0%} / 其他 {1 - ratio:.0%}")
            synth += sy
            rest += ot
    tracks["Synth"] = synth
    tracks["Other"] = rest

    print("\n分析結果 (音量相對原曲):")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for label, audio in tracks.items():
        present = is_present(audio, mix, sr)
        note = ""
        if label == "Other" and present:
            note = "  可能包含: " + "、".join(n for n, _ in clap_route.describe(audio, sr))
        print(f"  {'有' if present else '無'}  {label:<28} {rms_db(audio) - rms_db(mix):6.1f} dB{note}")
        if present:
            peak = np.abs(audio).max()
            if peak > 1.0:  # 24-bit WAV 超過 0 dBFS 會破音，等比例壓回來
                print(f"      注意: {label} 峰值超過 0 dBFS，已降低 {20 * np.log10(peak):.1f} dB 避免破音")
                audio = audio / peak
            path = out_dir / f"{name} - {label}.wav"
            sf.write(path, audio.T, sr, subtype="PCM_24")
            written.append(path)

    print(f"\n輸出 {len(written)} 軌到: {out_dir}")
    for p in written:
        print(f"  {p.name}")
    return written


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="音檔樂器分軌工具")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None, help="輸出資料夾 (預設: 原檔所在資料夾)")
    ap.add_argument("--model", choices=MODELS, default="roformer", help="主分軌模型 (預設 roformer)")
    ap.add_argument("--guitars", choices=["auto", "electric", "acoustic"], default="auto",
                    help="吉他類型: auto=自動判斷, electric=全部當電吉他, acoustic=全部當木吉他")
    args = ap.parse_args()
    sep = Separator(output_format="WAV", model_file_dir=str(DATA_DIR / "models"), log_level=30)
    failed = []
    for f in args.files:
        if not f.exists():
            print(f"找不到檔案: {f}", file=sys.stderr)
            failed.append(f)
            continue
        try:
            process(f.resolve(), (args.output or f.parent).resolve(), sep, args.model, args.guitars)
        except Exception as e:  # 一首失敗不影響其他首
            print(f"\n處理失敗: {f.name}: {e}", file=sys.stderr)
            failed.append(f)
    if failed:
        print(f"\n{len(failed)} 個檔案失敗: " + ", ".join(p.name for p in failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
