"""用 CLAP (文字-音訊零樣本分類) 判斷每一段聲音是哪種樂器，並依時間把訊號分到不同軌。

用途:
  - 吉他軌: 分成 木吉他 / 電吉他
  - 其他軌: 分出 合成器
"""
import numpy as np
import torch
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import resample_poly
from transformers import ClapModel, ClapProcessor

CLAP_ID = "laion/clap-htsat-unfused"
CLAP_SR = 48000
SEG_SEC = 4.0
FADE_SEC = 0.25

GUITAR_LABELS = {
    "acoustic": ["acoustic guitar", "nylon string classical guitar", "strummed acoustic guitar"],
    "electric": ["electric guitar", "distorted electric guitar", "clean electric guitar", "overdriven guitar riff"],
}
OTHER_LABELS = {
    "synth": ["synthesizer", "analog synth lead", "synth pad", "electronic synth bass", "arpeggiated synthesizer"],
    "other": ["string section", "violin", "brass section", "saxophone", "flute", "organ",
              "percussion", "sound effects", "choir", "harp", "accordion", "harmonica"],
}
# 報告用：描述「其他」軌裡可能有什麼
REPORT_LABELS = {
    "弦樂": "string section", "銅管": "brass section", "薩克斯風": "saxophone", "長笛": "flute",
    "風琴": "organ", "打擊樂": "percussion", "音效": "sound effects", "合唱": "choir",
    "豎琴": "harp", "手風琴": "accordion", "口琴": "harmonica", "合成器": "synthesizer",
}

_model = _proc = None
if torch.cuda.is_available():
    _device = "cuda"
elif torch.backends.mps.is_available():  # Apple Silicon
    _device = "mps"
else:
    _device = "cpu"


def _emb(out):
    # 新版 transformers 回傳 ModelOutput，舊版直接回傳 tensor
    if not torch.is_tensor(out):
        out = out.pooler_output
    return out / out.norm(dim=-1, keepdim=True)


def _load():
    global _model, _proc
    if _model is None:
        _model = ClapModel.from_pretrained(CLAP_ID).to(_device).eval()
        _proc = ClapProcessor.from_pretrained(CLAP_ID)
    return _model, _proc


def _segments(audio, sr):
    """切成固定長度片段，回傳 (48k mono 片段列表, 每段的 RMS)。"""
    mono = audio.mean(axis=0)
    mono48 = resample_poly(mono, CLAP_SR, sr).astype(np.float32)
    seg = int(SEG_SEC * CLAP_SR)
    n = max(1, int(np.ceil(len(mono48) / seg)))
    segs, rms = [], []
    for i in range(n):
        s = mono48[i * seg:(i + 1) * seg]
        s = np.pad(s, (0, seg - len(s)))
        segs.append(s)
        rms.append(np.sqrt(np.mean(s**2)))
    return segs, np.array(rms)


@torch.no_grad()
def _probs(segs, prompts):
    model, proc = _load()
    texts = [f"the sound of {p}" for p in prompts]
    t = proc(text=texts, return_tensors="pt", padding=True).to(_device)
    text_emb = _emb(model.get_text_features(**t))
    out = []
    for i in range(0, len(segs), 16):
        a = proc(audio=segs[i:i + 16], sampling_rate=CLAP_SR, return_tensors="pt").to(_device)
        emb = _emb(model.get_audio_features(**a))
        logits = model.logit_scale_a.exp() * emb @ text_emb.T
        out.append(logits.softmax(dim=-1).cpu().numpy())
    return np.concatenate(out)


def class_scores(audio, sr, groups):
    """每段對各群組的機率總和。回傳 (dict 群組->每段機率, 每段 RMS)。"""
    segs, rms = _segments(audio, sr)
    names, prompts = [], []
    for g, ps in groups.items():
        names += [g] * len(ps)
        prompts += ps
    p = _probs(segs, prompts)
    names = np.array(names)
    # 以各群組的平均機率比較，避免提示詞較多的群組佔便宜
    mean = np.stack([p[:, names == g].mean(axis=1) for g in groups], axis=1)
    mean /= mean.sum(axis=1, keepdims=True)
    return {g: mean[:, i] for i, g in enumerate(groups)}, rms


def route(audio, sr, groups, target):
    """依 CLAP 判斷把 audio 分成 (target 部分, 其餘部分)，並回傳 target 所佔比例。"""
    scores, rms = class_scores(audio, sr, groups)
    is_t = (scores[target] > 0.5).astype(float)
    # 太小聲的片段判斷不可靠，沿用鄰近片段結果
    loud = rms > rms.max() * 0.05 if rms.max() > 0 else np.zeros_like(rms, bool)
    if loud.any():
        idx = np.where(loud)[0]
        is_t = np.interp(np.arange(len(is_t)), idx, is_t[idx])
        is_t = (is_t > 0.5).astype(float)
    if len(is_t) >= 3:
        is_t = median_filter(is_t, size=3, mode="nearest")
    # 片段決策轉成取樣點的增益曲線，邊界做淡入淡出
    seg = int(SEG_SEC * sr)
    gain = np.repeat(is_t, seg)[: audio.shape[1]]
    gain = np.pad(gain, (0, audio.shape[1] - len(gain)), mode="edge")
    gain = uniform_filter1d(gain, size=max(1, int(FADE_SEC * sr)), mode="nearest")
    weight = float((is_t * loud).sum() / max(1, loud.sum()))
    return audio * gain, audio * (1 - gain), weight


def describe(audio, sr, top=3):
    """回傳這段聲音最可能包含的樂器（中文）。"""
    segs, rms = _segments(audio, sr)
    loud = rms > rms.max() * 0.1
    if not loud.any():
        return []
    p = _probs([s for s, l in zip(segs, loud) if l], list(REPORT_LABELS.values())).mean(axis=0)
    order = np.argsort(p)[::-1][:top]
    names = list(REPORT_LABELS)
    return [(names[i], float(p[i])) for i in order]
