"""依照立體聲定位 (panning) 把同一軌裡的多把吉他拆開。

混音時多把電吉他通常會擺在不同位置（例如節奏吉他左右各一、主奏在中間），
這裡用 STFT 計算每個時頻點的左右能量比，找出能量直方圖的峰值，每個峰值視為一把吉他，
再用軟遮罩 (soft mask) 把訊號分配到各把吉他。
"""
import numpy as np
from scipy.signal import find_peaks, stft, istft
from scipy.ndimage import gaussian_filter1d

N_FFT = 4096
HOP = 1024
BINS = 81  # 定位直方圖解析度 (-1 ~ +1)


def _pan_index(L, R):
    mag_l, mag_r = np.abs(L), np.abs(R)
    return (mag_r - mag_l) / (mag_l + mag_r + 1e-10), mag_l**2 + mag_r**2


def find_sources(audio, sr, min_prominence=0.12, min_distance=0.35):
    """回傳各把吉他的定位中心 (-1 = 最左, +1 = 最右)。audio shape: (2, n)。"""
    _, _, L = stft(audio[0], sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
    _, _, R = stft(audio[1], sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
    L, R = L.astype(np.complex64), R.astype(np.complex64)
    pan, energy = _pan_index(L, R)
    # 只看有聲音的時頻點，避免雜訊把直方圖拉平
    keep = energy > np.percentile(energy, 70)
    hist, edges = np.histogram(pan[keep], bins=BINS, range=(-1, 1), weights=energy[keep])
    hist = gaussian_filter1d(hist / (hist.max() + 1e-12), 1.5)
    centers = (edges[:-1] + edges[1:]) / 2
    dist_bins = max(1, int(min_distance / (2 / BINS)))
    # 兩端補零讓貼齊最左/最右的峰值也能被偵測到
    padded = np.concatenate([[0], hist, [0]])
    peaks, _ = find_peaks(padded, prominence=min_prominence, distance=dist_bins)
    peaks = peaks - 1
    return sorted(centers[peaks].tolist()), (L, R, pan)


def split_by_pan(audio, sr, positions, stft_cache=None, width=0.18):
    """依定位中心把 audio 拆成 len(positions) 軌。"""
    if stft_cache is None:
        _, stft_cache = find_sources(audio, sr)
    L, R, pan = stft_cache
    # 在 log 域做 softmax：離所有中心都很遠的時頻點也會完整分配出去，不會因數值下溢被丟掉
    logw = np.stack([-((pan - p) ** 2) / (2 * width**2) for p in positions])
    logw -= logw.max(axis=0, keepdims=True)
    weights = np.exp(logw)
    masks = weights / weights.sum(axis=0, keepdims=True)
    outs = []
    n = audio.shape[1]
    for m in masks:
        _, l = istft(L * m, sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
        _, r = istft(R * m, sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
        outs.append(np.stack([l[:n], r[:n]]))
    return outs


def position_label(p):
    if p < -0.6:
        return "Left"
    if p < -0.2:
        return "Center-Left"
    if p <= 0.2:
        return "Center"
    if p <= 0.6:
        return "Center-Right"
    return "Right"
