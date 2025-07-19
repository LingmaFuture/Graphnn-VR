import numpy as np
from scipy.signal import savgol_filter

def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    """
    x: (T, D) 预测序列
    window: 平滑窗口大小，越大越平滑但响应越迟钝
    """
    pad = window // 2
    # 在两端各重复 pad 帧，以保持长度不变
    xp = np.pad(x, ((pad, pad), (0,0)), mode='edge')
    # 每个维度做卷积
    kernel = np.ones(window) / window
    return np.stack([np.convolve(xp[:,i], kernel, mode='valid') for i in range(x.shape[1])], axis=1)

def savgol_smooth(x: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """
    window: 必须是奇数
    poly: 多项式阶数，一般取 2–5
    """
    return savgol_filter(x, window_length=window, polyorder=poly, axis=0) 