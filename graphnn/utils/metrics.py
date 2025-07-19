# graphnn/utils/metrics.py
import torch, torch.nn.functional as F, numpy as np
from scipy.stats import percentileofscore

def quat_angle_err(q1, q2):
    """四元数角差 (rad)，q[...,4]=[x,y,z,w]"""
    cos = torch.abs((q1 * q2).sum(-1)).clamp(-1, 1)
    return 2 * torch.acos(cos)

def accuracy(pred, gt,
             y_mean, y_std,
             pos_cm=1.0, rot_deg=3.0):
    """
    pred,gt : (B,14) 6 pos + 8 quat（已 *not* 反标准化）
    y_mean/y_std : 1-D tensor
    """
    pred = pred * y_std + y_mean
    gt   = gt   * y_std + y_mean

    # -------- 位置 --------
    pos_err = (pred[:, :6] - gt[:, :6]).reshape(-1,2,3).norm(dim=-1)  # (B,2)
    pos_ok  = pos_err.mean(dim=1) < pos_cm/100.                    # 使用平均误差

    # -------- 旋转 --------
    q_pred = F.normalize(pred[:, 6:14].reshape(-1,2,4), dim=-1)
    q_gt   = F.normalize(gt[:,   6:14].reshape(-1,2,4), dim=-1)
    ang_err = quat_angle_err(q_pred, q_gt) * 57.29578             # deg
    rot_ok  = ang_err.mean(dim=1) < rot_deg                       # 使用平均误差

    return (pos_ok & rot_ok).float().mean().item()

