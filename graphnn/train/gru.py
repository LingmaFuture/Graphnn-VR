"""
用法：
python -m graphnn.train.gru
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
import numpy as np
from pathlib import Path
from graphnn.data.dataset import StudentToTeacherDataset  # 假设数据集加载脚本是这个路径
from graphnn.engine.model import GRUModel  # 导入我们定义的GRU模型
import json
from graphnn.utils.metrics import accuracy

# ---------- 常量 ----------
ROOT = Path(__file__).resolve().parents[2] / "data/processed"  # 数据集路径
CKPT_DIR = Path("outputs/checkpoints")  # 模型保存路径
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_TRAIN = 256
BATCH_VAL = 512
EPOCHS = 50
HIDDEN_DIM = 128
LR = 3e-4
PATIENCE = 10  # 早停耐心值

# ---------- 反标准化参数（仅准确率用） ----------
y_mean = torch.tensor(np.load(CKPT_DIR / "scaler_y_mean.npy"))
y_std = torch.tensor(np.load(CKPT_DIR / "scaler_y_std.npy"))

# ---------- 数据集 ----------
def load_dataset():
    stu_tensors = [torch.load(p) for p in ROOT.glob("*/player_1_All.pt")]
    tea_tensors = [torch.load(p) for p in ROOT.glob("*/player_0_All.pt")]
    assert len(stu_tensors) == len(tea_tensors)
    return ConcatDataset([
        StudentToTeacherDataset(stu, tea) for stu, tea in zip(stu_tensors, tea_tensors)
    ])

# ---------- 训练 / 验证 ----------
def run_epoch(model, dl, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    losses, accs = [], []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train:
            optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        accs.append(accuracy(out.cpu(), y.cpu(), y_mean, y_std))
    return np.mean(losses), np.std(losses), np.mean(accs), np.std(accs)

# ---------- 主函数 ----------
def main():
    ds = load_dataset()
    tr_len = int(0.85 * len(ds))
    train_ds, val_ds = random_split(ds, [tr_len, len(ds) - tr_len],
                                    generator=torch.Generator().manual_seed(42))

    nworkers = 0 if os.name == "nt" else 4
    train_dl = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                          num_workers=nworkers, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_VAL, shuffle=False,
                        num_workers=nworkers, pin_memory=False)

    model = GRUModel(input_dim=32, hidden_dim=HIDDEN_DIM, output_dim=26).to(DEVICE)
    optim_ = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=EPOCHS)
    crit = nn.MSELoss()

    best_acc, epochs_no_improve, hist = 0.0, 0, {}

    for ep in range(EPOCHS):
        tlm, tls, tam, tas = run_epoch(model, train_dl, crit, optim_)
        vlm, vls, vam, vas = run_epoch(model, val_dl, crit)

        sched.step()
        print(f"Ep {ep + 1:02}/{EPOCHS}  "
              f"tloss {tlm:.3f}±{tls:.3f}  tacc {tam:.2%}±{tas:.2%}  "
              f"vloss {vlm:.3f}±{vls:.3f}  vacc {vam:.2%}±{vas:.2%}")

        for k, v in zip(
            ["tlm", "tls", "tam", "tas", "vlm", "vls", "vam", "vas"],
            [tlm, tls, tam, tas, vlm, vls, vam, vas]):
            hist.setdefault(k, []).append(v)

        if vam > best_acc:
            best_acc = vam
            epochs_no_improve = 0
            torch.save(model.state_dict(), CKPT_DIR / "gru/best_gru.pth")
            print(f"  ✔ save best_gru  vacc={best_acc:.2%}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {ep + 1} (no improvement for {PATIENCE} epochs)")
                break

    json.dump(hist, open("outputs/checkpoints/gru/history_gru.json", "w"))

if __name__ == "__main__":
    main()
