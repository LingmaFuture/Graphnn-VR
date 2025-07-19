"""
用法：
python -m graphnn.engine.infer_sliding_save --student_id Data_CYJ_J
python -m graphnn.engine.infer_sliding_save --student_id Data_GWX_J
python -m graphnn.engine.infer_sliding_save --student_id Data_LZB_J

outputs/predictions/{student_id}_preds.pt 文件内容说明：
- predictions: (N, 14)  # 模型预测的老师轨迹（归一化，已平滑），N为样本数，14维为位置+四元数
- ground_truth: (N, 14) # 学生轨迹（归一化）
- y_mean: (14,)         # 归一化均值
- y_std: (14,)          # 归一化标准差

所有张量均为 torch.Tensor，需反归一化后可还原真实物理量。
"""
import torch, os, numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from graphnn.data.dataset import StudentToTeacherDataset
#from graphnn.models.gru import GRUModel
from graphnn.engine.model import LSTMModel
from graphnn.utils.metrics import accuracy
from graphnn.utils.smooth import moving_average, savgol_smooth

# ---------- 常量 ----------
CKPT_DIR   = Path("outputs/checkpoints")
DATA_DIR   = Path(__file__).resolve().parents[2] / "data/processed"
WINDOW     = 50
BATCH      = 64
RADIUS     = 150.0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NWORKERS   = 0 if os.name == "nt" else 4

# ---------- 载入 scaler（反标准化用） ----------
y_mean = torch.tensor(np.load(CKPT_DIR / "scaler_y_mean.npy"))
y_std  = torch.tensor(np.load(CKPT_DIR / "scaler_y_std.npy"))

# ---------- 模型 ----------
def load_model(model_path=None):
    """加载模型"""
    if model_path is None:
        model_path = CKPT_DIR / "best_lstm.pth"
    model = LSTMModel(input_dim=32, hidden_dim=128, output_dim=26).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# ---------- 单个 .pt 的滑窗推理 ----------
@torch.no_grad()
def sliding_predict(stu_pt_path: Path, tea_pt_path: Path, model):
    stu_arr = torch.load(stu_pt_path)  # (T,32)
    tea_arr = torch.load(tea_pt_path)  # (T,32)
    ds     = StudentToTeacherDataset(stu_arr, tea_arr, win=WINDOW)
    dl     = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=NWORKERS, pin_memory=False)

    preds, gts = [], []
    for seq, lbl in dl:
        out = model(seq.to(DEVICE)).cpu()
        preds.append(out)
        gts.append(lbl)
    return torch.cat(preds), torch.cat(gts)

# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_id', type=str, required=True, help='Student folder name to process (e.g., Data_CYJ_J)')
    args = parser.parse_args()

    model = load_model()
    # 构建数据文件路径
    stu_pt_path = DATA_DIR / args.student_id / "player_1_All.pt"
    tea_pt_path = DATA_DIR / args.student_id / "player_0_All.pt"
    if not stu_pt_path.exists() or not tea_pt_path.exists():
        raise FileNotFoundError(f"Data file not found: {stu_pt_path} 或 {tea_pt_path}")

    # 生成预测
    preds, gts = sliding_predict(stu_pt_path, tea_pt_path, model)
    
    # --- 平滑处理示例 ---
    pred_np = preds.numpy()
    pred_smooth = moving_average(pred_np, window=21)
    pred_sg = savgol_smooth(pred_np, window=31, poly=5)
    preds = torch.from_numpy(pred_sg)  # 或 pred_smooth
    # 如需保存平滑结果，可加参数或另存

    # 保存预测结果
    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存预测和真实值
    torch.save({
        'predictions': preds,
        'ground_truth': gts,
        'y_mean': y_mean,
        'y_std': y_std
    }, output_dir / f"{args.student_id}_preds.pt")
    
    # 计算并打印准确率
    acc = accuracy(preds, gts, y_mean, y_std)
    print(f"Accuracy for {args.student_id}: {acc:.2%}")
    print(f"Predictions saved to: {output_dir / f'{args.student_id}_preds.pt'}")

if __name__ == "__main__":
    main()
