import torch, os, numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from graphnn.data.dataset import StudentToTeacherGraphDataset
from graphnn.models.gnn import GNNModel
from graphnn.utils.metrics import accuracy  # 统一使用utils中的accuracy
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# ---------- 0 解析命令行参数 ----------
parser = argparse.ArgumentParser(description='Evaluate model generalization')
parser.add_argument('--model_file', type=str, default='outputs/checkpoints/gnn/best_gnn.pth', 
                    help='Model checkpoint file name (default: best_gnn.pth)')
args = parser.parse_args()

CKPT_DIR   = Path("outputs/checkpoints")
DATA_DIR   = Path(__file__).resolve().parents[2] / "data/processed"
EVAL_DIR   = Path("outputs/eval"); EVAL_DIR.mkdir(parents=True, exist_ok=True)
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NWORKERS   = 0 if os.name == "nt" else 4
BATCH      = 512
WIN        = 50
RADIUS     = 150.0  # 位置归一化半径

# ---------- 1 读 scaler，仅为了 accuracy ----------
y_mean = torch.tensor(np.load(CKPT_DIR/"scaler_y_mean.npy"))
y_std  = torch.tensor(np.load(CKPT_DIR/"scaler_y_std.npy"))

# ---------- 2 载模型 ----------
from graphnn.models.gnn import GNNModel
model = GNNModel(input_dim=41, hidden_dim=256, output_dim=26).to(DEVICE)

model.load_state_dict(torch.load(CKPT_DIR / args.model, map_location=DEVICE))
model.eval()

# ---------- 3 辅助函数 ----------
def load_dataset(split):
    datasets = []
    split_dir = DATA_DIR / split
    for person_dir in split_dir.iterdir():
        stu = torch.load(person_dir / "player_1_All.pt")
        tea = torch.load(person_dir / "player_0_All.pt")
        datasets.append(StudentToTeacherGraphDataset(stu, tea, win=WIN))
    return ConcatDataset(datasets)

def quat_to_euler(quat):
    """将四元数转换为欧拉角（度）"""
    r = R.from_quat(quat)
    return r.as_euler('xyz', degrees=True)

def evaluate(loader, tag):
    all_preds = []
    all_labels = []
    
    # 计算推理延迟
    import time
    inference_times = []
    
    with torch.no_grad():
        for x,y in loader:
            start_time = time.time()
            p = model(x.to(DEVICE)).cpu()
            end_time = time.time()
            inference_times.append((end_time - start_time) / x.size(0))  # 每个样本的推理时间
            
            all_preds.append(p)
            all_labels.append(y)
    
    # 合并所有预测和标签
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # 计算准确率
    acc = accuracy(preds, labels, y_mean, y_std)
    
    # 计算MSE
    mse = torch.mean((preds - labels)**2).item()
    
    # 2. 计算位置误差（厘米）
    # 反归一化，恢复原始数值范围
    preds = preds * y_std + y_mean
    labels = labels * y_std + y_mean
    
    pos_preds = preds[:, :6] * RADIUS  # 转换为厘米
    pos_labels = labels[:, :6] * RADIUS
    pos_error = torch.mean(torch.abs(pos_preds - pos_labels), dim=0)
    
    # 3. 计算姿态角误差（度）
    # 分别处理左右手的四元数
    quat_preds_l = preds[:, 6:10].numpy()  # 左手四元数
    quat_preds_r = preds[:, 10:].numpy()   # 右手四元数
    quat_labels_l = labels[:, 6:10].numpy()
    quat_labels_r = labels[:, 10:].numpy()
    
    # 计算欧拉角误差
    euler_errors_l = []
    euler_errors_r = []
    for i in range(len(quat_preds_l)):
        euler_pred_l = quat_to_euler(quat_preds_l[i])
        euler_label_l = quat_to_euler(quat_labels_l[i])
        euler_errors_l.append(np.abs(euler_pred_l - euler_label_l))
        
        euler_pred_r = quat_to_euler(quat_preds_r[i])
        euler_label_r = quat_to_euler(quat_labels_r[i])
        euler_errors_r.append(np.abs(euler_pred_r - euler_label_r))
    
    euler_errors_l = np.mean(euler_errors_l, axis=0)
    euler_errors_r = np.mean(euler_errors_r, axis=0)
    
    # 计算平均推理延迟
    avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒
    
    # 打印评估结果
    print(f"{tag:12}:")
    print(f"  Accuracy : {acc:.3%}")
    
    print("\nPosition Error (cm):")
    print(f"  Left Hand (X,Y,Z): {pos_error[0]:.2f}, {pos_error[1]:.2f}, {pos_error[2]:.2f}")
    print(f"  Right Hand (X,Y,Z): {pos_error[3]:.2f}, {pos_error[4]:.2f}, {pos_error[5]:.2f}")
    
    print("\nRotation Error (degrees):")
    print(f"  Left Hand (Roll,Pitch,Yaw): {euler_errors_l[0]:.2f}, {euler_errors_l[1]:.2f}, {euler_errors_l[2]:.2f}")
    print(f"  Right Hand (Roll,Pitch,Yaw): {euler_errors_r[0]:.2f}, {euler_errors_r[1]:.2f}, {euler_errors_r[2]:.2f}")
    
    return acc, pos_error, euler_errors_l, euler_errors_r, mse, avg_inference_time

# ---------- 4 加载测试数据 ----------
# 加载验证集
val_ds = load_dataset('val')
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                   num_workers=NWORKERS, pin_memory=False)

# ---------- 5 评估 ----------
val_acc, val_pos_err, val_rot_err_l, val_rot_err_r, val_mse, val_inference_time = evaluate(val_dl, "Validation Set")

# 计算模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# 保存评估结果到文件
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = EVAL_DIR / f"eval_gnn_{timestamp}.txt"

with open(report_path, 'w') as f:
    f.write(f"Evaluation Report - {timestamp}\n")
    f.write(f"Model File: {args.model_file}\n\n")
    
    f.write("Model Information:\n")
    f.write(f"  Total Parameters: {total_params:,}\n")
    f.write(f"  Trainable Parameters: {trainable_params:,}\n")
    f.write(f"  Window Size: {WIN}\n")
    f.write(f"  Inference Latency: {val_inference_time:.2f} ms/sample\n")
    f.write(f"  MSE: {val_mse:.6f}\n\n")
    
    f.write("Validation Set:\n")
    f.write(f"  Accuracy: {val_acc:.3%}\n\n")
    
    f.write("Position Error (cm):\n")
    f.write(f"  Left Hand (X,Y,Z): {val_pos_err[0]:.2f}, {val_pos_err[1]:.2f}, {val_pos_err[2]:.2f}\n")
    f.write(f"  Right Hand (X,Y,Z): {val_pos_err[3]:.2f}, {val_pos_err[4]:.2f}, {val_pos_err[5]:.2f}\n")
    f.write(f"  Average: {val_pos_err.mean().item():.2f}\n\n")
    
    f.write("Rotation Error (degrees):\n")
    f.write(f"  Left Hand (Roll,Pitch,Yaw): {val_rot_err_l[0]:.2f}, {val_rot_err_l[1]:.2f}, {val_rot_err_l[2]:.2f}\n")
    f.write(f"  Right Hand (Roll,Pitch,Yaw): {val_rot_err_r[0]:.2f}, {val_rot_err_r[1]:.2f}, {val_rot_err_r[2]:.2f}\n")
    f.write(f"  Average: {np.mean([val_rot_err_l, val_rot_err_r]):.2f}\n")

print(f"\nEvaluation report saved to: {report_path}")
