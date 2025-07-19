"""
用法：
python -m graphnn.engine.infer_sliding_gnn --student_id Data_CYJ_J --smooth 
python -m graphnn.engine.infer_sliding_gnn --student_id Data_GWX_J --smooth
python -m graphnn.engine.infer_sliding_gnn --student_id Data_LZB_J --smooth

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
from graphnn.train.gnn_vr_2 import DualStreamGNN, PositionalEncoding, AUTOREGRESSIVE, TIME_EMB_DIM, TEACHER_DIM
from graphnn.utils.smooth import savgol_smooth
from sklearn.metrics import r2_score, mean_absolute_error

# ---------- 常量 ----------
CKPT_DIR   = Path("./outputs/checkpoints/gnn_vr_2")  # 与训练脚本保持一致
DATA_DIR   = Path(__file__).resolve().parents[2] / "data/processed"
SEQ_LEN    = 50  # 与训练脚本中SEQ_LEN保持一致 
IN_DIM     = 32  # 与训练脚本保持一致
OUT_DIM    = 26  # 与训练脚本保持一致
GNN_HID_DIM = 64  # 与训练脚本保持一致
LSTM_HID_DIM = 128  # 与训练脚本保持一致
BATCH_SIZE = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if os.name == "nt" else 4
OUTPUT_DIM = 14  # 最终输出文件只保存前14维

# ---------- 载入 scaler（反标准化用）---------- 
# 如果有需要的话，使用训练时保存的归一化参数
y_mean = None
y_std = None
try:
    y_mean = torch.tensor(np.load(CKPT_DIR / "scaler_y_mean.npy"))
    y_std = torch.tensor(np.load(CKPT_DIR / "scaler_y_std.npy"))
except FileNotFoundError:
    print("未找到归一化参数文件，将使用原始预测值")

# ---------- 模型 ----------
def load_model(model_path=None):
    """加载模型"""
    if model_path is None:
        model_path = CKPT_DIR / "final_model.pth"
    
    # 使用双流架构模型
    model = DualStreamGNN(
        in_dim=IN_DIM, 
        teacher_dim=TEACHER_DIM,
        gnn_hid_dim=GNN_HID_DIM, 
        lstm_hid_dim=LSTM_HID_DIM,
        time_emb_dim=TIME_EMB_DIM,
        out_dim=OUT_DIM,
        use_teacher_input=AUTOREGRESSIVE
    ).to(DEVICE)
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

# ---------- 单个 .pt 的滑窗推理 ----------
class InferenceSeq2OneDataset(torch.utils.data.Dataset):
    def __init__(self, student, teacher, seq_len=SEQ_LEN):
        self.student = student
        self.teacher = teacher
        self.seq_len = seq_len
        self.length = len(student) - seq_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_student = self.student[idx:idx + self.seq_len]  # (50, 32)
        
        # 为自回归模式准备教师数据
        if idx == 0:
            # 第一个窗口，教师数据用0填充
            x_teacher = torch.zeros(self.seq_len, OUT_DIM)
        else:
            # 后续窗口，使用前一个窗口的教师数据
            # 最后一帧没有前一帧，用零填充第一个位置
            zero_pad = torch.zeros(1, OUT_DIM)
            teacher_prev = self.teacher[idx:idx+self.seq_len-1]
            x_teacher = torch.cat([zero_pad, teacher_prev], dim=0)
            
        y = self.teacher[idx + self.seq_len]  # (26,)
        return x_student, x_teacher, y

@torch.no_grad()
def sliding_predict(stu_pt_path: Path, tea_pt_path: Path, model):
    student = torch.load(stu_pt_path, map_location="cpu")
    teacher = torch.load(tea_pt_path, map_location="cpu")
    if teacher.shape[1] > OUT_DIM:
        teacher = teacher[:, :OUT_DIM]

    dataset = InferenceSeq2OneDataset(student, teacher, SEQ_LEN)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    # 收集预测和真实值
    all_preds = []
    all_targets = []
    
    for x_student, x_teacher, y in dataloader:
        x_student = x_student.to(DEVICE, non_blocking=True)
        x_teacher = x_teacher.to(DEVICE, non_blocking=True)
        
        # 使用双流输入
        pred = model(x_student, x_teacher).cpu()
        all_preds.append(pred)
        all_targets.append(y)
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    # 只取前14维的预测和目标值
    return preds[:, :OUTPUT_DIM], targets[:, :OUTPUT_DIM]

@torch.no_grad()
def autoregressive_predict(stu_pt_path: Path, tea_pt_path: Path, model):
    """自回归预测模式 - 每次使用上一次的预测作为新的输入"""
    student = torch.load(stu_pt_path, map_location="cpu")
    teacher = torch.load(tea_pt_path, map_location="cpu")
    if teacher.shape[1] > OUT_DIM:
        teacher = teacher[:, :OUT_DIM]
    
    # 预测结果和真实值
    all_preds = []
    all_targets = []
    
    # 初始化教师缓冲区（全零）
    teacher_buffer = torch.zeros(1, SEQ_LEN, OUT_DIM, device=DEVICE)
    
    # 滑动窗口预测
    for idx in range(len(student) - SEQ_LEN):
        # 获取学生窗口
        x_student = student[idx:idx + SEQ_LEN].unsqueeze(0).to(DEVICE)
        target = teacher[idx + SEQ_LEN]
        
        # 预测下一帧
        pred = model(x_student, teacher_buffer)
        
        # 更新教师缓冲区（移除最旧的帧，添加新预测）
        teacher_buffer = torch.cat([
            teacher_buffer[:, 1:], 
            pred.unsqueeze(1)
        ], dim=1)
        
        # 收集结果
        all_preds.append(pred.cpu())
        all_targets.append(target)
        
        # 每100帧打印一次进度
        if idx % 100 == 0:
            print(f"处理进度: {idx}/{len(student) - SEQ_LEN}")
    
    preds = torch.stack(all_preds).squeeze(1)
    targets = torch.stack(all_targets)
    
    return preds[:, :OUTPUT_DIM], targets[:, :OUTPUT_DIM]

# ---------- 评估函数 ----------
def evaluate_predictions(preds, targets):
    """计算评估指标"""
    # 转为numpy以便计算
    preds_np = preds.numpy()
    targets_np = targets.numpy()
    
    # 计算评估指标
    r2 = r2_score(targets_np, preds_np)
    mae = mean_absolute_error(targets_np, preds_np)
    
    return {
        'r2': r2,
        'mae': mae
    }

# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_id', type=str, required=True, help='Student folder name to process')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to predictions')
    parser.add_argument('--window', type=int, default=21, help='Smoothing window size')
    parser.add_argument('--autoregressive', action='store_true', help='Use true autoregressive prediction')
    args = parser.parse_args()

    # 加载模型
    model = load_model(args.model_path)
    print(f"模型已加载")
    
    # 构建数据文件路径
    stu_pt_path = DATA_DIR / args.student_id / "player_1_All.pt"
    tea_pt_path = DATA_DIR / args.student_id / "player_0_All.pt"
    if not stu_pt_path.exists() or not tea_pt_path.exists():
        raise FileNotFoundError(f"数据文件未找到: {stu_pt_path} 或 {tea_pt_path}")

    # 生成预测
    print(f"正在生成预测...")
    if args.autoregressive:
        print("使用真实自回归模式预测...")
        preds, targets = autoregressive_predict(stu_pt_path, tea_pt_path, model)
    else:
        preds, targets = sliding_predict(stu_pt_path, tea_pt_path, model)
    
    print(f"预测完成: 生成了 {len(preds)} 条预测, 维度: {preds.shape}")
    
    # 评估结果
    metrics = evaluate_predictions(preds, targets)
    print(f"评估指标: R² = {metrics['r2']:.4f}, MAE = {metrics['mae']:.4f}")
    
    # 平滑处理（如需要）
    if args.smooth:
        print(f"正在应用平滑处理 (窗口大小: {args.window})...")
        pred_np = preds.numpy()
        pred_smooth = savgol_smooth(pred_np, window=args.window, poly=3)
        preds = torch.from_numpy(pred_smooth)
        print("平滑处理完成")
        
        # 重新评估平滑后的结果
        metrics_smooth = evaluate_predictions(preds, targets)
        print(f"平滑后评估指标: R² = {metrics_smooth['r2']:.4f}, MAE = {metrics_smooth['mae']:.4f}")

    # 保存预测结果
    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备输出保存内容
    output_dict = {
        'predictions': preds,
        'ground_truth': targets,
    }
    
    # 如果有归一化参数，也一并保存
    if y_mean is not None and y_std is not None:
        # 只保存对应的前14维的归一化参数
        output_dict['y_mean'] = y_mean[:OUTPUT_DIM] if y_mean.shape[0] > OUTPUT_DIM else y_mean
        output_dict['y_std'] = y_std[:OUTPUT_DIM] if y_std.shape[0] > OUTPUT_DIM else y_std
    
    # 保存文件
    output_path = output_dir / f"{args.student_id}_preds.pt"
    torch.save(output_dict, output_path)
    print(f"预测结果已保存至: {output_path}")
    print(f"输出张量形状: 预测={preds.shape}, 真实值={targets.shape}")

if __name__ == "__main__":
    main()
