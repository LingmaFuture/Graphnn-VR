# improved_train_gnn_vr.py
"""
配置：
windows11
python 3.10
pytorch 2.1.0
cuda 12.1
显卡：4060Ti-16G
cpu：i5-12400F
内存：32GB
"""
"""
进一步改进的GNN+LSTM模型训练脚本
- 添加时间步编码
- 实现双流输入架构
- 添加趋势损失函数
- 保留之前所有改进
"""
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.amp import GradScaler
from tqdm import tqdm
import numpy as np
import time
import random
from sklearn.metrics import r2_score, mean_absolute_error

# PyG 用于 GCNConv
from torch_geometric.nn import GCNConv

# -------------------------
# 1. 超参数
# -------------------------
SEQ_LEN       = 50         # 序列长度（50 帧）
IN_DIM        = 32         # 输入维度（每帧 32 维预处理后向量）
OUT_DIM       = 26         # 输出维度（6+8+6+6）
GNN_HID_DIM   = 64         # GNN 隐藏维度
LSTM_HID_DIM  = 128        # LSTM 隐藏维度
TIME_EMB_DIM  = 16         # 时间嵌入维度
TEACHER_DIM   = OUT_DIM    # 教师输出维度
BATCH_SIZE    = 256        # 批大小
LR            = 1e-3       # 学习率
EPOCHS        = 50         # 训练总轮次
PATIENCE      = 5          # 早停耐心值
TRAIN_VAL_SPLIT = 0.8      # 训练集比例
NUM_WORKERS   = 5          # 数据加载器工作线程数
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./outputs/checkpoints/gnn_vr_2"
LOG_INTERVAL  = 10         # 每N批次打印一次信息
TREND_LOSS_WEIGHT = 0.2    # 趋势损失权重
AUTOREGRESSIVE = True      # 是否使用自回归模式

# 用于 GCN 的固定图结构（左右手两个节点，全连通）
EDGE_INDEX = torch.tensor([[0, 1], 
                           [1, 0]], dtype=torch.long)

# 确保检查点目录存在
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------
# 2. 改进的数据集定义（支持双流输入）
# -------------------------
class ImprovedVRDataset(Dataset):
    def __init__(self, data_list, seq_len=SEQ_LEN, use_teacher_input=AUTOREGRESSIVE):
        """
        进一步改进的数据集实现
        - 支持教师数据作为额外输入
        - 保留预先计算的滑动窗口
        
        Args:
            data_list: [(student_tensor, teacher_tensor)]列表
            seq_len: 序列长度
            use_teacher_input: 是否使用教师数据作为额外输入
        """
        self.seq_len = seq_len
        self.use_teacher_input = use_teacher_input
        self.windows = []  # 存储所有可用的滑动窗口索引: [(data_idx, start_idx)]
        
        # 预计算所有可用的滑动窗口
        for data_idx, (student, teacher) in enumerate(data_list):
            total_frames = student.shape[0]
            if total_frames <= seq_len:
                continue
                
            # 对每个样本创建所有可能的滑动窗口
            for start_idx in range(total_frames - seq_len - 1):
                self.windows.append((data_idx, start_idx))
                
        self.data_list = data_list
        
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        # 获取预计算的窗口索引
        data_idx, start_idx = self.windows[idx]
        student, teacher = self.data_list[data_idx]
        
        # 提取学生窗口数据
        x_student = student[start_idx:start_idx+self.seq_len]  # [seq_len, 32]
        
        # 提取对应的教师数据（用于双流输入）
        if self.use_teacher_input:
            # 使用教师前一帧数据作为输入
            # 在训练时，使用真实的教师数据
            x_teacher = teacher[start_idx:start_idx+self.seq_len-1]  # [seq_len-1, OUT_DIM]
            # 在第一帧没有前一帧数据，用零填充
            zero_pad = torch.zeros(1, teacher.shape[1], dtype=teacher.dtype)
            x_teacher = torch.cat([zero_pad, x_teacher], dim=0)  # [seq_len, OUT_DIM]
        else:
            # 不使用教师数据作为输入，创建空张量占位
            x_teacher = torch.zeros((self.seq_len, OUT_DIM), dtype=student.dtype)
        
        # 标签是第seq_len+1帧的教师数据
        y = teacher[start_idx+self.seq_len]
        
        # 获取下一个目标球位置（假设在教师数据中的位置），用于趋势损失
        # 这里假设目标球位置在教师数据的前6个维度中
        # 如果没有下一帧数据，则使用当前帧
        if start_idx + self.seq_len + 1 < teacher.shape[0]:
            next_target = teacher[start_idx+self.seq_len+1, :6]
        else:
            next_target = teacher[start_idx+self.seq_len, :6]
            
        return x_student, x_teacher, y, next_target

def load_data(root_dir, val_ratio=0.2, seed=42):
    """加载数据并分割训练集/验证集"""
    random.seed(seed)
    
    all_data = []
    for fld in glob.glob(os.path.join(root_dir, "*")):
        t_path = os.path.join(fld, "player_0_All.pt")
        s_path = os.path.join(fld, "player_1_All.pt")
        
        if not (os.path.exists(t_path) and os.path.exists(s_path)):
            continue
            
        student = torch.load(s_path, map_location="cpu")
        teacher = torch.load(t_path, map_location="cpu")[:, :OUT_DIM]
        
        if student.size(0) > SEQ_LEN:
            all_data.append((student, teacher))
    
    # 打乱数据并分割
    random.shuffle(all_data)
    split_idx = int(len(all_data) * (1 - val_ratio))
    
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    return train_data, val_data

# -------------------------
# 3. 周期性时间编码函数
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        """
        周期性位置编码
        
        Args:
            d_model: 编码维度
            max_len: 最大序列长度
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区而不是参数
        self.register_buffer('pe', pe)
        
    def forward(self, x, seq_dim=1):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, ...]
            seq_dim: 序列维度的索引
        """
        batch_size = x.size(0)
        seq_len = x.size(seq_dim)
        
        # Get position encodings for this sequence length
        pe = self.pe[:seq_len]  # [seq_len, d_model]
        
        # Reshape to [batch_size, seq_len, d_model]
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        
        return pe

# -------------------------
# 4. 改进的模型定义
# -------------------------
class DualStreamGNN(nn.Module):
    def __init__(self, in_dim=IN_DIM, teacher_dim=TEACHER_DIM, gnn_hid_dim=GNN_HID_DIM, 
                 lstm_hid_dim=LSTM_HID_DIM, time_emb_dim=TIME_EMB_DIM, out_dim=OUT_DIM, 
                 dropout=0.2, use_teacher_input=AUTOREGRESSIVE):
        super().__init__()
        
        # 每个节点特征维度
        self.node_feat_dim = in_dim // 2
        self.use_teacher_input = use_teacher_input
        
        # GCN层: 只创建一次，重复使用
        self.gcn = GCNConv(self.node_feat_dim, gnn_hid_dim)
        
        # 时间编码层
        self.time_encoder = PositionalEncoding(time_emb_dim)
        
        # 教师输入嵌入层
        if use_teacher_input:
            self.teacher_embedding = nn.Linear(teacher_dim, gnn_hid_dim)
        
        # LSTM层输入维度计算
        lstm_input_dim = gnn_hid_dim * 2  # 左右手节点特征拼接
        
        if use_teacher_input:
            lstm_input_dim += gnn_hid_dim  # 加上教师嵌入维度
            
        lstm_input_dim += time_emb_dim  # 加上时间编码维度
        
        # LSTM层: 处理时序
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hid_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # 全连接层: 带有Dropout正则化
        self.fc = nn.Sequential(
            nn.Linear(lstm_hid_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim)
        )
        
    def forward(self, x_student, x_teacher=None):
        """
        双流输入的前向传播
        
        Args:
            x_student: [B, SEQ_LEN, IN_DIM] 学生输入序列
            x_teacher: [B, SEQ_LEN, TEACHER_DIM] 教师输入序列（可选）
            
        返回: [B, OUT_DIM]
        """
        B, T, _ = x_student.size()  # 批大小，序列长度，特征维度
        
        # 获取时间编码
        time_encoding = self.time_encoder(torch.zeros(B, T, 1, device=x_student.device))  # [B, T, TIME_EMB_DIM]
        
        # 存储每个时间步的特征
        seq_feats = []
        
        # 为每个时间步处理图
        for t in range(T):
            # 获取当前帧
            frame = x_student[:, t, :]  # [B, IN_DIM]
            
            # 拆分为左右手节点特征
            left_hand = frame[:, :self.node_feat_dim]    # [B, IN_DIM//2]
            right_hand = frame[:, self.node_feat_dim:]   # [B, IN_DIM//2]
            
            # 高效批处理: 创建批图而不是每个样本一个图
            batch_size = B
            num_nodes_per_graph = 2  # 每个图有2个节点(左右手)
            
            # 准备节点特征: 对每个样本在批中排列节点
            # [B*2, node_feat_dim] - 交替排列左右手特征
            node_feats = torch.zeros(batch_size * num_nodes_per_graph, 
                                    self.node_feat_dim, device=x_student.device)
            
            # 左手节点 (偶数索引)
            node_feats[0::2] = left_hand
            # 右手节点 (奇数索引)
            node_feats[1::2] = right_hand
            
            # 准备边索引: 对批中每个图重复边索引
            batch_edge_index = EDGE_INDEX.clone().to(x_student.device)
            
            # 为批中每个图添加偏移
            edge_indices = [EDGE_INDEX.to(x_student.device) + i * num_nodes_per_graph for i in range(batch_size)]
            batch_edge_index = torch.cat(edge_indices, dim=1)
            
            # 应用GCN - 一次性处理整个批
            out = self.gcn(node_feats, batch_edge_index)  # [B*2, GNN_HID_DIM]
            
            # 重塑为 [B, 2*GNN_HID_DIM] - 将每个图的节点特征拼接在一起
            out = out.view(batch_size, num_nodes_per_graph, GNN_HID_DIM)
            out = out.reshape(batch_size, -1)  # [B, 2*GNN_HID_DIM]
            
            # Get timestep encoding and ensure it has the right shape
            timestep_encoding = time_encoding[:, t]  # [B, TIME_EMB_DIM]
            if timestep_encoding.dim() == 1:
                timestep_encoding = timestep_encoding.unsqueeze(0).expand(out.size(0), -1)
            out = torch.cat([out, timestep_encoding], dim=1)
            
            # 添加教师输入特征（如果启用）
            if self.use_teacher_input and x_teacher is not None:
                teacher_feat = x_teacher[:, t]  # [B, TEACHER_DIM]
                teacher_emb = self.teacher_embedding(teacher_feat)  # [B, GNN_HID_DIM]
                out = torch.cat([out, teacher_emb], dim=1)  # [B, 2*GNN_HID_DIM + TIME_EMB_DIM + GNN_HID_DIM]
            
            seq_feats.append(out)
        
        # 堆叠所有时间步特征
        # [B, SEQ_LEN, 2*GNN_HID_DIM + TIME_EMB_DIM + (GNN_HID_DIM if use_teacher_input)]
        seq_feats = torch.stack(seq_feats, dim=1)
        
        # LSTM层处理时序关系
        lstm_out, _ = self.lstm(seq_feats)  # [B, SEQ_LEN, LSTM_HID_DIM]
        
        # 获取序列最后一步的输出
        final_hidden = lstm_out[:, -1, :]  # [B, LSTM_HID_DIM]
        
        # 映射到输出维度
        output = self.fc(final_hidden)  # [B, OUT_DIM]
        
        return output

# -------------------------
# 5. 趋势损失函数
# -------------------------
class TrendAwareLoss(nn.Module):
    def __init__(self, mse_weight=1.0, trend_weight=TREND_LOSS_WEIGHT):
        """
        结合MSE和趋势的损失函数
        
        Args:
            mse_weight: MSE损失的权重
            trend_weight: 趋势损失的权重
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.trend_weight = trend_weight
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target, next_target, student_input=None):
        """
        计算损失
        
        Args:
            pred: [B, OUT_DIM] 模型预测
            target: [B, OUT_DIM] 当前真实标签
            next_target: [B, 6] 下一帧的目标位置
            student_input: [B, SEQ_LEN, IN_DIM] 学生输入序列（可选）
            
        返回: 标量损失
        """
        # 基础MSE损失
        mse_loss = self.mse(pred, target)
        
        # 如果不需要趋势损失或权重为0
        if self.trend_weight <= 0.0:
            return mse_loss
            
        # 计算趋势损失
        # 提取位置信息（假设前6维是位置）
        pred_pos = pred[:, :6]  # [B, 6]
        target_pos = target[:, :6]  # [B, 6]
        
        # 计算当前预测位置到下一个目标的方向向量
        pred_to_next = next_target - pred_pos  # [B, 6]
        target_to_next = next_target - target_pos  # [B, 6]
        
        # 归一化方向向量
        pred_to_next_norm = torch.nn.functional.normalize(pred_to_next, p=2, dim=1)
        target_to_next_norm = torch.nn.functional.normalize(target_to_next, p=2, dim=1)
        
        # 计算方向的余弦相似度
        cos_sim = torch.sum(pred_to_next_norm * target_to_next_norm, dim=1)  # [B]
        
        # 余弦相似度越高越好，所以用1-cos_sim作为损失
        direction_loss = 1.0 - cos_sim.mean()
        
        # 如果提供了学生输入，计算学生动作幅度
        movement_weight = 1.0
        if student_input is not None:
            # 计算学生最后5帧的平均移动幅度
            last_frames = student_input[:, -5:, :16]  # 假设前16维是位置
            movement = torch.norm(last_frames[:, 1:] - last_frames[:, :-1], dim=2).mean()
            
            # 当学生移动幅度小时，增加趋势损失的权重
            movement_threshold = 0.05  # 阈值，低于此值为"微小"移动
            movement_weight = torch.exp(-movement / movement_threshold).item()
        
        # 组合损失：MSE + 加权的趋势损失
        total_loss = self.mse_weight * mse_loss + self.trend_weight * movement_weight * direction_loss
        
        return total_loss

# -------------------------
# 6. 评估函数
# -------------------------
def evaluate(model, data_loader, criterion, device):
    """在给定数据集上评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_student, x_teacher, y, next_target in data_loader:
            x_student = x_student.to(device, non_blocking=True)
            x_teacher = x_teacher.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            next_target = next_target.to(device, non_blocking=True)
            
            # 模型预测
            pred = model(x_student, x_teacher)
            loss = criterion(pred, y, next_target, x_student)
            
            # 累计损失
            total_loss += loss.item() * x_student.size(0)
            
            # 收集预测和目标值用于计算其他指标
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader.dataset)
    
    # 拼接预测和目标值
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 计算其他评估指标
    r2 = r2_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    
    return avg_loss, r2, mae

# -------------------------
# 7. 自回归推理函数
# -------------------------
def autoregressive_inference(model, x_student, steps=10):
    """
    自回归推理（用于演示）
    
    Args:
        model: 训练好的模型
        x_student: [1, SEQ_LEN, IN_DIM] 学生输入序列
        steps: 自回归步数
        
    返回: [steps, OUT_DIM] 预测的教师轨迹
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 将输入移到模型所在设备
    x_student = x_student.to(device)
    
    # 初始化教师输入（全零）
    x_teacher = torch.zeros(1, SEQ_LEN, OUT_DIM, device=device)
    
    # 存储所有步骤的预测
    predictions = []
    
    with torch.no_grad():
        for _ in range(steps):
            # 预测下一帧
            pred = model(x_student, x_teacher)  # [1, OUT_DIM]
            predictions.append(pred.clone())
            
            # 更新教师输入（自回归）
            x_teacher = torch.cat([x_teacher[:, 1:], pred.unsqueeze(1)], dim=1)
            
            # 可选：更新学生输入（如果在推理场景中学生有新数据）
            # 这里我们假设学生保持静止
    
    # 堆叠所有预测
    predictions = torch.cat(predictions, dim=0)  # [steps, OUT_DIM]
    
    return predictions.cpu()

# -------------------------
# 8. 改进的训练流程
# -------------------------
def train():
    # 加载并分割数据
    print("正在加载数据...")
    train_data, val_data = load_data(root_dir="./data/processed/", val_ratio=0.2)
    print(f"加载完成: {len(train_data)}个训练样本, {len(val_data)}个验证样本")
    
    # 创建数据集和数据加载器
    train_dataset = ImprovedVRDataset(train_data, use_teacher_input=AUTOREGRESSIVE)
    val_dataset = ImprovedVRDataset(val_data, use_teacher_input=AUTOREGRESSIVE)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)} 窗口, 验证集大小: {len(val_dataset)} 窗口")
    
    # 初始化模型
    model = DualStreamGNN(use_teacher_input=AUTOREGRESSIVE).to(DEVICE)
    
    # 损失函数和优化器
    criterion = TrendAwareLoss(mse_weight=1.0, trend_weight=TREND_LOSS_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 训练状态追踪
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    print(f"开始训练 - 共 {EPOCHS} 轮次")
    print(f"模型配置: 时间嵌入={TIME_EMB_DIM}维, 双流输入={AUTOREGRESSIVE}, 趋势损失权重={TREND_LOSS_WEIGHT}")
    
    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 使用tqdm显示进度条
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS}") as pbar:
            for batch_idx, (x_student, x_teacher, y, next_target) in enumerate(train_loader):
                x_student = x_student.to(DEVICE, non_blocking=True)
                x_teacher = x_teacher.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                next_target = next_target.to(DEVICE, non_blocking=True)
                
                # 零梯度
                optimizer.zero_grad()
                
                # 混合精度前向传播
                with autocast(device_type='cuda'):
                    pred = model(x_student, x_teacher)
                    loss = criterion(pred, y, next_target, x_student)
                
                # 反向传播和优化
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # 累计损失
                train_loss += loss.item() * x_student.size(0)
                
                # 更新进度条
                if batch_idx % LOG_INTERVAL == 0:
                    pbar.set_postfix({'loss': f"{loss.item():.6f}"})
                pbar.update(1)
        
        # 计算平均训练损失
        train_loss /= len(train_dataset)
        
        # 验证阶段
        val_loss, val_r2, val_mae = evaluate(model, val_loader, criterion, DEVICE)
        
        # 打印训练和验证结果
        print(f"Epoch {epoch}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}, R²: {val_r2:.4f}, MAE: {val_mae:.4f}")
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 检查是否需要保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存最佳模型
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
                'train_loss': train_loss,
                'config': {
                    'time_emb_dim': TIME_EMB_DIM,
                    'autoregressive': AUTOREGRESSIVE,
                    'trend_loss_weight': TREND_LOSS_WEIGHT
                }
            }, checkpoint_path)
            
            print(f"  → 保存最佳模型 (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  → 验证损失未改善 ({patience_counter}/{PATIENCE})")
        
        # 早停检查
        if patience_counter >= PATIENCE:
            print(f"连续 {PATIENCE} 轮次未改善，提前停止训练")
            break
    
    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_r2': val_r2,
        'config': {
            'time_emb_dim': TIME_EMB_DIM,
            'autoregressive': AUTOREGRESSIVE,
            'trend_loss_weight': TREND_LOSS_WEIGHT
        }
    }, os.path.join(CHECKPOINT_DIR, "final_model.pth"))
    
    # 打印总训练时间
    total_time = time.time() - start_time
    print(f"训练完成. 共耗时: {total_time/60:.2f} 分钟")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型已保存至: {CHECKPOINT_DIR}")
    
    # 简单的自回归演示（可选）
    if AUTOREGRESSIVE:
        print("\n演示自回归推理效果:")
        # 从验证集中选择一个样本
        test_sample = next(iter(val_loader))
        x_student = test_sample[0][:1]  # 只取第一个样本
        
        # 执行自回归推理
        trajectory = autoregressive_inference(model, x_student, steps=10)
        print(f"生成的轨迹形状: {trajectory.shape}")
        print("生成的前三帧预测:")
        print(trajectory[:3])

# -------------------------
# 9. 推理函数（用于部署）
# -------------------------
class TeacherPredictor:
    """用于实时推理的教师预测器"""
    
    def __init__(self, model_path, device=None):
        """
        初始化预测器
        
        Args:
            model_path: 模型检查点路径
            device: 运行设备
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取模型配置
        config = checkpoint.get('config', {})
        time_emb_dim = config.get('time_emb_dim', TIME_EMB_DIM)
        autoregressive = config.get('autoregressive', AUTOREGRESSIVE)
        
        # 创建模型
        self.model = DualStreamGNN(
            time_emb_dim=time_emb_dim,
            use_teacher_input=autoregressive
        ).to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 初始化状态
        self.seq_len = SEQ_LEN
        self.autoregressive = autoregressive
        self.student_buffer = None
        self.teacher_buffer = None
        
    def reset(self):
        """重置内部缓冲区"""
        self.student_buffer = None
        self.teacher_buffer = None
        
    def predict(self, student_frame):
        """
        预测下一个教师帧
        
        Args:
            student_frame: [32] 当前学生帧
            
        返回: [OUT_DIM] 预测的教师帧
        """
        # 转换为张量
        if not isinstance(student_frame, torch.Tensor):
            student_frame = torch.tensor(student_frame, dtype=torch.float32)
        
        # 确保形状正确
        student_frame = student_frame.view(1, -1)  # [1, 32]
        
        # 初始化或更新学生缓冲区
        if self.student_buffer is None:
            # 首次调用，用相同帧填充历史
            self.student_buffer = student_frame.repeat(1, self.seq_len, 1)
        else:
            # 更新缓冲区（移除最旧的帧，添加新帧）
            self.student_buffer = torch.cat([
                self.student_buffer[:, 1:], 
                student_frame.unsqueeze(1)
            ], dim=1)
        
        # 处理教师缓冲区（如果使用自回归）
        if self.autoregressive:
            if self.teacher_buffer is None:
                # 首次调用，初始化为零
                self.teacher_buffer = torch.zeros(1, self.seq_len, OUT_DIM, 
                                                 dtype=torch.float32,
                                                 device=self.device)
        
        # 执行预测
        with torch.no_grad():
            # 将输入移到设备
            x_student = self.student_buffer.to(self.device)
            
            # 使用教师缓冲区（如果启用自回归）
            if self.autoregressive and self.teacher_buffer is not None:
                x_teacher = self.teacher_buffer.to(self.device)
                pred = self.model(x_student, x_teacher)
            else:
                pred = self.model(x_student)
        
        # 更新教师缓冲区（如果使用自回归）
        if self.autoregressive:
            self.teacher_buffer = torch.cat([
                self.teacher_buffer[:, 1:],
                pred.unsqueeze(1)
            ], dim=1)
        
        # 返回预测结果
        return pred.cpu().squeeze(0).numpy()

if __name__ == "__main__":
    train()