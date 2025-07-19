# train_gnn_vr.py
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
改进的GNN+LSTM模型训练脚本
- 修复GNN实现效率问题 
- 添加验证集评估
- 增加学习率调度
- 改进数据加载策略
- 添加早停机制和更多评估指标
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
BATCH_SIZE    = 256        # 批大小
LR            = 1e-3       # 学习率
EPOCHS        = 50         # 训练总轮次
PATIENCE      = 5          # 早停耐心值
TRAIN_VAL_SPLIT = 0.8      # 训练集比例
NUM_WORKERS   = 5          # 数据加载器工作线程数
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./outputs/checkpoints/gnn_vr"
LOG_INTERVAL  = 10         # 每N批次打印一次信息

# 用于 GCN 的固定图结构（左右手两个节点，全连通）
EDGE_INDEX = torch.tensor([[0, 1], 
                           [1, 0]], dtype=torch.long)

# 确保检查点目录存在
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------
# 2. 改进的数据集定义
# -------------------------
class VRDataset(Dataset):
    def __init__(self, data_list, seq_len=SEQ_LEN):
        """
        改进的数据集实现，提前计算所有可用窗口
        
        Args:
            data_list: [(student_tensor, teacher_tensor)]列表
            seq_len: 序列长度
        """
        self.seq_len = seq_len
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
        
        # 提取窗口数据
        x = student[start_idx:start_idx+self.seq_len]  # [seq_len, 32]
        y = teacher[start_idx+self.seq_len]          # # 这样 label 就是第 51 帧，而不是第 50 帧
        
        return x, y

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
# 3. 改进的模型定义
# -------------------------
class ImprovedSeqGNN(nn.Module):
    def __init__(self, in_dim=IN_DIM, gnn_hid_dim=GNN_HID_DIM, 
                 lstm_hid_dim=LSTM_HID_DIM, out_dim=OUT_DIM, dropout=0.2):
        super().__init__()
        
        # 每个节点特征维度
        self.node_feat_dim = in_dim // 2
        
        # GCN层: 只创建一次，重复使用
        self.gcn = GCNConv(self.node_feat_dim, gnn_hid_dim)
        
        # LSTM层: 处理时序
        self.lstm = nn.LSTM(
            input_size=gnn_hid_dim * 2,  # 左右手节点特征拼接
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
        
    def forward(self, x):
        """
        x: [B, SEQ_LEN, IN_DIM]
        
        返回: [B, OUT_DIM]
        """
        B, T, _ = x.size()  # 批大小，序列长度，特征维度
        
        # 存储每个时间步的特征
        seq_feats = []
        
        # 为每个时间步处理图
        for t in range(T):
            # 获取当前帧
            frame = x[:, t, :]  # [B, IN_DIM]
            
            # 拆分为左右手节点特征
            left_hand = frame[:, :self.node_feat_dim]    # [B, IN_DIM//2]
            right_hand = frame[:, self.node_feat_dim:]   # [B, IN_DIM//2]
            
            # 高效批处理: 创建批图而不是每个样本一个图
            batch_size = B
            num_nodes_per_graph = 2  # 每个图有2个节点(左右手)
            
            # 准备节点特征: 对每个样本在批中排列节点
            # [B*2, node_feat_dim] - 交替排列左右手特征
            node_feats = torch.zeros(batch_size * num_nodes_per_graph, 
                                    self.node_feat_dim, device=x.device)
            
            # 左手节点 (偶数索引)
            node_feats[0::2] = left_hand
            # 右手节点 (奇数索引)
            node_feats[1::2] = right_hand
            
            # 准备边索引: 对批中每个图重复边索引
            batch_edge_index = EDGE_INDEX.clone().to(x.device)
            
            # 为批中每个图添加偏移
            edge_indices = [EDGE_INDEX.to(x.device) + i * num_nodes_per_graph for i in range(batch_size)]
            batch_edge_index = torch.cat(edge_indices, dim=1)
            
            # 应用GCN - 一次性处理整个批
            out = self.gcn(node_feats, batch_edge_index)  # [B*2, GNN_HID_DIM]
            
            # 重塑为 [B, 2*GNN_HID_DIM] - 将每个图的节点特征拼接在一起
            out = out.view(batch_size, num_nodes_per_graph, GNN_HID_DIM)
            out = out.reshape(batch_size, -1)  # [B, 2*GNN_HID_DIM]
            
            seq_feats.append(out)
        
        # 堆叠所有时间步特征
        seq_feats = torch.stack(seq_feats, dim=1)  # [B, SEQ_LEN, 2*GNN_HID_DIM]
        
        # LSTM层处理时序关系
        lstm_out, _ = self.lstm(seq_feats)  # [B, SEQ_LEN, LSTM_HID_DIM]
        
        # 获取序列最后一步的输出
        final_hidden = lstm_out[:, -1, :]  # [B, LSTM_HID_DIM]
        
        # 映射到输出维度
        output = self.fc(final_hidden)  # [B, OUT_DIM]
        
        return output

# -------------------------
# 4. 评估函数
# -------------------------
def evaluate(model, data_loader, criterion, device):
    """在给定数据集上评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # 模型预测
            pred = model(x)
            loss = criterion(pred, y)
            
            # 累计损失
            total_loss += loss.item() * x.size(0)
            
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
# 5. 改进的训练流程
# -------------------------
def train():
    # 加载并分割数据
    print("正在加载数据...")
    train_data, val_data = load_data(root_dir="./data/processed/", val_ratio=0.2)
    print(f"加载完成: {len(train_data)}个训练样本, {len(val_data)}个验证样本")
    
    # 创建数据集和数据加载器
    train_dataset = VRDataset(train_data)
    val_dataset = VRDataset(val_data)
    
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
    model = ImprovedSeqGNN().to(DEVICE)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
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
    
    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 使用tqdm显示进度条
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS}") as pbar:
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                
                # 零梯度
                optimizer.zero_grad()
                
                # 混合精度前向传播
                with autocast(device_type='cuda'):
                    pred = model(x)
                    loss = criterion(pred, y)
                
                # 反向传播和优化
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # 累计损失
                train_loss += loss.item() * x.size(0)
                
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
                'train_loss': train_loss
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
        'val_r2': val_r2
    }, os.path.join(CHECKPOINT_DIR, "final_model.pth"))
    
    # 打印总训练时间
    total_time = time.time() - start_time
    print(f"训练完成. 共耗时: {total_time/60:.2f} 分钟")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型已保存至: {CHECKPOINT_DIR}")

if __name__ == "__main__":
    train()