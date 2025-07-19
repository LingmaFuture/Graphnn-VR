 # graphnn/server/gnn_socket.py

import asyncio, json, websockets, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial.transform import Rotation as R
import numpy as np

# PyG 用于 GCNConv
from torch_geometric.nn import GCNConv

# ==== 载入训练同款 μ / σ ====
scaler_y_mean = np.load("outputs/checkpoints/scaler_y_mean.npy")   # shape (26,)
scaler_y_std  = np.load("outputs/checkpoints/scaler_y_std.npy")

# ---------- 配置 ----------
WINDOW       = 50
PORT         = 8081
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IN_DIM       = 32           # 与预处理一致
OUT_DIM      = 26           # 输出维度
GNN_HID_DIM  = 64           # GNN 隐藏维度
LSTM_HID_DIM = 128          # LSTM 隐藏维度
# --------------------------

# 用于 GCN 的固定图结构（左右手两个节点，全连通）
EDGE_INDEX = torch.tensor([[0, 1], 
                          [1, 0]], dtype=torch.long).to(DEVICE)

# ---------- GNN+LSTM 模型定义 ----------
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

# 导入必要的包
import torch.nn as nn

# 载入模型
model = ImprovedSeqGNN().to(DEVICE)
model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth", map_location=DEVICE)["model_state_dict"])
model.eval()

# ---------- 工具函数 ----------
def inverse_transform(pred: np.ndarray) -> np.ndarray:
    """反标准化 (26,)"""
    return pred * scaler_y_std + scaler_y_mean

def pack_response(vec26: np.ndarray):
    """
    vec26 : 前26维数据，包含:
      - 位置 (6): [lx ly lz rx ry rz]
      - 旋转四元数 (8): [qLx qLy qLz qLw qRx qRy qRz qRw]
      - 速度 (6): [vlx vly vlz vrx vry vrz]
      - 距离 (6): [dlx dly dlz drx dry drz]
      
    返回 UE 需要的 Vector3 + Rotator(yaw,pitch,roll)
    """
    # ----- 位置 -----
    l_loc = vec26[0:3].tolist()
    r_loc = vec26[3:6].tolist()

    # ----- 四元数 → 欧拉角 (Yaw,Pitch,Roll) -----
    quat_l = vec26[6:10]            # [x,y,z,w]
    quat_r = vec26[10:14]

    eul_l = R.from_quat(quat_l).as_euler('zyx', degrees=True).tolist()  # [yaw,pitch,roll]
    eul_r = R.from_quat(quat_r).as_euler('zyx', degrees=True).tolist()

    return {
        "status":   "success",
        "message":  "GNN预测结果",
        "L_location": l_loc,
        "L_rotation": eul_l,   # 3 维
        "R_location": r_loc,
        "R_rotation": eul_r,
        # 可选：也可以包含速度和距离信息
        "L_velocity": vec26[14:17].tolist(),
        "R_velocity": vec26[17:20].tolist(),
        "L_distance": vec26[20:23].tolist(),
        "R_distance": vec26[23:26].tolist()
    }

# ---------- WebSocket 处理 ----------
async def send_initial(websocket):
    msg = {"status": "success", "message": "GNN Server ready. Send frames."}
    await websocket.send(json.dumps(msg))

# 加载offset_dict相关函数
def load_offset_json(json_path):
    """
    加载 location_data_task_offset.json，返回 dict: {(Task, hands, index): [x, y, z]}
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    offset_dict = {}
    for item in data:
        key = (item["Task"], item["hands"], item["index"])
        offset_dict[key] = [item["x"], item["y"], item["z"]]
    return offset_dict

# 在服务启动时加载offset_dict
OFFSET_DICT = load_offset_json(Path(__file__).resolve().parents[2] / "data/location_offsets.json")
RADIUS = 150.0  # cm → 归一化

async def handle_client(ws):
    print("🟢 UE Connected to GNN Server:", ws.remote_address)
    await send_initial(ws)
    buf = []                                # 环形缓冲区 (max 50)
    try:
        async for raw in ws:
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"status":"error","msg":"JSON parse fail"}))
                continue

            # -------- 解析 32 维特征 --------
            # 注意：这里假设前端传来不含next_ball_offset_l/r，而是task_id、tar_l、tar_r
            try:
                # 取task_id、tar_l、tar_r，字段名需与预处理一致
                task_id = d.get("Task")
                tar_l = d.get("it.hand_tar_l")
                tar_r = d.get("it.hand_tar_r")
                if task_id not in [1, 2, 3]:
                    task_id = 4
                # 查偏移量
                off_l = OFFSET_DICT.get((task_id, "l", tar_l+1))
                if off_l is None:
                    off_l = OFFSET_DICT.get((task_id, "l", 0))
                off_r = OFFSET_DICT.get((task_id, "r", tar_r+1))
                if off_r is None:
                    off_r = OFFSET_DICT.get((task_id, "r", 0))
                off_l = [v / RADIUS for v in off_l]
                off_r = [v / RADIUS for v in off_r]

                rec = np.array([
                    # 1. 手部位置 (6)
                    *d.get("hand_loc_l", {}).values(),      # x y z
                    *d.get("hand_loc_r", {}).values(),      # x y z
                    # 2. 手部旋转四元数 (8)
                    *d.get("hand_quat_l", {}).values(),     # x y z w
                    *d.get("hand_quat_r", {}).values(),     # x y z w
                    # 3. 手部速度 (6)
                    *d.get("hand_vel_l", {}).values(),      # x y z
                    *d.get("hand_vel_r", {}).values(),      # x y z
                    # 4. 当前目标球距离 (6)
                    *d.get("hand_dis_l", {}).values(),      # x y z
                    *d.get("hand_dis_r", {}).values(),      # x y z
                    # 5. 下一个目标球偏移 (6) 由服务端查表
                    *off_l, *off_r
                ], dtype=np.float32)

                # 确保输入是32维
                if len(rec) != IN_DIM:
                    raise ValueError(f"Input dimension mismatch: expected {IN_DIM}, got {len(rec)}")
                
                buf.append(rec)
                if len(buf) < WINDOW:    # 未满 50 帧
                    continue
                if len(buf) > WINDOW:
                    buf.pop(0)

                # -------- 调模型 --------
                x = torch.tensor(np.stack(buf), dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,50,32)
                with torch.no_grad():
                    pred = model(x).cpu().squeeze(0).numpy()  # (26,)

                vec26 = inverse_transform(pred)

                await ws.send(json.dumps(pack_response(vec26)))
                
            except Exception as e:
                await ws.send(json.dumps({"status":"error","msg":f"处理错误: {str(e)}"}))
                continue

    except websockets.exceptions.ConnectionClosedError:
        print("🔴 UE Disconnected from GNN Server")

async def main():
    print(f"GNN WebSocket server listening : ws://127.0.0.1:{PORT}")
    async with websockets.serve(handle_client, "127.0.0.1", PORT):
        await asyncio.Future()          # run forever

if __name__ == "__main__":
    asyncio.run(main())