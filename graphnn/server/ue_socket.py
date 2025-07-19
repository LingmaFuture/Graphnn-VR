# graphnn/server/ue_socket.py
"""
WebSocket ⇆ UE5
---------------
• 学员每帧上传 27 维原始特征（和训练时一致）
• 服务器缓存最近 50 帧，调用 TinyGRU 推理下一帧教师 12 维
• 反标准化后打包成 Vector3 / Rotator 发送回 UE
"""

import asyncio, json, websockets, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial.transform import Rotation as R   # ← 新增
import numpy as np

# ==== 载入训练同款 μ / σ ====
scaler_X_mean = np.load("scaler_X_mean.npy")   # shape (27,)
scaler_X_std  = np.load("scaler_X_std.npy")
scaler_y_mean = np.load("scaler_y_mean.npy")   # shape (14,)
scaler_y_std  = np.load("scaler_y_std.npy")


# ---------- 配置 ----------
WINDOW       = 50
PORT         = 8080
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES_IN  = 27           # 与预处理一致
FEATURES_OUT = 14           # pos6 + quat8 = 14
HIDDEN_DIM   = 128
# --------------------------

# ---------- 模型 ----------
from graphnn.models.gru import TinyGRU   # 如用 HandGNN 修改此行
model = TinyGRU(in_dim=41, hid=HIDDEN_DIM, out_dim=FEATURES_OUT).to(DEVICE)
model.load_state_dict(torch.load("outputs/checkpoints/best.pth", map_location=DEVICE))
model.eval()

# ---------- 工具函数 ----------
def preprocess_frame(arr27: np.ndarray) -> np.ndarray:
    """单帧归一化 (27,) → (27,)"""
    return (arr27 - scaler_X_mean) / scaler_X_std

def inverse_transform(pred14: np.ndarray) -> np.ndarray:
    """反标准化 (14,)"""
    return pred14 * scaler_y_std + scaler_y_mean

def seq_to_model_input(buffer: list[np.ndarray]) -> torch.Tensor:
    """
    buffer 长度 = 50，每元素 27
    1) 连续 26 + Task One-Hot 15 → (50,41)
    2) 添加 batch 维 → (1,50,41)
    """
    X = np.stack(buffer)                       # (50,27)
    cont = X[:, 1:27]                           # (50,26)
    task = X[:, 27].astype(int) - 1            # 0–14
    task_oh = np.eye(15)[task]                 # (50,15)
    seq = np.concatenate([cont, task_oh], axis=-1)  # (50,41)
    return torch.tensor(seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)

# ---------- WebSocket 处理 ----------
async def send_initial(websocket):
    msg = {"status": "success", "message": "Server ready. Send frames."}
    await websocket.send(json.dumps(msg))

from scipy.spatial.transform import Rotation as R   # 顶部只需导入一次

def pack_response(vec14: np.ndarray):
    """
    vec14 : [ lx ly lz  rx ry rz  qLx qLy qLz qLw  qRx qRy qRz qRw ]
    返回 UE 需要的 Vector3 + Rotator(yaw,pitch,roll)
    """
    # ----- 位置 -----
    l_loc = vec14[0:3].tolist()
    r_loc = vec14[3:6].tolist()

    # ----- 四元数 → 欧拉角 (Yaw,Pitch,Roll) -----
    quat_l = vec14[6:10]            # [x,y,z,w]
    quat_r = vec14[10:14]

    eul_l = R.from_quat(quat_l).as_euler('zyx', degrees=True).tolist()  # [yaw,pitch,roll]
    eul_r = R.from_quat(quat_r).as_euler('zyx', degrees=True).tolist()

    return {
        "status":   "success",
        "message":  "预测结果",
        "L_location": l_loc,
        "L_rotation": eul_l,   # 3 维
        "R_location": r_loc,
        "R_rotation": eul_r
    }


async def handle_client(ws):
    print("🟢 UE Connected:", ws.remote_address)
    await send_initial(ws)
    buf = []                                # 环形缓冲区 (max 50)
    try:
        async for raw in ws:
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"status":"error","msg":"JSON parse fail"}))
                continue

            # -------- 解析 27 维 --------
            rec = np.array([
                d.get("task",0),
                *d.get("hand_loc_l",{}).values(),      # x y z
                *d.get("hand_rot_l",{}).values(),      # pitch yaw roll
                *d.get("hand_vel_l",{}).values(),      # x y z
                *d.get("hand_loc_r",{}).values(),
                *d.get("hand_rot_r",{}).values(),
                *d.get("hand_vel_r",{}).values(),
                *d.get("hand_dis_l",{}).values(),
                *d.get("hand_dis_r",{}).values(),
                d.get("hand_tar_l",0),
                d.get("hand_tar_r",0)
            ], dtype=np.float32)

            buf.append(preprocess_frame(rec))
            if len(buf) < WINDOW:    # 未满 50 帧
                continue
            if len(buf) > WINDOW:
                buf.pop(0)

            # -------- 调模型 --------
            x = seq_to_model_input(buf)                # (1,50,41)
            with torch.no_grad():
                pred = model(x).cpu().squeeze(0).numpy()  # (14,)

            vec14 = inverse_transform(pred)

            await ws.send(json.dumps(pack_response(vec14)))

    except websockets.exceptions.ConnectionClosedError:
        print("🔴 UE Disconnected")

async def main():
    print(f"WebSocket server listening : ws://127.0.0.1:{PORT}")
    async with websockets.serve(handle_client, "127.0.0.1", PORT):
        await asyncio.Future()          # run forever

if __name__ == "__main__":
    asyncio.run(main())
