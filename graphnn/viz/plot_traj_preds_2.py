"""
用法：
python -m graphnn.viz.plot_traj_preds --student_id Data_CYJ_J --start 0 --end 1000
python -m graphnn.viz.plot_traj_preds --student_id Data_CYJ_J --start 1000 --end 2000
python -m graphnn.viz.plot_traj_preds --student_id Data_GWX_J --start 0 --end 1000
python -m graphnn.viz.plot_traj_preds --student_id Data_LZB_J --start 1000 --end 2000
python -m graphnn.viz.plot_traj_preds --student_id Data_LZB_J --start 20500 --end 21000
"""
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# 从预处理脚本中的归一化常量
RADIUS = 150.0  # cm

def plot_trajectory(preds, gts, y_mean, y_std, start=0, end=None, save_path=None):
    """绘制预测轨迹和真实轨迹（左右手分图）"""
    if end is None:
        end = len(preds)
    preds = preds[start:end]
    gts = gts[start:end]
    
    # 这里直接乘以RADIUS (150.0 cm)将归一化坐标转回原始单位，不需要额外的标准化逻辑
    # 只处理位置数据（索引0-5）- 手部位置坐标
    preds_pos = preds[:, :6] * RADIUS
    gts_pos = gts[:, :6] * RADIUS

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_left, ax_right = axes

    # 左手 YZ
    ax_left.plot(preds_pos[:, 1], preds_pos[:, 2], '-', color='orange', label='Teacher Pred')
    ax_left.plot(gts_pos[:, 1], gts_pos[:, 2], '--', color='blue', label='Teacher Real')
    ax_left.set_title('Left Hand YZ Trajectory')
    ax_left.set_xlabel('Y (cm)')
    ax_left.set_ylabel('Z (cm)')
    ax_left.legend()
    ax_left.grid(True)

    # 右手 YZ
    ax_right.plot(preds_pos[:, 4], preds_pos[:, 5], '-', color='orange', label='Teacher Pred')
    ax_right.plot(gts_pos[:, 4], gts_pos[:, 5], '--', color='blue', label='Teacher Real')
    ax_right.set_title('Right Hand YZ Trajectory')
    ax_right.set_xlabel('Y (cm)')
    ax_right.set_ylabel('Z (cm)')
    ax_right.legend()
    ax_right.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_id', type=str, required=True, help='Student folder name to process (e.g., Data_CYJ_J)')
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    parser.add_argument('--end', type=int, default=None, help='End frame')
    args = parser.parse_args()

    preds_path = Path("outputs/predictions") / f"{args.student_id}_preds.pt"
    if not preds_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {preds_path}")

    data = torch.load(preds_path)
    preds = data['predictions']
    gts = data['ground_truth']
    # 不再使用y_mean和y_std，因为我们直接用RADIUS进行反归一化
    
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_name = f"{args.student_id}_preds_trajectory_{args.start}_{args.end}.png"

    plot_trajectory(
        preds=preds,
        gts=gts,
        y_mean=None,  # 不再使用标准化参数
        y_std=None,   # 不再使用标准化参数
        start=args.start,
        end=args.end,
        save_path=output_dir / save_name
    )

if __name__ == "__main__":
    main() 