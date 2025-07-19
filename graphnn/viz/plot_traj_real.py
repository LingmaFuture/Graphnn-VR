"""
用法：
python -m graphnn.viz.plot_traj_real --student_id Data_CYJ_J --start 0 --end 1000
python -m graphnn.viz.plot_traj_real --student_id Data_LZB_J --start 1000 --end 2000
python -m graphnn.viz.plot_traj_real --student_id Data_GWX_J --start 0 --end 1000
python -m graphnn.viz.plot_traj_real --student_id Data_LZB_J --start 20500 --end 21000
python -m graphnn.viz.plot_traj_real --student_id Data_LZB_J --start 20000 --end 22500
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# 从预处理脚本中的归一化常量
RADIUS = 150.0  # cm

def plot_real_trajectories(stu_data, tea_data, start=0, end=None, save_path=None):
    """绘制学生和老师真实左右手YZ轨迹"""
    if end is None:
        end = min(len(stu_data), len(tea_data))
    stu_data = stu_data[start:end]
    tea_data = tea_data[start:end]
    
    # 反归一化：乘以RADIUS (150.0 cm)将归一化坐标转回原始单位
    # 学生
    stu_left_y = stu_data[:, 1] * RADIUS
    stu_left_z = stu_data[:, 2] * RADIUS
    stu_right_y = stu_data[:, 4] * RADIUS
    stu_right_z = stu_data[:, 5] * RADIUS
    # 老师
    tea_left_y = tea_data[:, 1] * RADIUS
    tea_left_z = tea_data[:, 2] * RADIUS
    tea_right_y = tea_data[:, 4] * RADIUS
    tea_right_z = tea_data[:, 5] * RADIUS

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(stu_left_y, stu_left_z, label='Student Real', color='green')
    plt.plot(tea_left_y, tea_left_z, label='Teacher Real', color='blue', linestyle='--')
    plt.xlabel('Y (cm)')
    plt.ylabel('Z (cm)')
    plt.title('Left Hand YZ Trajectory')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(stu_right_y, stu_right_z, label='Student Real', color='green')
    plt.plot(tea_right_y, tea_right_z, label='Teacher Real', color='blue', linestyle='--')
    plt.xlabel('Y (cm)')
    plt.ylabel('Z (cm)')
    plt.title('Right Hand YZ Trajectory')
    plt.legend()
    plt.grid(True)

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

    # 加载学生和老师真实数据
    stu_path = Path('data/processed') / args.student_id / 'player_1_All.pt'
    tea_path = Path('data/processed') / args.student_id / 'player_0_All.pt'
    if not stu_path.exists() or not tea_path.exists():
        raise FileNotFoundError(f"Student or teacher data file not found: {stu_path} 或 {tea_path}")
    stu_data = torch.load(stu_path)
    tea_data = torch.load(tea_path)

    # 创建输出目录
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_name = f"{args.student_id}_stu_teacher_real_trajectory.png"

    plot_real_trajectories(
        stu_data=stu_data,
        tea_data=tea_data,
        start=args.start,
        end=args.end,
        save_path=output_dir / save_name
    )

if __name__ == "__main__":
    main()
