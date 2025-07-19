"""
用法：
python -m graphnn.viz.plot_traj_all --student_id Data_CYJ_J --start 0 --end 1000
python -m graphnn.viz.plot_traj_all --student_id Data_CYJ_J --start 1500 --end 2000
python -m graphnn.viz.plot_traj_all --student_id Data_CYJ_J --start 20500 --end 21000
python -m graphnn.viz.plot_traj_all --student_id Data_LZB_J --start 0 --end 1000
python -m graphnn.viz.plot_traj_all --student_id Data_LZB_J --start 20000 --end 22500
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# 从预处理脚本中的归一化常量
RADIUS = 150.0  # cm

def plot_all_trajectories(stu_data, tea_data, pred_data, start=0, end=None, save_path=None):
    """
    在同一图中绘制学生真实轨迹、老师真实轨迹和模型预测轨迹
    
    Args:
        stu_data: 学生数据，(N, D) tensor
        tea_data: 老师数据，(N, D) tensor
        pred_data: 预测数据，(N, D) tensor
        start: 起始帧索引
        end: 结束帧索引
        save_path: 保存图像的路径
    """
    if end is None:
        end = min(len(stu_data), len(tea_data), len(pred_data))
    
    stu_data = stu_data[start:end]
    tea_data = tea_data[start:end]
    pred_data = pred_data[start:end]
    
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
    
    # 预测
    pred_left_y = pred_data[:, 1] * RADIUS
    pred_left_z = pred_data[:, 2] * RADIUS
    pred_right_y = pred_data[:, 4] * RADIUS
    pred_right_z = pred_data[:, 5] * RADIUS

    # 创建图表和子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左手轨迹
    axes[0].plot(stu_left_y, stu_left_z, '-', color='green', label='Student Real')
    axes[0].plot(tea_left_y, tea_left_z, '--', color='blue', label='Teacher Real')
    axes[0].plot(pred_left_y, pred_left_z, '-', color='orange', label='Teacher Pred')
    axes[0].set_title('Left Hand YZ Trajectory')
    axes[0].set_xlabel('Y (cm)')
    axes[0].set_ylabel('Z (cm)')
    axes[0].legend()
    axes[0].grid(True)

    # 右手轨迹
    axes[1].plot(stu_right_y, stu_right_z, '-', color='green', label='Student Real')
    axes[1].plot(tea_right_y, tea_right_z, '--', color='blue', label='Teacher Real')
    axes[1].plot(pred_right_y, pred_right_z, '-', color='orange', label='Teacher Pred')
    axes[1].set_title('Right Hand YZ Trajectory')
    axes[1].set_xlabel('Y (cm)')
    axes[1].set_ylabel('Z (cm)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    
    return fig

def add_info_text(fig, student_id, start, end, r2=None, mae=None):
    """添加信息文本到图表中"""
    info_text = f"Student ID: {student_id}\nFrames: {start}-{end}"
    if r2 is not None:
        info_text += f"\nR²: {r2:.4f}"
    if mae is not None:
        info_text += f"\nMAE: {mae:.4f}"
    
    fig.text(0.02, 0.02, info_text, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))

def main():
    parser = argparse.ArgumentParser(description="Plot student, teacher, and prediction trajectories")
    parser.add_argument('--student_id', type=str, required=True, help='Student folder name to process (e.g., Data_CYJ_J)')
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    parser.add_argument('--end', type=int, default=None, help='End frame')
    parser.add_argument('--output_dir', type=str, default="outputs/plots", help='Directory to save plots')
    args = parser.parse_args()

    # 1. 加载学生和老师真实数据
    data_dir = Path('data/processed') / args.student_id
    stu_path = data_dir / 'player_1_All.pt'
    tea_path = data_dir / 'player_0_All.pt'
    if not stu_path.exists() or not tea_path.exists():
        raise FileNotFoundError(f"Student or teacher data file not found: {stu_path} or {tea_path}")
    
    stu_data = torch.load(stu_path)
    tea_data = torch.load(tea_path)
    
    # 确保老师数据维度正确，只取前14维
    if tea_data.shape[1] > 14:
        tea_data = tea_data[:, :14]
    
    # 2. 加载预测数据
    pred_path = Path("outputs/predictions") / f"{args.student_id}_preds.pt"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    
    pred_data = torch.load(pred_path)
    predictions = pred_data['predictions']
    
    # 可选: 提取R²和MAE指标，如果在预测文件中有保存的话
    r2 = None
    mae = None
    if 'metrics' in pred_data and 'r2' in pred_data['metrics']:
        r2 = pred_data['metrics']['r2']
    if 'metrics' in pred_data and 'mae' in pred_data['metrics']:
        mae = pred_data['metrics']['mae']
    
    # 3. 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{args.student_id}_all_trajectories_{args.start}_{args.end if args.end else 'end'}.png"
    
    # 4. 生成图表
    fig = plot_all_trajectories(
        stu_data=stu_data,
        tea_data=tea_data,
        pred_data=predictions,
        start=args.start,
        end=args.end,
        save_path=save_path 
    )
    
    # 5. 添加额外信息
    add_info_text(fig, args.student_id, args.start, args.end or len(predictions), r2, mae)
    
    
    plt.show()

if __name__ == "__main__":
    main()
