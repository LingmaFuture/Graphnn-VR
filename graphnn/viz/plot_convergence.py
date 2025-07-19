import json
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from pathlib import Path

# 设置绘图风格
plt.style.use('seaborn')
#sns.set_palette("husl")

# 1) 读取所有模型的历史数据
hist_files = {
    'LSTM': 'outputs/history_lstm.json',
    'GRU': 'outputs/history_gru.json',
    'GNN': 'outputs/history_gnn.json'
}

histories = {}
for model_name, hist_file in hist_files.items():
    try:
        with open(hist_file, 'r') as f:
            histories[model_name] = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {hist_file} not found, skipping {model_name}")

# 2) 创建子图布局
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Training Convergence Analysis', fontsize=16)

# 3) Loss 收敛图
ax = axes[0, 0]
for model_name, hist in histories.items():
    epochs = np.arange(1, len(hist["train_loss_means"]) + 1)
    ax.errorbar(epochs, hist["train_loss_means"], yerr=hist["train_loss_stds"],
                label=f'{model_name} Train', marker='o', alpha=0.7)
    ax.errorbar(epochs, hist["val_loss_means"], yerr=hist["val_loss_stds"],
                label=f'{model_name} Val', marker='s', alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Convergence: Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 4) Accuracy 收敛图
ax = axes[0, 1]
for model_name, hist in histories.items():
    epochs = np.arange(1, len(hist["train_acc_means"]) + 1)
    ax.errorbar(epochs, hist["train_acc_means"], yerr=hist["train_acc_stds"],
                label=f'{model_name} Train', marker='o', alpha=0.7)
    ax.errorbar(epochs, hist["val_acc_means"], yerr=hist["val_acc_stds"],
                label=f'{model_name} Val', marker='s', alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Training Convergence: Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

# 5) 最终性能对比
ax = axes[1, 0]
model_names = list(histories.keys())
final_train_acc = [hist["train_acc_means"][-1] for hist in histories.values()]
final_val_acc = [hist["val_acc_means"][-1] for hist in histories.values()]
x = np.arange(len(model_names))
width = 0.35
ax.bar(x - width/2, final_train_acc, width, label='Train')
ax.bar(x + width/2, final_val_acc, width, label='Validation')
ax.set_ylabel('Final Accuracy')
ax.set_title('Final Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()
ax.grid(True, alpha=0.3)

# 6) 训练时间对比（如果可用）
ax = axes[1, 1]
if all('train_time' in hist for hist in histories.values()):
    train_times = [hist["train_time"][-1] for hist in histories.values()]
    ax.bar(model_names, train_times)
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Training time data not available',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
    ax.set_title('Training Time Comparison')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# 保存图片
plt.savefig('outputs/convergence_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
