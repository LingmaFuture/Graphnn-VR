"""
GNN+LSTM模型训练脚本 - 调试版本
"""
# 运行此脚本进行调试模式训练，使用较小的数据子集和批大小

# 导入主训练脚本
from improved_train_gnn_vr import *

if __name__ == "__main__":
    # 修改关键参数为调试值
    globals()['BATCH_SIZE'] = 64       # 更小的批大小
    globals()['DEBUG_MODE'] = True     # 启用调试模式
    globals()['NUM_WORKERS'] = 2       # 减少工作线程
    
    print("="*50)
    print("调试模式训练")
    print(f"批大小: {BATCH_SIZE}")
    print(f"工作线程: {NUM_WORKERS}")
    print("使用小数据子集")
    print("="*50)
    
    # 运行训练
    train()