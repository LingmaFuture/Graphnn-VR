# GraphNN-VR: 基于图神经网络的VR教学系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.5+-green.svg)](https://pytorch-geometric.readthedocs.io/)

基于图神经网络(GNN)的VR教学系统，实现教师与学员协同控制虚拟角色。通过动态权重调整，优化学员动作跟随效果，最终提升学员动作准确性。

## 🎯 项目概述

本项目实现了一个基于图神经网络的实时VR教学系统，能够根据学员输入数据预测教师动作。系统具有动态权重调整功能，优化教师指导与学员自主性之间的平衡。

### 核心特性

- **实时预测**: 基于学员过去1秒（50帧）数据，预测教师未来位置和旋转
- **动态权重调整**: 根据手部距离偏移自动调节权重（学员权重↑若偏移↑，保持权重和=1）
- **高精度**: 达到≥95%准确率，推理延迟<40ms
- **多模型支持**: 包含GNN、LSTM和GCN基线实现
- **全面可视化**: 轨迹对比图和消融实验热力图

## 📊 数据规格

### 输入数据格式
- **采样频率**: 50Hz
- **单帧数据**: 32维张量
  - **0-5**: 手部位置（6维）- [左手X, Y, Z, 右手X, Y, Z]
  - **6-13**: 手部旋转（8维）- [左手四元数x,y,z,w, 右手四元数x,y,z,w]
  - **14-19**: 手部速度（6维）- [左手速度X,Y,Z, 右手速度X,Y,Z]
  - **20-25**: 当前目标球距离（6维）- [左手到球X,Y,Z, 右手到球X,Y,Z]
  - **26-31**: 下一个目标球偏移（6维）- [左手偏移X,Y,Z, 右手偏移X,Y,Z]

### 数据归一化
- 所有位置、速度、距离、偏移量均除以RADIUS（150cm）实现归一化
- 欧拉角转换为四元数，顺序为[x, y, z, w]

### 数据集结构
```
data/
├── processed/          # 处理后的.pt文件
│   ├── Data_CYJ_J/
│   │   ├── player_0_All.pt  # 教师数据
│   │   └── player_1_All.pt  # 学员数据
│   └── ... (25个数据集)
└── raw/               # 原始JSON数据
    ├── Data_CYJ_J/
    ├── Data_GWX_J/
    └── ... (25个参与者)
```

## 🚀 安装

### 环境要求
- Python 3.10+
- CUDA兼容GPU（可选，用于加速）

### 使用Poetry（推荐）
```bash
# 安装Poetry（如果尚未安装）
curl -sSL https://install.python-poetry.org | python3 -

# 克隆仓库
git clone <repository-url>
cd GraphNN-VR-v5.2

# 安装依赖
poetry install

# 激活虚拟环境
poetry shell
```

### 使用pip
```bash
# 克隆仓库
git clone <repository-url>
cd GraphNN-VR-v5.2

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install torch torch-geometric numpy tqdm hydra-core matplotlib
```

## 📁 项目结构

```
GraphNN-VR-v5.2/
├── graphnn/                    # 主包
│   ├── data/                   # 数据处理模块
│   │   ├── dataset.py         # 数据集类
│   │   ├── preprocess.py      # 数据预处理
│   │   └── gen_mean_std.py    # 统计生成
│   ├── train/                  # 训练脚本
│   │   ├── gnn_vr_2.py       # 主GNN训练
│   │   ├── improved_train_gnn_vr.py
│   │   ├── gru.py             # GRU基线
│   │   └── lstm.py            # LSTM基线
│   ├── engine/                 # 模型和推理
│   │   ├── model.py           # 模型架构
│   │   ├── eval.py            # 评估指标
│   │   └── infer_sliding_gnn.py  # 推理引擎
│   ├── server/                 # 实时服务器
│   │   ├── gnn_socket.py      # GNN推理服务器
│   │   └── ue_socket.py       # UE4通信
│   ├── utils/                  # 工具
│   │   ├── metrics.py         # 性能指标
│   │   └── smooth.py          # 数据平滑
│   └── viz/                    # 可视化
│       ├── plot_traj_all.py   # 轨迹绘制
│       └── plot_convergence.py # 训练收敛
├── data/                       # 数据目录
├── docs/                       # 文档
├── logs/                       # 训练日志
├── outputs/                    # 模型输出
└── scripts/                    # 工具脚本
```

## 🎮 使用方法

### 数据预处理
```bash
# 预处理原始数据
python -m graphnn.data.preprocess

# 生成归一化统计
python -m graphnn.data.gen_mean_std
```

### 模型训练

#### GNN模型（主要）
```bash
# 训练GNN模型
python -m graphnn.train.gnn_vr_2

# 训练改进版GNN
python -m graphnn.train.improved_train_gnn_vr
```

#### 基线模型
```bash
# 训练LSTM基线
python -m graphnn.train.lstm

# 训练GRU基线
python -m graphnn.train.gru
```

### 评估
```bash
# 评估模型性能
python -m graphnn.engine.eval

# 运行滑动窗口推理
python -m graphnn.engine.infer_sliding_gnn
```

### 实时服务器
```bash
# 启动GNN推理服务器
python -m graphnn.server.gnn_socket

# 启动UE4通信服务器
python -m graphnn.server.ue_socket
```

### 可视化
```bash
# 绘制轨迹对比
python -m graphnn.viz.plot_traj_all

# 绘制训练收敛
python -m graphnn.viz.plot_convergence
```

## 🎯 性能指标

| 指标 | 目标值 | 当前状态 |
|------|--------|----------|
| 准确度 | ≥95% | ✅ |
| 推理延迟 | <40ms | ✅ |
| 模型大小 | 优化 | ✅ |

### 基线对比
- **GNN**: 基于图架构的主要模型
- **LSTM**: 用于对比的序列基线
- **GCN**: 图卷积网络基线

## 🔧 配置

项目使用Hydra进行配置管理。关键配置文件位于训练脚本中，可以为不同实验进行自定义。

### 关键参数
- **序列长度**: 50帧（1秒）
- **预测范围**: 未来位置和旋转
- **权重调整**: 基于基线偏移的动态调整
- **归一化半径**: 150cm

## 📈 结果

### 轨迹预测
系统成功以高精度预测教师动作，实现VR环境中的平滑协同控制。

### 权重调整
动态权重调整确保基于性能指标的教师指导与学员自主性之间的最佳平衡。

## 🤝 贡献

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

## 🙏 致谢

- VR教学系统研究团队
- PyTorch Geometric 社区
- 所有贡献者和测试者

## 📞 联系方式

如有问题和支持需求，请在GitHub上开启issue或联系开发团队。

---

**注意**: 本项目专为VR教学应用设计，需要适当的VR硬件和软件才能实现完整功能。

## 📚 技术文档

### 数据说明
- **原始数据**: JSON格式，包含手部位置、旋转、速度等信息
- **预处理**: 转换为32维张量，归一化处理
- **任务场景**: 五角星轨迹触碰（左右手目标球索引固定）

### 模型架构
- **图神经网络**: 主要架构，支持动态图结构
- **时序处理**: 50帧滑动窗口
- **权重调节**: 基于偏移区间的自动调整

### 开发里程碑
- **4.6-4.13**: GNN模型架构设计和基础代码实现
- **4.14-4.20**: 数据清洗与模型训练
- **4.21-4.27**: 超参数调优和性能测试

### 交付物
1. **模型设计**: 架构示意图（含数据流与模块说明）
2. **性能报告**: 准确率/F1/混淆矩阵
3. **代码与数据**: 清洗后数据集、训练代码、权重调整模块
4. **可视化结果**: 轨迹对比动画/GIF、消融实验贡献度图表 