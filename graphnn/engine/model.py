# graphnn/engine/model.py

import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
from torch_geometric.data import HeteroData

class VRGNNTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, gat_layers=3, trans_layers=4, trans_heads=8, dropout=0.1):
        super().__init__()
        # GAT layers
        self.convs = nn.ModuleList()
        d_in = in_dim
        for _ in range(gat_layers):
            conv = HeteroConv({
                ('L_hand','time','L_hand'): SAGEConv(d_in, hidden_dim),
                ('R_hand','time','R_hand'): SAGEConv(d_in, hidden_dim),
                ('L_hand','inter','R_hand'): SAGEConv(d_in, hidden_dim),
                ('R_hand','inter','L_hand'): SAGEConv(d_in, hidden_dim)
            }, aggr='sum')
            self.convs.append(conv)
            d_in = hidden_dim

        # Transformer
        enc_layer = TransformerEncoderLayer(d_model=2*hidden_dim, nhead=trans_heads, dropout=dropout, batch_first=True)
        self.transformer = TransformerEncoder(enc_layer, num_layers=trans_layers)

        # prediction heads
        self.head_pos = nn.Linear(2*hidden_dim, 6)
        self.head_quat = nn.Linear(2*hidden_dim, 8)
        self.head_vel = nn.Linear(2*hidden_dim, 6)
        self.head_dist = nn.Linear(2*hidden_dim, 6)

    def forward(self, data: HeteroData):
        x_dict = {'L_hand': data['L_hand'].x, 'R_hand': data['R_hand'].x}
        edge_index_dict = data.edge_index_dict
        # GNN
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: x.relu() for k, x in x_dict.items()}
        # flatten seq
        L = x_dict['L_hand']  # [N, hidden]
        R = x_dict['R_hand']  # [N, hidden]
        h = torch.cat([L, R], dim=1)  # [N, 2*hidden]
        # batch: [N, 2*hidden], N=seq_len, batch_size graphs in batch
        # 需要将 batch 拆分出来
        batch_size = data['L_hand'].x.shape[0] // data['L_hand'].num_graphs if hasattr(data['L_hand'], 'num_graphs') else 1
        seq_len = data['L_hand'].x.shape[0] // batch_size
        h = h.view(batch_size, seq_len, -1).transpose(0,1)  # [seq_len, batch, 2*hidden]
        out = self.transformer(h)  # [seq_len, batch, 2*hidden]
        h_final = out[-1]          # [batch, 2*hidden]

        p = self.head_pos(h_final)      # [batch, 6]
        q = self.head_quat(h_final)     # [batch, 8]
        # normalize每个batch的左右手四元数
        q_l = q[:, :4] / q[:, :4].norm(dim=1, keepdim=True)
        q_r = q[:, 4:] / q[:, 4:].norm(dim=1, keepdim=True)
        q = torch.cat([q_l, q_r], dim=1)  # [batch, 8]
        v = self.head_vel(h_final)       # [batch, 6]
        d = self.head_dist(h_final)      # [batch, 6]
        return p, q, v, d


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GRUModel, self).__init__()
        
        # GRU层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (batch_size, sequence_length, input_dim)
        out, _ = self.gru(x)  # 获取GRU的输出和最后的隐藏状态
        out = out[:, -1, :]   # 取序列的最后一帧的输出
        out = self.fc(out)    # 通过全连接层得到最终输出
        return out
    

class LSTMModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, output_dim=26, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim) or (batch, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return out


