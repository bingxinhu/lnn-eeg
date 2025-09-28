import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import LTC
import torch_geometric.nn as gnn
from torch_geometric.data import Data
import numpy as np

class FrequencyAttention(nn.Module):
    """频段注意力模块，学习不同频率带的权重分布"""
    def __init__(self, n_bands):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_bands, n_bands // 2),
            nn.ELU(),
            nn.Linear(n_bands // 2, n_bands),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (batch, n_bands, features)
        b, n_bands, _ = x.size()
        y = self.avg_pool(x.transpose(1, 2)).squeeze(-1)  # (batch, n_bands)
        att_weights = self.fc(y).view(b, n_bands, 1)      # (batch, n_bands, 1)
        return x * att_weights  # 频段加权


class SpatialGraphConv(nn.Module):
    """基于图卷积的EEG空间特征提取（利用电极拓扑结构）"""
    def __init__(self, in_channels, out_channels, adj_matrix):
        super().__init__()
        self.adj = torch.from_numpy(adj_matrix).float()  # 电极邻接矩阵
        self.graph_conv = gnn.GCNConv(in_channels, out_channels)
        
    def forward(self, x):
        # 动态适配输入维度，确保为三维 (batch, chans, time)
        if x.dim() > 3:
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
        
        batch_size, chans, time = x.shape
        
        # 构建批次图数据
        x_out = []
        for t in range(time):
            # 为每个样本构建图数据
            batch_edge_indices = []
            for i in range(batch_size):
                # 获取当前样本的边索引
                edge_index = self.adj.nonzero().t().contiguous()
                # 为当前样本的节点添加偏移量
                offset = i * chans
                edge_index = edge_index + offset
                batch_edge_indices.append(edge_index)
            
            # 合并所有样本的边索引
            edge_index = torch.cat(batch_edge_indices, dim=1)
            
            # 构建当前时间点的图数据
            graph_data = Data(
                x=x[:, :, t].reshape(-1, 1),  # (batch_size * chans, 1)
                edge_index=edge_index.to(x.device)
            )
            
            # 应用图卷积
            graph_out = self.graph_conv(graph_data.x, graph_data.edge_index)
            # 恢复批次维度 (batch_size, chans, 1)
            graph_out = graph_out.reshape(batch_size, chans, -1)
            x_out.append(graph_out)
        
        # 堆叠时间维度 (batch_size, chans, time)
        return torch.stack(x_out, dim=-1).squeeze(2)


class ChannelAttention(nn.Module):
    """改进的通道注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Dropout(0.2),  # 增加dropout减轻过拟合
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = self.fc(avg_y + max_y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """精简空间注意力模块"""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(1)  # 增加批归一化
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv(x_cat)
        x_att = self.bn(x_att)
        return x * self.sigmoid(x_att)


class LC_Block(nn.Module):
    """增强版特征提取块，融合图卷积与注意力"""
    def __init__(self, F1, kernLength, Chans, adj_matrix, D=2, dropout=0.4, activation='elu'):
        super(LC_Block, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding='same')
        self.bn1 = nn.BatchNorm2d(F1)
        self.drop1 = nn.Dropout(dropout/2)
        
        # 加入图卷积层（空间建模）
        self.graph_conv = SpatialGraphConv(in_channels=1, out_channels=F1, adj_matrix=adj_matrix)
        
        self.dwconv = nn.Conv2d(F1, F1 * D, kernel_size=(Chans, 1), groups=F1)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        
        self.channel_att = ChannelAttention(F1 * D)
        self.spatial_att = SpatialAttention()
        
        self.activation = nn.ELU() if activation == 'elu' else nn.ReLU()
        
        pool_size = (1, kernLength // 4)
        self.pool1 = nn.AvgPool2d(pool_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.sep_conv = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, kernLength // 4), 
                                 padding='same', groups=F1 * D)
        self.bn3 = nn.BatchNorm2d(F1 * D)
        self.pool2 = nn.AvgPool2d(pool_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, 1, chans, time)
        batch_size, _, chans, time = x.shape
        
        # 时间卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.activation(x)  # (batch, F1, chans, time)
        
        # 空间图卷积（提取空间拓扑特征）- 添加维度检查
        if x.shape[1] == 1:  # 如果是单通道输入
            x_graph_input = x.squeeze(1)  # (batch, chans, time)
        else:
            x_graph_input = x.mean(dim=1)  # 多通道时取平均
            
        x_graph = self.graph_conv(x_graph_input)  # (batch, chans, time)
        x_graph = x_graph.unsqueeze(1)  # (batch, 1, chans, time)
        
        # 确保维度匹配
        if x_graph.shape[-1] != x.shape[-1]:
            # 进行插值或截断以匹配时间维度
            x_graph = F.interpolate(x_graph, size=x.shape[-1], mode='linear', align_corners=False)
        
        x = x + x_graph  # 特征融合
        
        # 深度可分离卷积与注意力
        x = self.dwconv(x)
        x = self.bn2(x)
        x = self.channel_att(x)
        x = self.spatial_att(x)
        x = self.activation(x)
        
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.sep_conv(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.squeeze(2)  # 移除通道维度 (batch, F1*D, time)
        return x


class EEG_STFNet(nn.Module):
    """时空频融合网络：整合频率注意力、空间图卷积和时序建模"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9):
        super(EEG_STFNet, self).__init__()
        self.n_bands = n_bands
        self.Chans = Chans
        self.Samples = Samples
        
        # 构建电极邻接矩阵（示例：基于BCI2a 22电极的空间距离）
        self.adj_matrix = self._build_adj_matrix(Chans)
        
        # 多频段特征提取
        self.lc_blocks = nn.ModuleList([
            nn.Sequential(
                LC_Block(F1=8, kernLength=64, Chans=Chans, adj_matrix=self.adj_matrix, D=2, dropout=0.4),
                LC_Block(F1=16, kernLength=96, Chans=Chans, adj_matrix=self.adj_matrix, D=2, dropout=0.4)
            ) for _ in range(n_bands)
        ])
        
        # 简化特征维度计算
        # 经过两层LC_Block后的时间维度计算
        time_dim1 = Samples  # 第一层输出时间维度
        time_dim2 = time_dim1 // 4  # 第一次池化后
        time_dim3 = time_dim2 // 4  # 第二次池化后
        self.feat_dim_per_band = 16 * 2 * time_dim3  # F1*D * 时间维度
        
        # 频段注意力融合
        self.freq_att = FrequencyAttention(n_bands)
        
        # 特征融合与降维
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_dim_per_band * n_bands, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.5)
        )
        
        # 时序建模（NCP）
        self.ncp1 = LTC(input_size=128, units=64)
        self.drop_ncp = nn.Dropout(0.3)
        self.fc_out = nn.Linear(64, nb_classes)
        
        # 权重初始化
        self._initialize_weights()

    def _build_adj_matrix(self, n_chans):
        """构建电极邻接矩阵（示例：简单距离矩阵）"""
        adj = np.ones((n_chans, n_chans)) * 0.1  # 基础连接
        np.fill_diagonal(adj, 0)  # 自连接为0
        # 模拟邻近电极强连接
        for i in range(n_chans):
            adj[i, min(i+1, n_chans-1)] = 0.8
            adj[i, max(i-1, 0)] = 0.8
        return adj

    def _initialize_weights(self):
        """权重初始化增强训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (batch, n_bands, chans, time)
        batch_size = x.shape[0]
        band_feats = []
        
        for i in range(self.n_bands):
            band_data = x[:, i:i+1, ...]  # (batch, 1, chans, time)
            x1 = self.lc_blocks[i][0](band_data)
            x2 = self.lc_blocks[i][1](band_data)
            
            # 展平特征
            x1_flat = x1.reshape(batch_size, -1)
            x2_flat = x2.reshape(batch_size, -1)
            
            # 合并特征
            band_feat = torch.cat([x1_flat, x2_flat], dim=1)
            band_feats.append(band_feat.unsqueeze(1))
        
        # 频段融合与注意力加权
        band_feats = torch.cat(band_feats, dim=1)  # (batch, n_bands, feat_dim_per_band)
        band_feats_att = self.freq_att(band_feats)  # 频段注意力加权
        x_combined = band_feats_att.reshape(batch_size, -1)  # 合并所有频段特征
        
        # 特征融合与时序建模
        x_fused = self.fusion(x_combined)
        x_ncp = x_fused.unsqueeze(1)  # (batch, 1, 128)
        out, _ = self.ncp1(x_ncp)
        out = self.drop_ncp(out)
        
        return self.fc_out(out.squeeze(1))


# 备用方案：简化版网络（如果图卷积仍有问题）
class SimpleEEGNet(nn.Module):
    """简化版EEG网络，避免图卷积复杂度"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9):
        super(SimpleEEGNet, self).__init__()
        self.n_bands = n_bands
        
        # 每个频段的特征提取
        self.band_conv = nn.ModuleList([
            nn.Sequential(
                # 第一层卷积
                nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Dropout(0.3),
                # 空间卷积
                nn.Conv2d(16, 32, kernel_size=(Chans, 1), groups=16),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(0.3),
                # 第二层卷积
                nn.Conv2d(32, 32, kernel_size=(1, 32), padding=(0, 16), groups=32),
                nn.Conv2d(32, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8)),
                nn.Dropout(0.3),
            ) for _ in range(n_bands)
        ])
        
        # 计算特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            for conv in self.band_conv:
                out = conv(dummy)
            self.feat_dim = out.numel()
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim * n_bands, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, nb_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, n_bands, chans, time)
        batch_size = x.shape[0]
        band_outputs = []
        
        for i in range(self.n_bands):
            band_data = x[:, i:i+1, :, :]  # (batch, 1, chans, time)
            features = self.band_conv[i](band_data)
            features = features.reshape(batch_size, -1)
            band_outputs.append(features)
        
        # 合并所有频段特征
        combined = torch.cat(band_outputs, dim=1)
        return self.classifier(combined)


# 根据需求选择使用哪个模型
#EEGNet = EEG_STFNet  # 使用完整的时空频融合网络
EEGNet = SimpleEEGNet  # 使用简化版网络（如果图卷积有问题）
