import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import LTC


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # 通道权重相乘


class LC_Block(nn.Module):
    def __init__(self, F1, kernLength, Chans, D=2, dropout=0.25, activation='elu', AveragePooling=True):
        super(LC_Block, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding='same')
        self.bn1 = nn.BatchNorm2d(F1)
        
        self.dwconv = nn.Conv2d(F1, F1 * D, kernel_size=(Chans, 1), groups=F1)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        
        # 添加通道注意力
        self.attention = ChannelAttention(F1 * D)
        
        self.activation = nn.ELU() if activation == 'elu' else nn.ReLU()
        
        pool_size = (1, kernLength // 8)
        self.pool1 = nn.AvgPool2d(pool_size) if AveragePooling else nn.MaxPool2d(pool_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.sep_conv = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, kernLength // 4), 
                                 padding='same', groups=F1 * D)
        self.bn3 = nn.BatchNorm2d(F1 * D)
        self.pool2 = nn.AvgPool2d(pool_size) if AveragePooling else nn.MaxPool2d(pool_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.dwconv(x)
        x = self.bn2(x)
        x = self.attention(x)  # 应用通道注意力
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


class EEG_NCPNet(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9):
        super(EEG_NCPNet, self).__init__()
        # 多频段适配：为每个频段创建特征提取分支
        self.n_bands = n_bands
        self.lc_blocks = nn.ModuleList([
            nn.Sequential(
                LC_Block(F1=8, kernLength=48, Chans=Chans, dropout=0.3, activation='elu'),
                LC_Block(F1=16, kernLength=64, Chans=Chans, dropout=0.3, activation='elu')
            ) for _ in range(n_bands)
        ])
        
        # 计算特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            feat_dim = 0
            for block in self.lc_blocks:
                x1 = block[0](dummy)
                x2 = block[1](dummy)
                feat_dim += x1.flatten(1).shape[1] + x2.flatten(1).shape[1]
        
        # NCP网络配置
        self.fc = nn.Linear(feat_dim, 128)  # 增加降维维度
        
        # 堆叠LTC层增强时序建模
        self.ncp1 = LTC(input_size=128, units=128)
        self.ncp2 = LTC(input_size=128, units=64)
        self.fc_out = nn.Linear(64, nb_classes)  # 输出层

    def forward(self, x):
        # 多频段特征提取
        band_feats = []
        for i in range(self.n_bands):
            band_data = x[:, i:i+1, ...]  # 提取单个频段 (batch, 1, chans, time)
            x1 = self.lc_blocks[i][0](band_data)
            x2 = self.lc_blocks[i][1](band_data)
            band_feats.append(x1.flatten(1))
            band_feats.append(x2.flatten(1))
        
        # 融合所有频段特征
        x_combined = torch.cat(band_feats, dim=1)
        
        # NCP处理
        x_ncp = self.fc(x_combined).unsqueeze(1)  # (batch, seq_len=1, 128)
        out, _ = self.ncp1(x_ncp)
        out, _ = self.ncp2(out)  # (batch, seq_len=1, 64)
        
        # 输出层映射到类别
        out = self.fc_out(out.squeeze(1))  # 移除时间步维度
        
        return out
