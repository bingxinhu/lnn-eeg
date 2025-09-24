import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import LTC  # 注意：低版本LTC的参数不同

class LC_Block(nn.Module):
    def __init__(self, F1, kernLength, Chans, D=2, dropout=0.25, activation='elu', AveragePooling=True):
        super(LC_Block, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding='same')
        self.bn1 = nn.BatchNorm2d(F1)
        
        self.dwconv = nn.Conv2d(F1, F1 * D, kernel_size=(Chans, 1), groups=F1)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        
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
    def __init__(self, nb_classes=4, Chans=22, Samples=1125):
        super(EEG_NCPNet, self).__init__()
        # 特征提取
        self.lc_block1 = LC_Block(F1=8, kernLength=48, Chans=Chans, dropout=0.3, activation='elu')
        self.lc_block2 = LC_Block(F1=16, kernLength=64, Chans=Chans, dropout=0.3, activation='elu')
        
        # 计算特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            feat1 = self.lc_block1(dummy)
            feat2 = self.lc_block2(dummy)
            combined_feat = torch.cat([feat1.flatten(1), feat2.flatten(1)], dim=1)
            ncp_input_size = combined_feat.shape[1]
        
        # NCP网络配置（适配低版本ncps库）
        self.fc = nn.Linear(ncp_input_size, 64)  # 降维到64维
        
        # 关键修正：低版本LTC仅需要input_size和units参数
        self.ncp = LTC(input_size=64, units=64)  # input_size=输入维度，units=隐藏单元数
        self.fc_out = nn.Linear(64, nb_classes)  # 输出层：从隐藏单元映射到类别数

    def forward(self, x):
        # 特征提取与融合
        x1 = self.lc_block1(x)
        x2 = self.lc_block2(x)
        x1_flat = x1.flatten(1)
        x2_flat = x2.flatten(1)
        x_combined = torch.cat([x1_flat, x2_flat], dim=1)
        
        # 适配NCP输入
        x_ncp = self.fc(x_combined).unsqueeze(1)  # (batch, seq_len=1, 64)
        
        # NCP处理
        out, _ = self.ncp(x_ncp)  # (batch, seq_len=1, 64)
        
        # 输出层映射到类别
        out = self.fc_out(out.squeeze(1))  # 移除时间步维度并映射到类别
        
        return out
