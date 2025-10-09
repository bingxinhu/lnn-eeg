import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SimpleCrossModalAttention(nn.Module):
    """稳定版的跨模态注意力机制"""
    
    def __init__(self, channels, n_bands, reduction_ratio=8):
        super(SimpleCrossModalAttention, self).__init__()
        self.n_bands = n_bands
        self.channels = channels
        
        # 频段注意力
        self.band_fc = nn.Sequential(
            nn.Linear(n_bands, max(4, n_bands // reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, n_bands // reduction_ratio), n_bands),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, max(8, channels // reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, channels // reduction_ratio), channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, n_bands, channels, time_len = x.shape
        
        # 频段注意力 - 在通道和时间维度平均
        band_avg = torch.mean(x, dim=[2, 3])  # (batch, n_bands)
        band_weights = self.band_fc(band_avg)  # (batch, n_bands)
        band_weights = band_weights.view(batch_size, n_bands, 1, 1)
        x = x * band_weights
        
        # 通道注意力 - 在频段和时间维度平均
        channel_avg = torch.mean(x, dim=[1, 3])  # (batch, channels)
        channel_weights = self.channel_fc(channel_avg)  # (batch, channels)
        channel_weights = channel_weights.view(batch_size, 1, channels, 1)
        x = x * channel_weights
        
        return x

class DepthWiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthWiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                  padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.activation(x)

class StableEEGNet(nn.Module):
    """稳定版EEG网络 - 修复所有维度问题"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9, dropout_rate=0.4):
        super(StableEEGNet, self).__init__()
        
        print(f"初始化StableEEGNet: n_bands={n_bands}, Chans={Chans}, Samples={Samples}")
        
        self.n_bands = n_bands
        self.Chans = Chans
        
        # 稳定的跨模态注意力
        self.cross_modal_attention = SimpleCrossModalAttention(
            channels=Chans, 
            n_bands=n_bands,
            reduction_ratio=4
        )
        
        # 特征提取器 - 不使用动态维度计算
        self.feature_extractor = nn.Sequential(
            # 频段融合
            nn.Conv2d(n_bands, 32, kernel_size=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(0.2),
            
            # 时间特征提取
            nn.Conv2d(32, 64, kernel_size=(1, 32), padding=(0, 16)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(0.3),
            nn.AvgPool2d(kernel_size=(1, 4)),
            
            # 空间特征提取
            nn.Conv2d(64, 128, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout2d(0.3),
            nn.AvgPool2d(kernel_size=(1, 4)),
            
            # 深度特征
            nn.Conv2d(128, 256, kernel_size=(1, 16), padding=(0, 8)),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Dropout2d(0.4),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # 手动计算特征维度 (基于网络结构)
        # 经过所有层后的特征维度应该是 256
        self.feat_dim = 256
        
        print(f"特征维度: {self.feat_dim}")
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.4),
            
            nn.Linear(64, nb_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 应用跨模态注意力
        x = self.cross_modal_attention(x)
        
        # 特征提取
        x = self.feature_extractor(x)
        x = x.view(batch_size, -1)
        
        # 分类
        return self.classifier(x)

class EEG_STFNet(nn.Module):
    """STFNet使用稳定版实现"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9):
        super(EEG_STFNet, self).__init__()
        self.model = StableEEGNet(nb_classes, Chans, Samples, n_bands, dropout_rate=0.4)
    
    def forward(self, x):
        return self.model(x)

class EnhancedEEGNet(nn.Module):
    """EnhancedEEGNet使用稳定版实现"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9):
        super(EnhancedEEGNet, self).__init__()
        self.model = StableEEGNet(nb_classes, Chans, Samples, n_bands, dropout_rate=0.35)
    
    def forward(self, x):
        return self.model(x)

# 统一模型入口
class UnifiedEEGModel(nn.Module):
    """统一模型接口"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9, model_type="STFNet"):
        super().__init__()
        if model_type == "STFNet":
            self.model = EEG_STFNet(nb_classes, Chans, Samples, n_bands)
        elif model_type == "StableEEGNet":
            self.model = StableEEGNet(nb_classes, Chans, Samples, n_bands)
        elif model_type == "EnhancedEEGNet":
            self.model = EnhancedEEGNet(nb_classes, Chans, Samples, n_bands)
        else:
            raise ValueError(f"未知模型类型: {model_type}")
    
    def forward(self, x):
        return self.model(x)

# 默认导出
EEGNet = EEG_STFNet
