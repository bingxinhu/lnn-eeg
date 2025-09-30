# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleEEGNet(nn.Module):
    """简单但稳定的EEG网络"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9, dropout_rate=0.5):
        super(SimpleEEGNet, self).__init__()
        self.n_bands = n_bands
        
        # 频段特征提取器
        self.band_encoder = nn.Sequential(
            # 时间卷积
            nn.Conv2d(1, 16, kernel_size=(1, 32), padding=(0, 16)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AvgPool2d(kernel_size=(1, 4)),
            
            # 空间卷积
            nn.Conv2d(16, 32, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AvgPool2d(kernel_size=(1, 4)),
            
            # 深度特征提取
            nn.Conv2d(32, 64, kernel_size=(1, 16), padding=(0, 8)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # 动态计算特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            out = self.band_encoder(dummy)
            self.feat_dim = out.numel()
        
        print(f"特征维度: {self.feat_dim}, 频段数量: {n_bands}")
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim * n_bands, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(dropout_rate - 0.1),
            nn.Linear(64, nb_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch, n_bands, chans, time)
        batch_size = x.shape[0]
        band_outputs = []
        
        # 对每个频段应用相同的特征提取器
        for i in range(self.n_bands):
            band_data = x[:, i:i+1, :, :]  # (batch, 1, chans, time)
            features = self.band_encoder(band_data)
            features = features.reshape(batch_size, -1)
            band_outputs.append(features)
        
        # 合并所有频段特征
        combined = torch.cat(band_outputs, dim=1)
        return self.classifier(combined)

class EnhancedEEGNet(nn.Module):
    """改进的增强版EEG网络"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9, dropout_rate=0.4):
        super(EnhancedEEGNet, self).__init__()
        
        print(f"初始化EnhancedEEGNet: n_bands={n_bands}, Chans={Chans}, Samples={Samples}")
        
        # 简化的特征提取器
        self.feature_extractor = nn.Sequential(
            # 频段融合 - 输入通道数应该是n_bands
            nn.Conv2d(n_bands, 16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            
            # 时间特征提取
            nn.Conv2d(16, 32, kernel_size=(1, 32), padding=(0, 16)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AvgPool2d(kernel_size=(1, 4)),
            
            # 空间特征提取
            nn.Conv2d(32, 64, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AvgPool2d(kernel_size=(1, 4)),
            
            # 深度特征
            nn.Conv2d(64, 128, kernel_size=(1, 16), padding=(0, 8)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # 简化分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout_rate - 0.1),
            nn.Linear(32, nb_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # print(f"模型输入形状: {x.shape}")  # 调试信息
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class EEG_STFNet(nn.Module):
    """简化版STFNet"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9):
        super(EEG_STFNet, self).__init__()
        self.model = SimpleEEGNet(nb_classes, Chans, Samples, n_bands, dropout_rate=0.5)
    
    def forward(self, x):
        return self.model(x)

class StableEEGNet(nn.Module):
    """稳定版EEG网络"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9):
        super(StableEEGNet, self).__init__()
        self.model = EnhancedEEGNet(nb_classes, Chans, Samples, n_bands, dropout_rate=0.4)
    
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
