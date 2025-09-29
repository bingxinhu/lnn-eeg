import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleEEGNet(nn.Module):
    """简单但稳定的EEG网络 - 增强正则化版本"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9, dropout_rate=0.6):
        super(SimpleEEGNet, self).__init__()
        self.n_bands = n_bands
        
        # 频段特征提取器 - 所有频段共享权重
        self.band_encoder = nn.Sequential(
            # 时间卷积
            nn.Conv2d(1, 32, kernel_size=(1, 64), padding=(0, 32)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),  # 使用2D dropout
            nn.AvgPool2d(kernel_size=(1, 4)),
            
            # 空间卷积
            nn.Conv2d(32, 64, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AvgPool2d(kernel_size=(1, 4)),
            
            # 深度特征提取
            nn.Conv2d(64, 128, kernel_size=(1, 32), padding=(0, 16)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AvgPool2d(kernel_size=(1, 4)),
        )
        
        # 动态计算特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            out = self.band_encoder(dummy)
            self.feat_dim = out.numel()
        
        print(f"特征维度: {self.feat_dim}")
        
        # 分类器 - 增强正则化
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim * n_bands, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(dropout_rate + 0.1),  # 分类层dropout更高
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(dropout_rate - 0.1),
            nn.Linear(128, nb_classes)
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

class EEGNetV2(nn.Module):
    """另一个稳定的EEG网络变体 - 增强正则化"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9, dropout_rate=0.6):
        super(EEGNetV2, self).__init__()
        
        # 首先合并频段维度
        self.initial_conv = nn.Sequential(
            nn.Conv2d(n_bands, 32, kernel_size=(1, 1)),  # 频段融合
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),  # 使用2D dropout
        )
        
        # 时空特征提取
        self.feature_extractor = nn.Sequential(
            # 时间卷积
            nn.Conv2d(32, 64, kernel_size=(1, 64), padding=(0, 32), groups=32),
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AvgPool2d(kernel_size=(1, 4)),
            
            # 空间卷积
            nn.Conv2d(64, 128, kernel_size=(Chans, 1), groups=64),
            nn.Conv2d(128, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AvgPool2d(kernel_size=(1, 4)),
            
            # 深度特征
            nn.Conv2d(128, 256, kernel_size=(1, 32), padding=(0, 16)),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout_rate + 0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
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
        
        # 合并频段维度
        x = self.initial_conv(x)
        
        # 特征提取
        x = self.feature_extractor(x)
        
        # 展平并分类
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class EnhancedEEGNet(nn.Module):
    """改进的增强版EEG网络 - 更强的正则化"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9, dropout_rate=0.5):
        super(EnhancedEEGNet, self).__init__()
        
        # 增强的特征提取器
        self.feature_extractor = nn.Sequential(
            # 第一层：频段融合
            nn.Conv2d(n_bands, 32, kernel_size=(1, 1)),  # 减少通道数
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),  # 使用2D dropout
            
            # 时间特征提取
            nn.Conv2d(32, 64, kernel_size=(1, 32), padding=(0, 16)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AvgPool2d(kernel_size=(1, 4)),
            
            # 空间特征提取
            nn.Conv2d(64, 128, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AvgPool2d(kernel_size=(1, 4)),
            
            # 深度特征
            nn.Conv2d(128, 256, kernel_size=(1, 16), padding=(0, 8)),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout_rate + 0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, nb_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """更严格的权重初始化"""
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
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class EEG_STFNet(nn.Module):
    """简化版STFNet - 使用SimpleEEGNet的实现"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9):
        super(EEG_STFNet, self).__init__()
        self.model = SimpleEEGNet(nb_classes, Chans, Samples, n_bands, dropout_rate=0.6)
    
    def forward(self, x):
        return self.model(x)

class StableEEGNet(nn.Module):
    """稳定版EEG网络 - 使用EEGNetV2的实现"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9):
        super(StableEEGNet, self).__init__()
        self.model = EEGNetV2(nb_classes, Chans, Samples, n_bands, dropout_rate=0.6)
    
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
