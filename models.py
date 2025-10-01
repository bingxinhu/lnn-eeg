import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 定义维度重排层
class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims
        
    def forward(self, x):
        return x.permute(*self.dims)

class CrossModalAttention(nn.Module):
    """简化版跨模态注意力机制 - 修复维度问题"""
    
    def __init__(self, channels, n_bands, reduction_ratio=8):
        super(CrossModalAttention, self).__init__()
        self.n_bands = n_bands
        self.channels = channels
        
        # 频段间注意力
        self.band_attention = nn.Sequential(
            nn.Linear(n_bands, max(1, n_bands // reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, n_bands // reduction_ratio), n_bands),
            nn.Sigmoid()
        )
        
        # 通道注意力 - 使用1D卷积处理通道维度
        self.channel_attention = nn.Sequential(
            nn.Conv1d(n_bands, max(1, n_bands // reduction_ratio), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, n_bands // reduction_ratio), n_bands, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 使用2D卷积处理空间维度
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch, n_bands, channels, time)
        batch_size, n_bands, channels, time_len = x.shape
        
        # 频段间注意力
        band_avg = torch.mean(x, dim=[2, 3])  # (batch, n_bands)
        band_weights = self.band_attention(band_avg)  # (batch, n_bands)
        band_weights = band_weights.view(batch_size, n_bands, 1, 1)  # (batch, n_bands, 1, 1)
        x_band = x * band_weights
        
        # 通道注意力 - 在时间维度上平均后应用1D卷积
        channel_avg = torch.mean(x_band, dim=3)  # (batch, n_bands, channels)
        channel_weights = self.channel_attention(channel_avg)  # (batch, n_bands, channels)
        channel_weights = channel_weights.unsqueeze(-1)  # (batch, n_bands, channels, 1)
        x_channel = x_band * channel_weights
        
        # 空间注意力
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)  # (batch, 1, channels, time)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)  # (batch, 1, channels, time)
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # (batch, 2, channels, time)
        spatial_weights = self.spatial_attention(spatial_input)  # (batch, 1, channels, time)
        x_spatial = x_channel * spatial_weights
        
        return x_spatial

class TemporalAttention(nn.Module):
    """简化版时间注意力机制"""
    
    def __init__(self, channels):
        super(TemporalAttention, self).__init__()
        self.channels = channels
        
        self.conv = nn.Conv1d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        batch_size, channels, time_len = x.size()
        
        # 计算注意力权重
        attention = self.conv(x)  # (batch, 1, time)
        attention = self.sigmoid(attention)
        
        # 应用注意力
        out = x * attention
        
        # 残差连接
        out = self.gamma * out + x
        
        return out

class SpatialAttention(nn.Module):
    """简化版空间注意力机制"""
    
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.channels = channels
        
        self.conv = nn.Conv1d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x shape: (batch, channels, spatial)
        batch_size, channels, spatial = x.size()
        
        # 计算注意力权重
        attention = self.conv(x)  # (batch, 1, spatial)
        attention = self.sigmoid(attention)
        
        # 应用注意力
        out = x * attention
        
        # 残差连接
        out = self.gamma * out + x
        
        return out

class SimpleEEGNet(nn.Module):
    """简单但稳定的EEG网络 - 带跨模态注意力"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9, dropout_rate=0.5):
        super(SimpleEEGNet, self).__init__()
        self.n_bands = n_bands
        
        # 跨模态注意力模块
        self.cross_modal_attention = CrossModalAttention(
            channels=Chans, 
            n_bands=n_bands,
            reduction_ratio=4
        )
        
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
        
        # 应用跨模态注意力
        x = self.cross_modal_attention(x)
        
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
    """改进的增强版EEG网络 - 修复维度问题"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9, dropout_rate=0.4):
        super(EnhancedEEGNet, self).__init__()
        
        print(f"初始化EnhancedEEGNet: n_bands={n_bands}, Chans={Chans}, Samples={Samples}")
        
        self.n_bands = n_bands
        self.Chans = Chans
        
        # 跨模态注意力模块
        self.cross_modal_attention = CrossModalAttention(
            channels=Chans, 
            n_bands=n_bands,
            reduction_ratio=4
        )
        
        # 时间注意力
        self.temporal_attention = TemporalAttention(channels=32)
        
        # 空间注意力
        self.spatial_attention = SpatialAttention(channels=64)
        
        # 简化的特征提取器 - 确保输入是4D
        self.feature_extractor = nn.Sequential(
            # 频段融合 - 输入形状: (batch, n_bands, Chans, Samples)
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
            
            # 应用时间注意力
            self._create_temporal_attention_block(),
            
            # 空间特征提取
            nn.Conv2d(32, 64, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.AvgPool2d(kernel_size=(1, 4)),
            
            # 应用空间注意力
            self._create_spatial_attention_block(),
            
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
    
    def _create_temporal_attention_block(self):
        """创建时间注意力块 - 修复版本"""
        class TemporalAttentionBlock(nn.Module):
            def __init__(self, attention_module):
                super(TemporalAttentionBlock, self).__init__()
                self.attention = attention_module
                
            def forward(self, x):
                # x shape: (batch, channels, height, width)
                batch, ch, h, w = x.shape
                # 重塑为 (batch, channels, time) 用于时间注意力
                x_reshaped = x.view(batch, ch, -1)
                x_attended = self.attention(x_reshaped)
                # 恢复形状
                x_out = x_attended.view(batch, ch, h, w)
                return x_out
                
        return TemporalAttentionBlock(self.temporal_attention)
    
    def _create_spatial_attention_block(self):
        """创建空间注意力块 - 修复版本"""
        class SpatialAttentionBlock(nn.Module):
            def __init__(self, attention_module):
                super(SpatialAttentionBlock, self).__init__()
                self.attention = attention_module
                
            def forward(self, x):
                # x shape: (batch, channels, height, width)
                # 对于空间注意力，我们直接在2D特征图上应用空间注意力
                batch, ch, h, w = x.shape
                
                # 重塑为 (batch, channels, spatial) 用于空间注意力
                # 这里spatial = h * w
                x_reshaped = x.view(batch, ch, -1)
                x_attended = self.attention(x_reshaped)
                
                # 恢复形状
                x_out = x_attended.view(batch, ch, h, w)
                return x_out
                
        return SpatialAttentionBlock(self.spatial_attention)
    
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
        # 输入x形状: (batch, n_bands, Chans, Samples)
        # 应用跨模态注意力
        x = self.cross_modal_attention(x)
        
        # 确保x是4D张量
        if x.dim() != 4:
            raise ValueError(f"期望4D输入，但得到 {x.dim()}D 输入")
        
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 其他模型类保持不变...
class EEG_STFNet(nn.Module):
    """简化版STFNet - 使用SimpleEEGNet的实现"""
    def __init__(self, nb_classes=4, Chans=22, Samples=1125, n_bands=9):
        super(EEG_STFNet, self).__init__()
        self.model = SimpleEEGNet(nb_classes, Chans, Samples, n_bands, dropout_rate=0.5)
    
    def forward(self, x):
        return self.model(x)

class StableEEGNet(nn.Module):
    """稳定版EEG网络 - 使用EnhancedEEGNet的实现"""
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
        elif model_type == "AdvancedEEGNet":
            self.model = AdvancedEEGNet(nb_classes, Chans, Samples, n_bands)
        else:
            raise ValueError(f"未知模型类型: {model_type}")
    
    def forward(self, x):
        return self.model(x)

# 默认导出
EEGNet = EEG_STFNet
