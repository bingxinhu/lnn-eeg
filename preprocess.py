# preprocess.py
import numpy as np
import scipy.signal as signal
import torch
from scipy import io
from sklearn.preprocessing import StandardScaler
import mne

def load_data_BCI2a(data_path, subject, training):
    """加载BCI2a数据集（被试内）"""
    n_channels = 22
    window_length = 7 * 250  # 7秒数据（250Hz采样率）

    data_list = []
    label_list = []

    if training:
        a = io.loadmat(f"{data_path}A0{subject}T.mat")
    else:
        a = io.loadmat(f"{data_path}A0{subject}E.mat")
    a_data = a["data"]

    for ii in range(a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]  # EEG数据
        a_trial = a_data3[1]  # 试次起始索引
        a_y = a_data3[2]  # 标签
        a_artifacts = a_data3[5]  # 伪影标记

        for trial in range(a_trial.size):
            if a_artifacts[trial] == 0:  # 排除有伪影的试次
                start_idx = int(a_trial[trial].item())
                end_idx = start_idx + window_length
                eeg_data = np.transpose(a_X[start_idx:end_idx, :n_channels])
                data_list.append(eeg_data)
                label_list.append(int(a_y[trial].item()) - 1)

    return np.array(data_list), np.array(label_list)

def load_data_loso(data_path, subject, dataset='BCI2a'):
    """留一法交叉验证数据加载"""
    X_train, y_train = [], []
    X_test, y_test = None, None

    for sub in range(1, 10):  # 9个被试
        if dataset == 'BCI2a':
            x1, y1 = load_data_BCI2a(data_path, sub, True)
            x2, y2 = load_data_BCI2a(data_path, sub, False)
        else:
            # 如果是BCI2b，使用相应的加载函数
            x1, y1 = load_data_BCI2a(data_path, sub, True)
            x2, y2 = load_data_BCI2a(data_path, sub, False)
        
        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        if sub == subject + 1:  # 当前被试作为测试集
            X_test, y_test = x, y
        else:  # 其他被试作为训练集
            X_train.append(x)
            y_train.append(y)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    return X_train, y_train, X_test, y_test

def standardize_data(X_train, X_test, channels):
    """标准化数据（按通道）"""
    for j in range(channels):
        scaler = StandardScaler()
        # 注意：这里需要根据实际数据形状调整
        if X_train.ndim == 4:
            train_data = X_train[:, 0, j, :].reshape(-1, 1)
            test_data = X_test[:, 0, j, :].reshape(-1, 1)
        else:
            train_data = X_train[:, j, :].reshape(-1, 1)
            test_data = X_test[:, j, :].reshape(-1, 1)
            
        scaler.fit(train_data)
        
        if X_train.ndim == 4:
            X_train[:, 0, j, :] = scaler.transform(
                X_train[:, 0, j, :].reshape(-1, 1)
            ).reshape(X_train.shape[0], X_train.shape[3])
            X_test[:, 0, j, :] = scaler.transform(
                X_test[:, 0, j, :].reshape(-1, 1)
            ).reshape(X_test.shape[0], X_test.shape[3])
        else:
            X_train[:, j, :] = scaler.transform(
                X_train[:, j, :].reshape(-1, 1)
            ).reshape(X_train.shape[0], X_train.shape[2])
            X_test[:, j, :] = scaler.transform(
                X_test[:, j, :].reshape(-1, 1)
            ).reshape(X_test.shape[0], X_test.shape[2])
    
    return X_train, X_test

def bandpass_filter(data, bandFiltCutF, fs, filtOrder=4):
    """简化的带通滤波器"""
    if (bandFiltCutF[0] in (0, None)) and (bandFiltCutF[1] in (None, fs/2.0)):
        return data
    
    # 使用scipy的滤波器
    nyquist = fs / 2
    low = bandFiltCutF[0] / nyquist
    high = bandFiltCutF[1] / nyquist
    
    if low == 0:
        b, a = signal.butter(filtOrder, high, btype='low')
    elif high >= 1:
        b, a = signal.butter(filtOrder, low, btype='high')
    else:
        b, a = signal.butter(filtOrder, [low, high], btype='band')
    
    # 处理不同维度的数据
    if data.ndim == 4:  # (samples, 1, channels, time)
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                filtered_data[i, 0, j, :] = signal.filtfilt(b, a, data[i, 0, j, :])
    elif data.ndim == 3:  # (samples, channels, time)
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                filtered_data[i, j, :] = signal.filtfilt(b, a, data[i, j, :])
    else:
        filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, training=False, augment_prob=0.2):
        self.X = X
        self.y = y
        self.training = training
        self.augment_prob = augment_prob
        self.fs = 250

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.training and np.random.rand() < self.augment_prob:
            augment_methods = [
                self.add_noise,
                self.random_scale,
            ]
            augment = np.random.choice(augment_methods)
            x = augment(x)
        
        return x, y
    
    def add_noise(self, x, noise_level=0.01):
        """添加高斯噪声"""
        noise = torch.randn_like(x) * noise_level
        return x + noise
    
    def random_scale(self, x, scale_range=(0.9, 1.1)):
        """随机缩放增强"""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return x * scale_factor

def get_data(data_path, subject, loso=False, is_standard=True, fre_filter=True, dataset='BCI2a', use_9_bands=True):
    """获取预处理后的数据
    
    参数:
        use_9_bands: 是否使用9个频段 (True) 还是4个频段 (False)
    """
    if dataset == 'BCI2a':
        fs = 250
        t1 = int(1.5 * fs)  # 1.5秒开始
        t2 = int(6 * fs)    # 6秒结束
        T = t2 - t1         # 时间长度（样本数）

        if loso:
            X_train, y_train, X_test, y_test = load_data_loso(data_path, subject, dataset=dataset)
        else:
            X_train, y_train = load_data_BCI2a(data_path, subject + 1, True)
            X_test, y_test = load_data_BCI2a(data_path, subject + 1, False)
    else:
        fs = 250
        t1 = int(2.5 * fs)  # 2.5秒开始
        t2 = int(7 * fs)    # 7秒结束
        T = t2 - t1         # 时间长度（样本数）

        if loso:
            X_train, y_train, X_test, y_test = load_data_loso(data_path, subject, dataset)
        else:
            X_train, y_train = load_data_BCI2a(data_path, subject + 1, True)
            X_test, y_test = load_data_BCI2a(data_path, subject + 1, False)

    # 调整数据形状为 (n_samples, n_channels, n_timepoints)
    n_tr, n_ch, _ = X_train.shape
    X_train = X_train[:, :, t1:t2]  # 形状: (n_samples, n_channels, T)
    
    n_te, _, _ = X_test.shape
    X_test = X_test[:, :, t1:t2]    # 形状: (n_samples, n_channels, T)

    # 标准化
    if is_standard:
        X_train, X_test = standardize_data(X_train, X_test, n_ch)

    # 频率滤波（多频段处理）
    if fre_filter:
        # 根据参数选择使用9个频段还是4个频段
        if use_9_bands:
            filt_banks = [[4, 8], [8, 12], [12, 16], [16, 20], 
                         [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]]
        else:
            filt_banks = [[4, 8], [8, 13], [13, 20], [20, 30]]
            
        X_train_temp = np.zeros((X_train.shape[0], len(filt_banks), n_ch, T))
        X_test_temp = np.zeros((X_test.shape[0], len(filt_banks), n_ch, T))
        
        for i, band in enumerate(filt_banks):
            # 对每个样本的每个通道进行滤波
            for sample_idx in range(X_train.shape[0]):
                for channel_idx in range(n_ch):
                    X_train_temp[sample_idx, i, channel_idx, :] = bandpass_filter(
                        X_train[sample_idx, channel_idx, :].reshape(1, -1), 
                        band, fs
                    )
            
            for sample_idx in range(X_test.shape[0]):
                for channel_idx in range(n_ch):
                    X_test_temp[sample_idx, i, channel_idx, :] = bandpass_filter(
                        X_test[sample_idx, channel_idx, :].reshape(1, -1), 
                        band, fs
                    )
        
        # 最终形状: (n_samples, n_bands, n_channels, n_timepoints)
        X_train = X_train_temp
        X_test = X_test_temp
    else:
        # 如果没有滤波，添加一个频段维度
        X_train = X_train[:, np.newaxis, :, :]  # (n_samples, 1, n_channels, T)
        X_test = X_test[:, np.newaxis, :, :]    # (n_samples, 1, n_channels, T)

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    print(f"数据形状 - X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"频段数量: {X_train.shape[1]}")

    return X_train, y_train, X_test, y_test
