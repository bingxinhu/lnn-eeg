import numpy as np
import scipy.signal as signal
import torch
from scipy import io
from sklearn.preprocessing import StandardScaler


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


def load_data_BCI2b(data_path, subject, training):
    """加载BCI2b数据集（被试内）"""
    n_channels = 3
    window_length = 8 * 250  # 8秒数据（250Hz采样率）

    data_list = []
    label_list = []

    if training:
        a = io.loadmat(f"{data_path}B0{subject}T.mat")
    else:
        a = io.loadmat(f"{data_path}B0{subject}E.mat")
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
            x1, y1 = load_data_BCI2b(data_path, sub, True)
            x2, y2 = load_data_BCI2b(data_path, sub, False)
        
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
        train_data = X_train[:, 0, j, :].reshape(-1, 1)
        scaler.fit(train_data)
        X_train[:, 0, j, :] = scaler.transform(
            X_train[:, 0, j, :].reshape(-1, 1)
        ).reshape(X_train.shape[0], X_train.shape[3])
        X_test[:, 0, j, :] = scaler.transform(
            X_test[:, 0, j, :].reshape(-1, 1)
        ).reshape(X_test.shape[0], X_test.shape[3])
    return X_train, X_test


def bandpass_filter(data, bandFiltCutF, fs, filtOrder=50, axis=1, filtType='filter'):
    """带通滤波器"""
    if (bandFiltCutF[0] in (0, None)) and (bandFiltCutF[1] in (None, fs/2.0)):
        print("不进行滤波（无效的截止频率设置）")
        return data
    
    # 设计FIR滤波器
    if bandFiltCutF[0] in (0, None):
        print(f"应用低通滤波（截止频率: {bandFiltCutF[1]}Hz）")
        h = signal.firwin(filtOrder + 1, cutoff=bandFiltCutF[1], 
                         pass_zero="lowpass", fs=fs)
    elif bandFiltCutF[1] in (None, fs/2.0):
        print(f"应用高通滤波（截止频率: {bandFiltCutF[0]}Hz）")
        h = signal.firwin(filtOrder + 1, cutoff=bandFiltCutF[0], 
                         pass_zero="highpass", fs=fs)
    else:
        print(f"应用带通滤波（{bandFiltCutF[0]}-{bandFiltCutF[1]}Hz）")
        h = signal.firwin(filtOrder + 1, cutoff=bandFiltCutF, 
                         pass_zero="bandpass", fs=fs)
    
    # 应用滤波
    if filtType == 'filtfilt':
        data_out = signal.filtfilt(h, [1], data, axis=axis)
    else:
        data_out = signal.lfilter(h, [1], data, axis=axis)
    return data_out


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, training=False, augment_prob=0.8):  # 提高增强概率
        self.X = X
        self.y = y
        self.training = training
        self.augment_prob = augment_prob

    def __len__(self):
        """返回数据集样本数量，供DataLoader使用"""
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.training and np.random.rand() < self.augment_prob:
            # 1. 更强的数据增强
            # 时间扭曲
            if np.random.rand() < 0.5:
                x = self.time_warp(x)
            
            # 2. 通道丢弃（更激进）
            if np.random.rand() < 0.4:
                n_channels_to_drop = int(x.shape[1] * 0.3)  # 丢弃30%通道
                channels_to_drop = np.random.choice(
                    x.shape[1], n_channels_to_drop, replace=False
                )
                x[:, channels_to_drop, :] = 0
            
            # 3. 频谱掩码
            if np.random.rand() < 0.3:
                x = self.frequency_mask(x)
                
        return x, y
    
    def time_warp(self, x, max_warp=0.2):
        """时间扭曲增强"""
        batch_size, channels, time_steps = x.shape
        warp_factor = 1 + np.random.uniform(-max_warp, max_warp)
        new_length = int(time_steps * warp_factor)
        
        # 重采样
        x_warped = torch.zeros(batch_size, channels, new_length)
        for i in range(channels):
            original_signal = x[0, i, :].numpy()
            resampled_signal = signal.resample(original_signal, new_length)
            x_warped[0, i, :] = torch.FloatTensor(resampled_signal)
        
        # 截断或填充到原始长度
        if new_length > time_steps:
            return x_warped[:, :, :time_steps]
        else:
            result = torch.zeros(batch_size, channels, time_steps)
            result[:, :, :new_length] = x_warped
            return result
    
    def frequency_mask(self, x, max_mask_fraction=0.2):
        """频率域掩码"""
        # FFT变换
        x_fft = torch.fft.fft(x, dim=-1)
        
        # 随机掩码部分频率
        mask_length = int(x.shape[-1] * max_mask_fraction * np.random.rand())
        mask_start = np.random.randint(0, x.shape[-1] - mask_length)
        
        # 应用掩码
        x_fft_masked = x_fft.clone()
        x_fft_masked[..., mask_start:mask_start+mask_length] = 0
        
        # 逆FFT
        x_masked = torch.fft.ifft(x_fft_masked, dim=-1).real
        return x_masked


def get_data(data_path, subject, loso=False, is_standard=True, fre_filter=True, dataset='BCI2a'):
    """获取预处理后的数据（返回PyTorch张量）"""
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
    else:  # BCI2b
        fs = 250
        t1 = int(2.5 * fs)  # 2.5秒开始
        t2 = int(7 * fs)    # 7秒结束
        T = t2 - t1         # 时间长度（样本数）

        if loso:
            X_train, y_train, X_test, y_test = load_data_loso(data_path, subject, dataset)
        else:
            X_train, y_train = load_data_BCI2b(data_path, subject + 1, True)
            X_test, y_test = load_data_BCI2b(data_path, subject + 1, False)

    # 调整数据形状为 (n_samples, 1, n_channels, n_timepoints)
    n_tr, n_ch, _ = X_train.shape
    X_train = X_train[:, :, t1:t2].reshape(n_tr, 1, n_ch, T)
    
    n_te, _, _ = X_test.shape
    X_test = X_test[:, :, t1:t2].reshape(n_te, 1, n_ch, T)

    # 标准化
    if is_standard:
        X_train, X_test = standardize_data(X_train, X_test, n_ch)

    # 频率滤波（多频段处理）
    if fre_filter:
        filt_banks = [[4, 8], [8, 12], [12, 16], [16, 20], 
                     [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]]
        X_train_temp = np.zeros(X_train.shape + (len(filt_banks),))
        X_test_temp = np.zeros(X_test.shape + (len(filt_banks),))
        
        for i, band in enumerate(filt_banks):
            X_train_temp[..., i] = bandpass_filter(X_train, band, fs, axis=-1)
            X_test_temp[..., i] = bandpass_filter(X_test, band, fs, axis=-1)
        
        # 调整形状为 (n_samples, n_bands, n_channels, n_timepoints)
        X_train = np.transpose(X_train_temp.squeeze(1), (0, 3, 1, 2))
        X_test = np.transpose(X_test_temp.squeeze(1), (0, 3, 1, 2))

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    return X_train, y_train, X_test, y_test
