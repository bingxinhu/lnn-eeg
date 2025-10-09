import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from preprocess import get_data, EEGDataset

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class ImprovedEarlyStopping:
    """改进的早停机制"""
    def __init__(self, patience=50, min_delta=0.002, window_size=10):
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.best_acc = 0
        self.counter = 0
        self.acc_history = []
    
    def __call__(self, val_acc):
        self.acc_history.append(val_acc)
        
        if len(self.acc_history) > self.window_size:
            # 计算滑动平均
            recent_avg = np.mean(self.acc_history[-self.window_size:])
            if recent_avg > self.best_acc + self.min_delta:
                self.best_acc = recent_avg
                self.counter = 0
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if val_acc > self.best_acc + self.min_delta:
                self.best_acc = val_acc
                self.counter = 0
            return False

def draw_learning_curves(history, sub, run, results_path):
    """绘制并保存学习曲线"""
    os.makedirs(f"{results_path}/learning_curves", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title(f'被试 {sub+1} 第 {run+1} 次运行 准确率曲线')
    plt.ylabel('准确率')
    plt.xlabel('轮次')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{results_path}/learning_curves/subj_{sub+1}_run_{run+1}_acc.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title(f'被试 {sub+1} 第 {run+1} 次运行 损失曲线')
    plt.ylabel('损失值')
    plt.xlabel('轮次')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{results_path}/learning_curves/subj_{sub+1}_run_{run+1}_loss.png", dpi=300)
    plt.close()

def draw_confusion_matrix(cf_matrix, name, results_path, display_labels=None):
    """绘制并保存混淆矩阵"""
    if display_labels is None:
        display_labels = ['左手', '右手', '脚', '舌头']
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cf_matrix, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, label='归一化比例')
    
    plt.xticks(np.arange(cf_matrix.shape[1]), display_labels, rotation=15)
    plt.yticks(np.arange(cf_matrix.shape[0]), display_labels)
    
    thresh = cf_matrix.max() / 2.
    for i in range(cf_matrix.shape[0]):
        for j in range(cf_matrix.shape[1]):
            plt.text(j, i, f"{cf_matrix[i, j]:.2f}",
                    horizontalalignment="center",
                    color="white" if cf_matrix[i, j] > thresh else "black")
    
    plt.title('BCI Competition Ⅳ-2a 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    
    os.makedirs(f"{results_path}/confusion_matrices", exist_ok=True)
    plt.savefig(f"{results_path}/confusion_matrices/cf_{name}.png", dpi=360)
    plt.close()

def draw_performance_barChart(num_sub, metric, label, results_path):
    """绘制性能指标条形图"""
    os.makedirs(f"{results_path}/performance_plots", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    x = list(range(1, num_sub + 1))
    bars = plt.bar(x, metric, 0.5, label=label)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.ylabel(label)
    plt.xlabel("被试编号")
    plt.xticks(x)
    plt.title(f'各被试{label}表现')
    plt.ylim([0, 1.05])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_path}/performance_plots/{label.lower()}_bar.png", dpi=300)
    plt.close()

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, device):
    """优化的训练函数"""
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    counter = 0
    best_model_weights = None
    
    # 减少梯度累积步数
    accumulation_steps = 2
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f"轮次 {epoch+1}/{epochs} 训练", leave=False)
        
        optimizer.zero_grad()
        
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * inputs.size(0) * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            current_acc = correct / total
            train_pbar.set_postfix({
                "损失": f"{loss.item() * accumulation_steps:.4f}", 
                "准确率": f"{current_acc:.4f}"
            })
        
        # 处理剩余的梯度
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        train_acc = correct / total
        train_loss /= total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"轮次 {epoch+1}/{epochs} 验证", leave=False)
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                current_val_acc = correct / total
                val_pbar.set_postfix({
                    "损失": f"{loss.item():.4f}", 
                    "准确率": f"{current_val_acc:.4f}"
                })
        
        val_acc = correct / total
        val_loss /= total
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印训练信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f"轮次 {epoch+1}/{epochs} | 训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
              f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f} | 学习率: {current_lr:.6f}")
        
        # 改进的早停机制 - 放宽条件
        min_delta = 0.001  # 降低阈值
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()  # 使用copy()
            counter = 0
            print(f"↑ 新的最佳验证准确率: {val_acc:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print(f"早停于轮次 {epoch+1}, 最佳验证准确率: {best_val_acc:.4f}")
                break
    
    # 加载最佳模型权重
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    return model, history

def test(model, test_loader, device):
    """模型测试函数"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="测试中", leave=False)
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    cf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    
    return acc, kappa, cf_matrix, class_report, all_labels, all_preds

def get_model(model_name, nb_classes=4, Chans=22, Samples=1125, n_bands=9):
    """根据模型名称获取对应的模型实例"""
    from models import EEG_STFNet, StableEEGNet, EnhancedEEGNet
    
    if model_name == "STFNet":
        return EEG_STFNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples, n_bands=n_bands)
    elif model_name == "StableEEGNet":
        return StableEEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples, n_bands=n_bands)
    elif model_name == "EnhancedEEGNet":
        return EnhancedEEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples, n_bands=n_bands)
    else:
        raise ValueError(f"未知模型: {model_name}")

def run(args):
    """主运行函数"""
    # 配置参数
    dataset_path = args.dataset_path
    results_path = f"./results_{args.model}"
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(f"{results_path}/saved_models", exist_ok=True)
    os.makedirs(f"{results_path}/metrics", exist_ok=True)
    
    # 超参数设置
    batch_size = args.batch_size
    epochs = args.epochs
    patience = args.patience
    lr = args.lr
    n_subjects = args.n_subjects
    n_train_runs = args.n_runs
    val_split = args.val_split
    
    # 日志文件
    log_file = open(f"{results_path}/log.txt", "w", encoding="utf-8")
    log_file.write("实验参数配置:\n")
    for arg in vars(args):
        log_file.write(f"  {arg}: {getattr(args, arg)}\n")
    log_file.write("\n")
    
    # 存储结果的数组
    acc = np.zeros((n_subjects, n_train_runs))
    kappa = np.zeros((n_subjects, n_train_runs))
    best_runs = np.zeros(n_subjects, dtype=int)
    
    # 数据缓存字典
    data_cache = {}
    
    for sub in range(n_subjects):
        print(f"\n正在训练被试 {sub + 1}")
        log_file.write(f"\n正在训练被试 {sub + 1}\n")
        
        # 加载数据 - 确保使用9个频段
        if sub not in data_cache:
            X_train, y_train, X_test, y_test = get_data(
                dataset_path, sub, loso=True, is_standard=True, 
                fre_filter=True, dataset='BCI2a', use_9_bands=True  # 确保使用9个频段
            )
            data_cache[sub] = (X_train, y_train, X_test, y_test)
        else:
            X_train, y_train, X_test, y_test = data_cache[sub]
        
        # 获取实际的频段数量
        n_bands_actual = X_train.shape[1]
        print(f"实际频段数量: {n_bands_actual}")
        print(f"X_train形状: {X_train.shape}")
        print(f"X_test形状: {X_test.shape}")
        
        # 检查数据平衡性
        unique, counts = np.unique(y_train.numpy(), return_counts=True)
        print(f"训练集类别分布: {dict(zip(unique, counts))}")
        
        # 划分训练集和验证集
        train_size = int((1 - val_split) * len(X_train))
        val_size = len(X_train) - train_size
        X_tr, X_val = X_train[:train_size], X_train[train_size:]
        y_tr, y_val = y_train[:train_size], y_train[train_size:]
        
        # 创建数据集和数据加载器
        train_dataset = EEGDataset(X_tr, y_tr, training=True, augment_prob=args.augment_prob)
        val_dataset = EEGDataset(X_val, y_val, training=False)
        test_dataset = EEGDataset(X_test, y_test, training=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=args.num_workers, pin_memory=True)
        
        best_subj_acc = 0.0
        best_run = 0
        
        for run in range(n_train_runs):
            start_time = time.time()
            print(f"第 {run + 1}/{n_train_runs} 次运行")
            
            # 使用实际的频段数量初始化模型
            model = get_model(args.model, nb_classes=4, Chans=22, Samples=1125, n_bands=n_bands_actual).to(device)
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
            
            # 使用余弦退火学习率调度
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=2, eta_min=1e-6
            )
            
            # 训练模型
            trained_model, history = train(
                model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, device
            )
            
            # 在测试集上评估
            acc_val, kappa_val, _, class_report, _, _ = test(trained_model, test_loader, device)
            acc[sub, run] = acc_val
            kappa[sub, run] = kappa_val
            
            run_time = (time.time() - start_time) / 60
            
            # 保存运行指标
            run_metrics = {
                "subject": sub + 1,
                "run": run + 1,
                "accuracy": float(acc_val),
                "kappa": float(kappa_val),
                "training_time_min": float(run_time),
                "best_epoch": len(history["val_acc"]),
                "class_report": class_report
            }
            with open(f"{results_path}/metrics/subj_{sub+1}_run_{run+1}.json", "w", encoding="utf-8") as f:
                json.dump(run_metrics, f, indent=4, ensure_ascii=False)
            
            # 保存训练历史
            np.savez(
                f"{results_path}/metrics/subj_{sub+1}_run_{run+1}_history.npz",
                train_loss=history["train_loss"],
                train_acc=history["train_acc"],
                val_loss=history["val_loss"],
                val_acc=history["val_acc"]
            )
            
            # 绘制学习曲线
            draw_learning_curves(history, sub, run, results_path)
            
            # 记录日志
            info = f"被试: {sub + 1}   运行: {run + 1}   时间: {run_time:.1f}分钟   "
            info += f"准确率: {acc_val:.4f}   Kappa: {kappa_val:.4f}\n"
            print(info)
            log_file.write(info)
            
            # 保存最佳模型
            if acc_val > best_subj_acc:
                best_subj_acc = acc_val
                best_run = run
                torch.save(
                    trained_model.state_dict(),
                    f"{results_path}/saved_models/best_subj_{sub+1}_run_{run+1}.pth"
                )
        
        # 记录被试的最佳结果
        best_runs[sub] = best_run
        np.save(f"{results_path}/best_runs.npy", best_runs)
        
        # 统计被试的平均表现
        info = "----------\n"
        info += f"被试: {sub + 1}   最佳运行: {best_run + 1}   "
        info += f"最佳准确率: {acc[sub, best_run]:.4f}   平均准确率: {np.mean(acc[sub]):.4f} ± {np.std(acc[sub]):.4f}\n"
        info += f"最佳Kappa: {kappa[sub, best_run]:.4f}   平均Kappa: {np.mean(kappa[sub]):.4f} ± {np.std(kappa[sub]):.4f}\n"
        info += "----------\n"
        print(info)
        log_file.write(info)
    
    # 保存所有运行结果
    np.savez(f"{results_path}/all_runs_performance.npz", acc=acc, kappa=kappa)
    log_file.close()
    
    # 最佳模型的最终评估
    print("\n在最佳模型上进行最终评估...")
    log_file = open(f"{results_path}/log.txt", "a", encoding="utf-8")
    acc_best = []
    kappa_best = []
    cf_matrices = []
    class_reports = []
    
    for sub in range(n_subjects):
        best_run = best_runs[sub]
        model_path = f"{results_path}/saved_models/best_subj_{sub+1}_run_{best_run+1}.pth"
        
        # 获取该被试的实际频段数量
        _, _, X_test, y_test = data_cache[sub]
        n_bands_actual = X_test.shape[1]
        
        # 加载最佳模型 - 使用正确的频段数量
        model = get_model(args.model, nb_classes=4, Chans=22, Samples=1125, n_bands=n_bands_actual).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 获取测试数据
        test_dataset = EEGDataset(X_test, y_test, training=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
        
        # 评估模型
        acc_val, kappa_val, cf_matrix, class_report, _, _ = test(model, test_loader, device)
        acc_best.append(acc_val)
        kappa_best.append(kappa_val)
        cf_matrices.append(cf_matrix)
        class_reports.append(class_report)
        
        # 绘制混淆矩阵
        draw_confusion_matrix(cf_matrix, f"subj_{sub+1}", results_path)
        
        # 记录最佳模型评估结果
        info = f"被试 {sub + 1} - 最佳准确率: {acc_val:.4f}, Kappa: {kappa_val:.4f}\n"
        print(info)
        log_file.write(info)
    
    # 计算平均性能
    avg_acc = np.mean(acc_best)
    avg_kappa = np.mean(kappa_best)
    info = f"\n平均准确率: {avg_acc:.4f}, 平均Kappa: {avg_kappa:.4f}\n"
    print(info)
    log_file.write(info)
    
    # 绘制平均混淆矩阵
    avg_cf = np.mean(cf_matrices, axis=0)
    draw_confusion_matrix(avg_cf, "average", results_path)
    
    # 绘制性能条形图
    draw_performance_barChart(n_subjects, acc_best, "准确率", results_path)
    draw_performance_barChart(n_subjects, kappa_best, "K系数", results_path)
    
    # 保存总结报告
    summary = {
        "subjects": list(range(1, n_subjects + 1)),
        "best_accuracy_per_subject": [float(x) for x in acc_best],
        "best_kappa_per_subject": [float(x) for x in kappa_best],
        "average_accuracy": float(avg_acc),
        "average_kappa": float(avg_kappa),
        "std_accuracy": float(np.std(acc_best)),
        "std_kappa": float(np.std(kappa_best)),
        "class_reports": class_reports,
        "parameters": vars(args)
    }
    with open(f"{results_path}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    log_file.close()
    print("所有实验已成功完成!")

def parse_args():
    """优化训练参数"""
    parser = argparse.ArgumentParser(description='EEG分类模型训练与评估')
    parser.add_argument('--dataset_path', type=str, default="./dataset/2a/", 
                       help='数据集路径')
    parser.add_argument('--model', type=str, default="EnhancedEEGNet",
                       choices=["STFNet", "StableEEGNet", "EnhancedEEGNet"],
                       help='选择模型')
    parser.add_argument('--batch_size', type=int, default=16, help='减少批处理大小')  # 从64减少到32
    parser.add_argument('--epochs', type=int, default=500, help='增加训练轮次')  # 增加到200
    parser.add_argument('--patience', type=int, default=80, help='增加早停耐心值')  # 增加到80
    parser.add_argument('--lr', type=float, default=1e-3, help='增加学习率')  # 增加到1e-3
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='减少权重衰减')  # 减少到1e-4
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='减少标签平滑')  # 减少到0.1
    parser.add_argument('--n_subjects', type=int, default=9, help='被试数量')
    parser.add_argument('--n_runs', type=int, default=3, help='每个被试的运行次数')
    parser.add_argument('--val_split', type=float, default=0.15, help='减少验证集比例')  # 减少到0.15
    parser.add_argument('--augment_prob', type=float, default=0.5, help='增加数据增强概率')  # 增加到0.5
    parser.add_argument('--num_workers', type=int, default=4, help='增加数据加载线程数')  # 增加到4
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
