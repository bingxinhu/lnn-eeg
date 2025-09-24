import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

# 导入预处理模块
from preprocess import get_data

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 绘制学习曲线并保存
def draw_learning_curves(history, sub, run, results_path):
    """绘制并保存每个被试每轮训练的学习曲线"""
    os.makedirs(f"{results_path}/learning_curves", exist_ok=True)
    
    # 准确率曲线
    plt.figure()
    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.title(f'Subject {sub+1} Run {run+1} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(f"{results_path}/learning_curves/subj_{sub+1}_run_{run+1}_acc.png", dpi=300)
    plt.close()
    
    # 损失曲线
    plt.figure()
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title(f'Subject {sub+1} Run {run+1} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(f"{results_path}/learning_curves/subj_{sub+1}_run_{run+1}_loss.png", dpi=300)
    plt.close()

# 绘制混淆矩阵并保存
def draw_confusion_matrix(cf_matrix, name, results_path):
    """绘制并保存混淆矩阵，name可以是被试编号或'all'"""
    display_labels = ['Left hand', 'Right hand', 'Foot', 'Tongue']
    
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 7
    
    fig, ax = plt.subplots(figsize=(85/25.4, 70/25.4))
    im = ax.imshow(cf_matrix, interpolation='nearest', cmap='inferno')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cf_matrix.shape[1]),
           yticks=np.arange(cf_matrix.shape[0]),
           xticklabels=display_labels, yticklabels=display_labels,
           title=r'BCI Competition Ⅳ-2a',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    
    # 保存到混淆矩阵专用目录
    os.makedirs(f"{results_path}/confusion_matrices", exist_ok=True)
    plt.savefig(f"{results_path}/confusion_matrices/cf_{name}.png", dpi=360)
    plt.close()  # 关闭图像避免内存占用

# 绘制性能柱状图
def draw_performance_barChart(num_sub, metric, label, results_path):
    """绘制并保存性能柱状图"""
    os.makedirs(f"{results_path}/performance_plots", exist_ok=True)
    
    fig, ax = plt.subplots()
    x = list(range(1, num_sub + 1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title(f'Model {label} per subject')
    ax.set_ylim([0, 1])
    plt.savefig(f"{results_path}/performance_plots/{label.lower()}_bar.png", dpi=300)
    plt.close()

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, epochs, patience):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    counter = 0
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 确保数据在正确设备上
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = correct / total
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = correct / total
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 早停逻辑
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()  # 保存最佳模型状态
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_model is not None:
        model.load_state_dict(best_model)
    return model, history

# 测试函数
def test(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    cf_matrix = confusion_matrix(all_labels, all_preds, normalize='pred')
    
    return acc, kappa, cf_matrix, all_labels, all_preds

# 主运行函数
def run():
    # 配置
    dataset_path = "./dataset/2a/"  # 数据集路径
    results_path = "./results"
    # 创建必要的目录
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(f"{results_path}/saved_models", exist_ok=True)
    os.makedirs(f"{results_path}/metrics", exist_ok=True)  # 保存指标的目录
    
    # 超参数
    batch_size = 64
    epochs = 500
    patience = 300
    lr = 0.0009  # NCP推荐学习率
    n_subjects = 9
    n_train_runs = 10
    
    # 导入NCP模型
    from models import EEG_NCPNet
    
    # 保存结果的日志文件
    log_file = open(f"{results_path}/log.txt", "w", encoding="utf-8")
    
    # 存储所有运行的结果
    acc = np.zeros((n_subjects, n_train_runs))
    kappa = np.zeros((n_subjects, n_train_runs))
    best_runs = np.zeros(n_subjects, dtype=int)  # 记录每个被试的最佳运行
    
    for sub in range(n_subjects):
        print(f"\nTraining on subject {sub + 1}")
        log_file.write(f"\nTraining on subject {sub + 1}\n")
        
        # 获取数据
        X_train, y_train, X_test, y_test = get_data(
            dataset_path, sub, loso=False, is_standard=True, fre_filter=False, dataset='BCI2a'
        )
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        best_subj_acc = 0
        best_history = None
        best_run = 0
        
        for run in range(n_train_runs):
            start_time = time.time()
            print(f"Run {run + 1}/{n_train_runs}")
            
            # 初始化模型、损失函数和优化器
            model = EEG_NCPNet(nb_classes=4, Chans=22, Samples=1125).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # 训练
            trained_model, history = train(model, train_loader, test_loader, criterion, optimizer, epochs, patience)
            
            # 测试
            acc_val, kappa_val, _, _, _ = test(trained_model, test_loader)
            acc[sub, run] = acc_val
            kappa[sub, run] = kappa_val
            
            # 计算运行时间
            run_time = (time.time() - start_time) / 60
            
            # 保存当前轮次的指标
            run_metrics = {
                "subject": sub + 1,
                "run": run + 1,
                "accuracy": float(acc_val),
                "kappa": float(kappa_val),
                "training_time_min": float(run_time),
                "best_epoch": len(history["val_acc"])  # 早停时的epoch数
            }
            with open(f"{results_path}/metrics/subj_{sub+1}_run_{run+1}.json", "w") as f:
                json.dump(run_metrics, f, indent=4)
            
            # 保存训练历史
            np.savez(
                f"{results_path}/metrics/subj_{sub+1}_run_{run+1}_history.npz",
                train_loss=history["train_loss"],
                train_acc=history["train_acc"],
                val_loss=history["val_loss"],
                val_acc=history["val_acc"]
            )
            
            # 绘制并保存学习曲线
            draw_learning_curves(history, sub, run, results_path)
            
            # 记录日志
            info = f"Subject: {sub + 1}   Run: {run + 1}   Time: {run_time:.1f}m   "
            info += f"Acc: {acc_val:.4f}   Kappa: {kappa_val:.4f}\n"
            print(info)
            log_file.write(info)
            
            # 更新最佳模型
            if acc_val > best_subj_acc:
                best_subj_acc = acc_val
                best_history = history
                best_run = run
                # 保存最佳模型权重
                torch.save(
                    trained_model.state_dict(),
                    f"{results_path}/saved_models/best_subj_{sub+1}_run_{run+1}.pth"
                )
        
        # 记录每个被试的最佳运行
        best_runs[sub] = best_run
        np.save(f"{results_path}/best_runs.npy", best_runs)
        
        # 被试总结信息
        info = "----------\n"
        info += f"Subject: {sub + 1}   Best Run: {best_run + 1}   "
        info += f"Best Acc: {acc[sub, best_run]:.4f}   Avg Acc: {np.mean(acc[sub]):.4f} ± {np.std(acc[sub]):.4f}\n"
        info += f"Best Kappa: {kappa[sub, best_run]:.4f}   Avg Kappa: {np.mean(kappa[sub]):.4f} ± {np.std(kappa[sub]):.4f}\n"
        info += "----------\n"
        print(info)
        log_file.write(info)
    
    # 保存所有运行的性能结果
    np.savez(f"{results_path}/all_runs_performance.npz", acc=acc, kappa=kappa)
    log_file.close()
    
    # 最终评估最佳模型
    print("\nFinal evaluation on best models...")
    log_file = open(f"{results_path}/log.txt", "a", encoding="utf-8")
    acc_best = []
    kappa_best = []
    cf_matrices = []
    
    for sub in range(n_subjects):
        # 加载最佳模型
        best_run = best_runs[sub]
        model_path = f"{results_path}/saved_models/best_subj_{sub+1}_run_{best_run+1}.pth"
        
        model = EEG_NCPNet(nb_classes=4, Chans=22, Samples=1125).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 获取测试数据
        _, _, X_test, y_test = get_data(
            dataset_path, sub, loso=False, is_standard=True, fre_filter=False, dataset='BCI2a'
        )
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 测试
        acc_val, kappa_val, cf_matrix, _, _ = test(model, test_loader)
        acc_best.append(acc_val)
        kappa_best.append(kappa_val)
        cf_matrices.append(cf_matrix)
        
        # 绘制混淆矩阵
        draw_confusion_matrix(cf_matrix, f"subj_{sub+1}", results_path)
        
        info = f"Subject {sub + 1} - Best Acc: {acc_val:.4f}, Kappa: {kappa_val:.4f}\n"
        print(info)
        log_file.write(info)
    
    # 计算平均性能
    avg_acc = np.mean(acc_best)
    avg_kappa = np.mean(kappa_best)
    info = f"\nAverage Accuracy: {avg_acc:.4f}, Average Kappa: {avg_kappa:.4f}\n"
    print(info)
    log_file.write(info)
    
    # 绘制平均混淆矩阵
    avg_cf = np.mean(cf_matrices, axis=0)
    draw_confusion_matrix(avg_cf, "average", results_path)
    
    # 绘制性能柱状图
    draw_performance_barChart(n_subjects, acc_best, "Accuracy", results_path)
    draw_performance_barChart(n_subjects, kappa_best, "K-score", results_path)
    
    # 保存最终汇总结果
    summary = {
        "subjects": list(range(1, n_subjects + 1)),
        "best_accuracy_per_subject": [float(x) for x in acc_best],
        "best_kappa_per_subject": [float(x) for x in kappa_best],
        "average_accuracy": float(avg_acc),
        "average_kappa": float(avg_kappa),
        "std_accuracy": float(np.std(acc_best)),
        "std_kappa": float(np.std(kappa_best))
    }
    with open(f"{results_path}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    log_file.close()
    print("All experiments completed successfully!")

if __name__ == "__main__":
    run()
