# optimize_training.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

def get_optimized_training_config():
    """返回优化的训练配置"""
    config = {
        'learning_rate': 5e-4,  # 降低学习率
        'batch_size': 64,       # 增加批大小
        'weight_decay': 1e-3,   # 增加权重衰减
        'label_smoothing': 0.2, # 增加标签平滑
        'dropout_rate': 0.5,    # 增加dropout
        'patience': 50,         # 增加早停耐心
    }
    return config

def create_improved_optimizer(model, lr=5e-4, weight_decay=1e-3):
    """创建改进的优化器"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    return optimizer

def create_improved_scheduler(optimizer, epochs=300):
    """创建改进的学习率调度器"""
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        epochs=epochs,
        steps_per_epoch=1,  # 会在训练中更新
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )
    return scheduler

def improved_train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, device):
    """改进的训练函数"""
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    counter = 0
    best_model_weights = None
    
    # 梯度累积步数
    accumulation_steps = 4
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # 梯度累积
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
            
            train_loss += loss.item() * accumulation_steps * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 处理剩余的梯度
        if total % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        train_acc = correct / total
        train_loss /= total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = correct / total
        val_loss /= total
        
        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印训练信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f"轮次 {epoch+1}/{epochs} | 训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
              f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f} | 学习率: {current_lr:.6f}")
        
        # 改进的早停机制
        min_delta = 0.002
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()
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

def analyze_current_training(log_file_path):
    """分析当前训练日志"""
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 提取准确率信息
        val_accuracies = []
        for line in lines:
            if '验证准确率:' in line:
                parts = line.split('验证准确率:')
                if len(parts) > 1:
                    acc_str = parts[1].split('|')[0].strip()
                    try:
                        val_acc = float(acc_str)
                        val_accuracies.append(val_acc)
                    except ValueError:
                        continue
        
        if val_accuracies:
            print(f"\n训练分析报告:")
            print(f"总轮次: {len(val_accuracies)}")
            print(f"最高验证准确率: {max(val_accuracies):.4f}")
            print(f"最低验证准确率: {min(val_accuracies):.4f}")
            print(f"平均验证准确率: {sum(val_accuracies)/len(val_accuracies):.4f}")
            print(f"最后10轮平均: {sum(val_accuracies[-10:])/len(val_accuracies[-10:]):.4f}")
            
            # 稳定性分析
            fluctuations = sum(1 for i in range(1, len(val_accuracies)) 
                            if abs(val_accuracies[i] - val_accuracies[i-1]) > 0.05)
            print(f"大幅波动次数 (>0.05): {fluctuations}")
            
            return val_accuracies
        else:
            print("无法从日志中提取准确率信息")
            return None
            
    except FileNotFoundError:
        print(f"日志文件不存在: {log_file_path}")
        return None

if __name__ == "__main__":
    # 分析当前训练
    model_name = "EnhancedEEGNet"
    log_file = f"./results_{model_name}/log.txt"
    val_accuracies = analyze_current_training(log_file)
    
    # 提供优化建议
    config = get_optimized_training_config()
    print(f"\n优化建议:")
    print(f"1. 降低学习率: {config['learning_rate']}")
    print(f"2. 增加批大小: {config['batch_size']}")
    print(f"3. 增加权重衰减: {config['weight_decay']}")
    print(f"4. 增加标签平滑: {config['label_smoothing']}")
    print(f"5. 增加Dropout: {config['dropout_rate']}")
    print(f"6. 增加早停耐心: {config['patience']}")
