import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from font_config import set_chinese_font

# 导入自定义模块
from data_preprocessing import load_monuseg_dataset, visualize_samples
from models import create_model, get_model_info
from utils import CombinedLoss, DiceLoss, FocalLoss, IoU, save_model, load_model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                num_epochs=50, save_dir='results', save_interval=5, scheduler_type='plateau'):
    """
    模型训练函数
    
    Args:
        model: 待训练模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 训练设备
        num_epochs: 训练轮数
        save_dir: 保存模型的目录
        save_interval: 模型保存间隔
        scheduler_type: 学习率调度器类型，'plateau', 'cosine', 或 'step'
    
    Returns:
        history: 训练历史记录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'train_iou': [],
        'val_iou': []
    }
    
    start_time = time.time()
    best_val_loss = float('inf')
    best_val_iou = 0.0
    patience_counter = 0
    patience = 15  # 早停的耐心值
    
    # 检查batch_size是否为1，如果是则提示用户可能需要修改BatchNorm的行为
    first_batch = next(iter(train_loader))
    batch_size = first_batch[0].size(0)
    if batch_size == 1:
        print("警告: 批量大小为1，这可能导致BatchNorm层出现问题。" 
              "已将ASPPPooling中的BatchNorm替换为GroupNorm，但模型中其他BatchNorm层可能仍有问题。")
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        
        # 训练一个轮次
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                # 添加梯度裁剪，避免梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 累积损失
                train_loss += loss.item()
                
                # 计算IoU
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                batch_iou = IoU(pred_masks, masks)
                train_iou += batch_iou.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'IoU': f"{batch_iou.item():.4f}"
                })
        
        # 计算平均训练损失和IoU
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                # 计算IoU
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                batch_iou = IoU(pred_masks, masks)
                val_iou += batch_iou.item()
        
        # 计算平均验证损失和IoU
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 根据调度器类型更新学习率
        if scheduler_type == 'plateau':
            scheduler.step(avg_val_loss)
        elif scheduler_type in ['cosine', 'step']:
            scheduler.step()
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)
        history['train_iou'].append(avg_train_iou)
        history['val_iou'].append(avg_val_iou)
        
        # 打印训练信息
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, "
              f"LR: {current_lr:.6f}")
        
        # 保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, optimizer, epoch, save_dir, "best_loss_model.pth")
            print(f"保存最佳损失模型，验证损失: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 保存最佳IoU模型
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            save_model(model, optimizer, epoch, save_dir, "best_iou_model.pth")
            print(f"保存最佳IoU模型，验证IoU: {best_val_iou:.4f}")
            patience_counter = 0
        
        # if (epoch + 1) % save_interval == 0:
        #     save_model(model, optimizer, epoch, save_dir, f"model_epoch_{epoch+1}.pth")
        
        # 早停机制
        if patience_counter >= patience:
            print(f"早停: 验证性能在{patience}轮内未改善")
            break
    
    # 保存最终模型
    save_model(model, optimizer, num_epochs-1, save_dir, "final_model.pth")
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"训练完成! 总时间: {total_time/60:.2f} 分钟")
    
    return history

def plot_learning_curves(history, save_dir='results'):
    """绘制学习曲线"""
    # 设置中文字体
    set_chinese_font()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('损失曲线')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_iou'], label='训练IoU')
    plt.plot(history['val_iou'], label='验证IoU')
    plt.xlabel('轮次')
    plt.ylabel('IoU')
    plt.title('IoU曲线')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()
    
    # 绘制学习率曲线
    plt.figure(figsize=(10, 4))
    plt.plot(history['lr'])
    plt.xlabel('轮次')
    plt.ylabel('学习率')
    plt.title('学习率调度')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'lr_schedule.png'))
    plt.close()
    
    print(f"学习曲线已保存至 {save_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='医学图像分割训练脚本')
    parser.add_argument('--data-dir', type=str, default='data', help='数据集目录')
    parser.add_argument('--model', type=str, default='deeplabv3plus', help='模型名称')
    parser.add_argument('--backbone', type=str, default='resnet50', help='骨干网络')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批大小')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--img-size', type=int, default=512, help='图像大小')
    parser.add_argument('--save-dir', type=str, default='results', help='保存目录')
    parser.add_argument('--no-aug', action='store_true', help='不使用数据增强')
    parser.add_argument('--strong-aug', action='store_true', help='使用强数据增强')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--loss', type=str, default='dice', choices=['dice', 'bce', 'focal', 'combo'], help='损失函数类型')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'step'], help='学习率调度器')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置训练设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    train_loader, val_loader, test_loader, datasets = load_monuseg_dataset(
        args.data_dir, 
        img_size=args.img_size, 
        augmentation=not args.no_aug,
        batch_size=args.batch_size,
        strong_augmentation=args.strong_aug
    )
    
    if train_loader is None:
        print("数据加载失败，请检查数据路径和预处理步骤")
        return
    
    # 创建模型
    model = create_model(
        args.model, 
        num_classes=1, 
        backbone=args.backbone, 
        output_stride=16,
        batch_size=args.batch_size
    )
    model = model.to(device)
    
    # 打印模型信息
    model_info = get_model_info(args.model)
    print(f"模型: {model_info['name']}")
    print(f"描述: {model_info['description']}")
    if 'advantages' in model_info:
        print("优势:")
        for adv in model_info['advantages']:
            print(f"- {adv}")
    
    # 定义损失函数
    if args.loss == 'dice':
        criterion = DiceLoss()
    elif args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2)
    elif args.loss == 'combo':
        criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    else:
        criterion = DiceLoss()
    
    # 定义优化器和学习率调度器
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # 选择学习率调度器
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.5
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
    
    # 训练模型
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        scheduler_type=args.scheduler
    )
    
    # 绘制学习曲线
    plot_learning_curves(history, args.save_dir)
    
    print("训练完成!")

if __name__ == "__main__":
    main() 