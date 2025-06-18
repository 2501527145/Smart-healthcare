import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from font_config import set_chinese_font

# 导入自定义模块
from data_preprocessing import load_monuseg_dataset
from models import create_model
from utils import IoU, DiceCoefficient, load_model

set_chinese_font()

def evaluate_model(model, test_loader, device, save_dir='results'):
    """
    在测试集上评估模型
    
    Args:
        model: 待评估模型
        test_loader: 测试数据加载器
        device: 使用设备
        save_dir: 结果保存目录
    
    Returns:
        metrics: 评估指标字典
    """
    model.eval()
    
    # 初始化指标
    total_iou = 0.0
    total_dice = 0.0
    all_preds = []
    all_targets = []
    
    # 创建保存可视化结果的目录
    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)
    
    with torch.no_grad():
        with tqdm(test_loader, desc="评估进度") as pbar:
            for i, (images, masks) in enumerate(pbar):
                images = images.to(device)
                masks = masks.to(device)
                
                # 模型预测
                outputs = model(images)
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                
                # 计算指标
                batch_iou = IoU(pred_masks, masks)
                batch_dice = DiceCoefficient(pred_masks, masks)
                
                total_iou += batch_iou.item()
                total_dice += batch_dice.item()
                
                # 收集所有预测和目标用于计算整体指标
                all_preds.extend(pred_masks.view(-1).cpu().numpy())
                all_targets.extend(masks.view(-1).cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({'IoU': f"{batch_iou.item():.4f}", 'Dice': f"{batch_dice.item():.4f}"})
                
                # 可视化部分结果
                if i < 10:  # 只展示前10个批次的结果
                    visualize_batch_predictions(images, masks, pred_masks, i, save_dir)
    
    # 计算平均指标
    avg_iou = total_iou / len(test_loader)
    avg_dice = total_dice / len(test_loader)
    
    # 将列表转换为二值数组
    all_preds = np.array(all_preds) > 0.5
    all_targets = np.array(all_targets) > 0.5
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds, labels=[0, 1]).ravel()
    
    # 计算其他指标
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # 创建指标字典
    metrics = {
        'IoU': avg_iou,
        'Dice': avg_dice,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'True Positive': tp,
        'False Positive': fp,
        'True Negative': tn,
        'False Negative': fn
    }
    
    # 打印测试结果
    print("\n测试集评估结果:")
    print(f"IoU: {avg_iou:.4f}")
    print(f"Dice系数: {avg_dice:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 保存评估结果到文件
    with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("测试集评估结果:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    return metrics

def visualize_batch_predictions(images, masks, pred_masks, batch_idx, save_dir):
    """可视化一个批次的预测结果"""
    # 设置中文字体
    set_chinese_font()
    
    batch_size = images.size(0)
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4*batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # 原始图像
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        
        # 真实掩码
        true_mask = masks[i].cpu().squeeze().numpy()
        
        # 预测掩码
        pred_mask = pred_masks[i].cpu().squeeze().numpy()
        
        # 显示原图
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('原始图像')
        axes[i, 0].axis('off')
        
        # 显示真实掩码
        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title('真实掩码')
        axes[i, 1].axis('off')
        
        # 显示预测掩码
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title('预测掩码')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions', f'batch_{batch_idx+1}.png'))
    plt.close()

def plot_confusion_matrix(metrics, save_dir):
    """绘制混淆矩阵"""
    # 设置中文字体
    set_chinese_font()
    
    cm = np.array([
        [metrics['True Negative'], metrics['False Positive']],
        [metrics['False Negative'], metrics['True Positive']]
    ])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    classes = ['背景', '细胞核']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # 在混淆矩阵中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.0f}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='医学图像分割评估脚本')
    parser.add_argument('--data-dir', type=str, default='data', help='数据集目录')
    parser.add_argument('--model', type=str, default='deeplabv3plus', help='模型名称')
    parser.add_argument('--backbone', type=str, default='resnet50', help='骨干网络')
    parser.add_argument('--model-path', type=str, default='results/best_model.pth', help='模型路径')
    parser.add_argument('--img-size', type=int, default=512, help='图像大小')
    parser.add_argument('--save-dir', type=str, default='results', help='保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='评估设备')
    parser.add_argument('--batch-size', type=int, default=1, help='批量大小')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置评估设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    _, _, test_loader, _ = load_monuseg_dataset(
        args.data_dir, 
        img_size=args.img_size, 
        augmentation=False,  # 测试不使用数据增强
        batch_size=args.batch_size
    )
    
    if test_loader is None:
        print("数据加载失败，请检查数据路径和预处理步骤")
        return
    
    # 创建模型
    model = create_model(
        args.model, 
        num_classes=1, 
        backbone=args.backbone, 
        output_stride=16,
        batch_size=args.batch_size  # 传递批量大小参数，以便决定是否使用GroupNorm
    )
    
    # 加载已训练的模型权重
    if os.path.exists(args.model_path):
        load_model(model, args.model_path, device)
        print(f"模型加载成功: {args.model_path}")
    else:
        print(f"模型文件不存在: {args.model_path}")
        return
    
    model = model.to(device)
    
    # 评估模型
    metrics = evaluate_model(model, test_loader, device, args.save_dir)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(metrics, args.save_dir)
    
    print(f"评估完成! 结果已保存至 {args.save_dir}")

if __name__ == "__main__":
    main() 