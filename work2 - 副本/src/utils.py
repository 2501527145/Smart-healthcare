import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from font_config import set_chinese_font

class DiceLoss(nn.Module):
    """
    Dice损失函数
    用于二分类图像分割，解决类别不平衡问题
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predict, target):
        predict = torch.sigmoid(predict)  # 对预测值进行sigmoid
        
        # 扁平化
        predict = predict.view(-1)
        target = target.view(-1)
        
        intersection = (predict * target).sum()
        
        dice = (2.0 * intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth)
        
        return 1.0 - dice

class FocalLoss(nn.Module):
    """
    Focal Loss
    解决分割中前景背景严重不平衡问题
    """
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        
        # 扁平化
        predict = predict.view(-1)
        target = target.view(-1)
        
        # BCE
        bce = F.binary_cross_entropy(predict, target, reduction='none')
        
        # Focal 权重
        pt = target * predict + (1 - target) * (1 - predict)
        focal_weight = (1 - pt).pow(self.gamma)
        
        # 添加alpha权重
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # 最终损失
        loss = alpha_weight * focal_weight * bce
        
        return loss.mean()

def IoU(predict, target, smooth=1e-5):
    """计算IoU指标"""
    # 扁平化
    predict = predict.view(-1)
    target = target.view(-1)
    
    # 计算交集和并集
    intersection = (predict * target).sum()
    union = predict.sum() + target.sum() - intersection
    
    # 计算IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def DiceCoefficient(predict, target, smooth=1e-5):
    """计算Dice系数"""
    # 扁平化
    predict = predict.view(-1)
    target = target.view(-1)
    
    # 计算交集
    intersection = (predict * target).sum()
    
    # 计算Dice系数
    dice = (2.0 * intersection + smooth) / (predict.sum() + target.sum() + smooth)
    
    return dice

def save_model(model, optimizer, epoch, save_dir, filename):
    """
    保存模型
    
    Args:
        model: 待保存的模型
        optimizer: 优化器
        epoch: 当前轮次
        save_dir: 保存目录
        filename: 保存文件名
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备保存数据
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # 保存模型
    torch.save(state, os.path.join(save_dir, filename))

def load_model(model, model_path, device=None):
    """
    加载模型
    
    Args:
        model: 目标模型
        model_path: 模型文件路径
        device: 设备
    
    Returns:
        model: 加载权重后的模型
        epoch: 保存的轮次
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取保存的模型状态字典
    saved_state_dict = checkpoint['model_state_dict']
    
    # 获取当前模型的状态字典
    current_state_dict = model.state_dict()
    
    # 过滤掉不匹配的键（主要是BatchNorm的状态参数）
    filtered_state_dict = {}
    for key, value in saved_state_dict.items():
        if key in current_state_dict:
            # 检查形状是否匹配
            if current_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                print(f"警告: 键 {key} 的形状不匹配，跳过加载")
        else:
            print(f"警告: 键 {key} 在当前模型中不存在，跳过加载")
    
    # 使用strict=False加载模型，允许部分键不匹配
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    if missing_keys:
        print(f"警告: 以下键在保存的模型中缺失: {missing_keys}")
    if unexpected_keys:
        print(f"警告: 以下键在当前模型中不存在: {unexpected_keys}")
    
    print(f"成功加载 {len(filtered_state_dict)} 个参数")
    
    return model, checkpoint.get('epoch', 0)

def visualize_augmentations(dataset, idx=0, num_samples=5):
    """
    可视化数据增强效果
    
    Args:
        dataset: 数据集
        idx: 样本索引
        num_samples: 展示数量
    """
    # 设置中文字体
    set_chinese_font()
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2*num_samples))
    
    # 获取原始样本
    orig_img, orig_mask = dataset[idx]
    
    # 转换回numpy进行可视化
    orig_img = orig_img.permute(1, 2, 0).numpy()
    orig_img = (orig_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    orig_img = np.clip(orig_img, 0, 1)
    
    orig_mask = orig_mask.squeeze().numpy()
    
    # 显示原始样本
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(orig_mask, cmap='gray')
    axes[0, 1].set_title('原始掩码')
    axes[0, 1].axis('off')
    
    # 展示增强后的样本
    for i in range(1, num_samples):
        # 重新获取同一样本，会应用不同的随机增强
        aug_img, aug_mask = dataset[idx]
        
        # 转换回numpy进行可视化
        aug_img = aug_img.permute(1, 2, 0).numpy()
        aug_img = (aug_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        aug_img = np.clip(aug_img, 0, 1)
        
        aug_mask = aug_mask.squeeze().numpy()
        
        # 显示增强后的样本
        axes[i, 0].imshow(aug_img)
        axes[i, 0].set_title(f'增强图像 {i}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(aug_mask, cmap='gray')
        axes[i, 1].set_title(f'增强掩码 {i}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/augmentation_visualization.png')
    plt.close()
    
    print("数据增强可视化已保存至 results/augmentation_visualization.png")

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_learning_rate(optimizer):
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

class CombinedLoss(nn.Module):
    """
    组合损失函数：Dice损失 + BCE损失 + Focal损失
    
    Args:
        dice_weight: Dice损失的权重
        bce_weight: BCE损失的权重
        focal_weight: Focal损失的权重
        smooth: Dice损失平滑系数
        alpha: Focal Loss的alpha参数
        gamma: Focal Loss的gamma参数
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.3, focal_weight=0.2, smooth=1.0, alpha=0.25, gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        
    def forward(self, predict, target):
        dice_loss = self.dice_loss(predict, target)
        bce_loss = self.bce_loss(predict, target.view(-1, 1, target.size(2), target.size(3)))
        focal_loss = self.focal_loss(predict, target)
        
        # 组合损失
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss + self.focal_weight * focal_loss
        
        return total_loss

class AdaptiveCombinedLoss(nn.Module):
    """
    自适应组合损失函数：动态调整损失权重
    
    Args:
        initial_dice_weight: Dice损失的初始权重
        initial_bce_weight: BCE损失的初始权重
        smooth: Dice损失平滑系数
    """
    def __init__(self, initial_dice_weight=0.6, initial_bce_weight=0.4, smooth=1.0):
        super(AdaptiveCombinedLoss, self).__init__()
        self.dice_weight = initial_dice_weight
        self.bce_weight = initial_bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.epoch_counter = 0
        
    def update_weights(self, epoch):
        """根据训练进度动态调整权重"""
        self.epoch_counter = epoch
        # 随着训练的进行，增加Dice损失权重
        if epoch > 30:
            self.dice_weight = min(0.8, 0.6 + epoch * 0.005)
            self.bce_weight = 1.0 - self.dice_weight
        
    def forward(self, predict, target):
        dice_loss = self.dice_loss(predict, target)
        bce_loss = self.bce_loss(predict, target.view(-1, 1, target.size(2), target.size(3)))
        
        # 组合损失
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return total_loss

class BoundaryLoss(nn.Module):
    """
    边界加权损失函数：对分割边界进行加权，改善边界分割质量
    
    Args:
        boundary_weight: 边界权重
    """
    def __init__(self, boundary_weight=5.0):
        super(BoundaryLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, predict, target):
        # 计算目标边界 (简单的边缘检测)
        target_cpu = target.detach().cpu().numpy()
        boundaries = np.zeros_like(target_cpu)
        
        for i in range(target.size(0)):
            # 通过腐蚀和膨胀提取边界
            temp = target_cpu[i, 0]
            from scipy import ndimage
            eroded = ndimage.binary_erosion(temp).astype(temp.dtype)
            boundaries[i, 0] = temp - eroded
        
        boundaries = torch.from_numpy(boundaries).to(target.device)
        
        # 普通BCE损失
        pixel_loss = self.bce_loss(predict, target.view(-1, 1, target.size(2), target.size(3)))
        
        # 增加边界的权重
        weighted_loss = pixel_loss * (1 + self.boundary_weight * boundaries)
        
        return weighted_loss.mean() 