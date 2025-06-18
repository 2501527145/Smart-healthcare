import os
import glob
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from font_config import set_chinese_font

class MoNuSegDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        """
        MoNuSeg细胞核分割数据集加载器
        Args:
            img_paths: 图像路径列表
            mask_paths: 掩码路径列表
            transform: 数据增强转换
        """
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # 加载图像和掩码
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # 使用PIL打开图像
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 灰度图
        
        # 应用转换
        if self.transform:
            # 自定义转换确保图像和掩码同步变换
            image, mask = self.transform(image, mask)
        else:
            # 默认转换为张量
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)
            
        # 二值化掩码
        mask = (mask > 0.5).float()
        
        return image, mask

class DataTransform:
    def __init__(self, output_size=512, data_augmentation=True, strong_augmentation=False):
        """
        数据转换和增强
        Args:
            output_size: 输出图像大小
            data_augmentation: 是否进行数据增强
            strong_augmentation: 是否进行更强的数据增强
        """
        self.output_size = output_size
        self.data_augmentation = data_augmentation
        self.strong_augmentation = strong_augmentation
    
    def __call__(self, image, mask):
        # 调整尺寸
        image = image.resize((self.output_size, self.output_size), Image.BICUBIC)
        mask = mask.resize((self.output_size, self.output_size), Image.NEAREST)
        
        # 数据增强
        if self.data_augmentation:
            # 随机水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # 随机垂直翻转
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # 随机旋转
            if self.strong_augmentation:
                # 更随机的角度
                angle = random.uniform(-30, 30)
            else:
                # 仅90度的旋转
                angle = random.randint(0, 3) * 90
                
            if angle != 0:
                # 修复rotate函数，移除不兼容的resample参数
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
                
            # 随机亮度和对比度变化
            if random.random() > 0.5:
                brightness_factor = 0.8 + random.random() * 0.4  # 0.8-1.2
                image = TF.adjust_brightness(image, brightness_factor)
            
            if random.random() > 0.5:
                contrast_factor = 0.8 + random.random() * 0.4  # 0.8-1.2
                image = TF.adjust_contrast(image, contrast_factor)
            
            # 强增强: 添加更多变换
            if self.strong_augmentation:
                # 随机剪裁和调整大小
                if random.random() > 0.5:
                    i, j, h, w = transforms.RandomResizedCrop.get_params(
                        image, scale=(0.8, 1.0), ratio=(0.9, 1.1))
                    image = TF.resized_crop(image, i, j, h, w, (self.output_size, self.output_size))
                    mask = TF.resized_crop(mask, i, j, h, w, (self.output_size, self.output_size))
                
                # 随机颜色抖动
                if random.random() > 0.5:
                    saturation_factor = 0.8 + random.random() * 0.4
                    image = TF.adjust_saturation(image, saturation_factor)
                
                # 随机锐化
                if random.random() > 0.7:
                    sharpness_factor = 0.5 + random.random() * 1.5
                    image = TF.adjust_sharpness(image, sharpness_factor)
                
                # 随机透视变换 - 使用兼容的方式
                if random.random() > 0.7:
                    width, height = image.size
                    startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
                    endpoints = [(random.randint(0, width // 10), random.randint(0, height // 10)),
                                (width - 1 - random.randint(0, width // 10), random.randint(0, height // 10)),
                                (width - 1 - random.randint(0, width // 10), height - 1 - random.randint(0, height // 10)),
                                (random.randint(0, width // 10), height - 1 - random.randint(0, height // 10))]
                    try:
                        # 在0.15.2版本中perspective不需要interpolation参数
                        image = TF.perspective(image, startpoints, endpoints)
                        mask = TF.perspective(mask, startpoints, endpoints)
                    except (TypeError, AttributeError):
                        # 如果函数不存在或参数错误，则跳过此步骤
                        pass
        
        # 转换为Tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # 标准化
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 在tensor上添加高斯噪声
        if self.data_augmentation and random.random() > 0.5:
            noise_level = 0.02
            if self.strong_augmentation:
                noise_level = 0.03  # 更强的噪声
            noise = torch.randn_like(image) * noise_level
            image = image + noise
            image = torch.clamp(image, 0, 1)
        
        return image, mask

def load_monuseg_dataset(data_dir, img_size=512, augmentation=True, batch_size=4, strong_augmentation=False):
    """
    加载MoNuSeg数据集并划分训练、验证和测试集
    Args:
        data_dir: 数据集目录
        img_size: 输出图像大小
        augmentation: 是否进行数据增强
        batch_size: 批量大小
        strong_augmentation: 是否使用强数据增强
    
    Returns:
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        test_loader: 测试集数据加载器
    """
    print("加载MoNuSeg数据集...")
    
    # 训练数据路径
    train_img_dir = os.path.join(data_dir, "MoNuSeg 2018 Training Data", "MoNuSeg 2018 Training Data", "Tissue Images")
    train_mask_dir = os.path.join(data_dir, "MoNuSeg 2018 Training Data", "MoNuSeg 2018 Training Data", "Annotations")
    
    # 测试数据路径
    test_img_dir = os.path.join(data_dir, "MoNuSegTestData", "MoNuSegTestData")
    
    # 获取所有训练图像
    train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.tif")))
    train_masks = []
    
    # 找到对应的掩码
    for img_path in train_imgs:
        img_name = os.path.basename(img_path).split('.')[0]
        mask_path = os.path.join(train_mask_dir, img_name + ".xml")
        
        # 如果XML文件存在，我们需要转换为掩码图像
        if os.path.exists(mask_path):
            # 这里需要XML解析和掩码生成逻辑
            # 简化处理，直接找已生成的掩码
            mask_img_path = os.path.join(train_mask_dir, img_name + "_mask.png")
            if os.path.exists(mask_img_path):
                train_masks.append(mask_img_path)
    
    # 确保找到了掩码
    assert len(train_imgs) == len(train_masks), "图像和掩码数量不匹配"
    
    if len(train_masks) == 0:
        print("警告：未找到掩码图像，请确保已将XML标注转换为掩码图像")
        return None, None, None
    
    # 划分训练集、验证集和测试集 (7:2:1)
    train_val_imgs, test_imgs, train_val_masks, test_masks = train_test_split(
        train_imgs, train_masks, test_size=0.1, random_state=42
    )
    
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        train_val_imgs, train_val_masks, test_size=0.22, random_state=42  # 0.22 of 90% = 20% of total
    )
    
    print(f"训练集: {len(train_imgs)} 样本")
    print(f"验证集: {len(val_imgs)} 样本")
    print(f"测试集: {len(test_imgs)} 样本")
    
    # 检查批量大小是否合适
    if batch_size > len(train_imgs) or batch_size > len(val_imgs) or batch_size > len(test_imgs):
        original_batch_size = batch_size
        batch_size = min(len(train_imgs), len(val_imgs), len(test_imgs))
        if batch_size <= 0:
            batch_size = 1
        print(f"警告: 批量大小({original_batch_size})大于某些数据集的大小，已调整为{batch_size}")
    
    # 确保批量大小至少为2，避免BatchNorm问题
    if batch_size == 1:
        print("警告: 批量大小为1，可能导致BatchNorm层出现问题。考虑将批量大小设置为大于1。")
    
    print(f"数据增强: {'启用' if augmentation else '禁用'}")
    if augmentation and strong_augmentation:
        print("使用强数据增强")
    
    # 创建数据增强转换
    train_transform = DataTransform(output_size=img_size, data_augmentation=augmentation, strong_augmentation=strong_augmentation)
    val_transform = DataTransform(output_size=img_size, data_augmentation=False)  # 验证和测试不增强
    
    # 创建数据集
    train_dataset = MoNuSegDataset(train_imgs, train_masks, transform=train_transform)
    val_dataset = MoNuSegDataset(val_imgs, val_masks, transform=val_transform)
    test_dataset = MoNuSegDataset(test_imgs, test_masks, transform=val_transform)
    
    # 设置数据加载器的工作线程数
    num_workers = 0  # 如果遇到内存问题，可以设置为0
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, drop_last=False)
    
    return train_loader, val_loader, test_loader, (train_dataset, val_dataset, test_dataset)

def visualize_samples(dataset, num_samples=3):
    """可视化数据集样本"""
    # 设置中文字体
    set_chinese_font()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        img, mask = dataset[i]
        
        # 转换回numpy进行可视化
        img = img.permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        
        mask = mask.squeeze().numpy()
        
        # 显示原图
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('原始图像')
        axes[i, 0].axis('off')
        
        # 显示掩码
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('掩码')
        axes[i, 1].axis('off')
        
        # 显示叠加图
        overlay = img.copy()
        overlay[mask > 0.5] = [1, 0, 0]  # 红色标记细胞核
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('叠加图')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/sample_visualization.png')
    plt.close()
    
    print("样本可视化已保存至 results/sample_visualization.png")

def get_dataset_info():
    """获取数据集信息"""
    print("MoNuSeg数据集信息:")
    print("类型: 医学显微镜图像 (H&E染色的组织切片)")
    print("任务: 细胞核分割")
    print("分辨率: 原始图像约为1000x1000像素")
    print("数据来源: Multi-Organ Nucleus Segmentation Challenge 2018")
    print("说明: 该数据集包含来自多个器官的H&E染色组织图像，用于细胞核分割任务")

if __name__ == "__main__":
    # 测试数据加载
    data_dir = "data"
    train_loader, val_loader, test_loader, datasets = load_monuseg_dataset(data_dir)
    
    if train_loader:
        train_dataset, val_dataset, test_dataset = datasets
        # 可视化几个样本
        visualize_samples(train_dataset)
        
        # 显示数据集信息
        get_dataset_info() 