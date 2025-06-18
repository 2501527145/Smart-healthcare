import os
import argparse
import torch
from xml_to_mask import convert_all_annotations
from data_preprocessing import load_monuseg_dataset, visualize_samples, get_dataset_info
from models import create_model, get_model_info
from utils import visualize_augmentations, count_parameters
from train import train_model, plot_learning_curves
from evaluate import evaluate_model, plot_confusion_matrix
from font_config import set_chinese_font

def parse_args():
    parser = argparse.ArgumentParser(description='医学图像分割全流程')
    parser.add_argument('--data-dir', type=str, default='data', help='数据集目录')
    parser.add_argument('--model', type=str, default='deeplabv3plus', help='模型名称')
    parser.add_argument('--backbone', type=str, default='resnet50', help='骨干网络')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--img-size', type=int, default=512, help='图像大小')
    parser.add_argument('--save-dir', type=str, default='results', help='保存目录')
    parser.add_argument('--no-aug', action='store_true', help='不使用数据增强')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--only-eval', action='store_true', help='只进行评估')
    parser.add_argument('--model-path', type=str, default=None, help='模型路径，用于评估')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置中文字体，在程序开始时全局设置
    set_chinese_font()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 确保结果目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 步骤1: 将XML标注转换为掩码图像
    annotations_dir = os.path.join(args.data_dir, "MoNuSeg 2018 Training Data", "MoNuSeg 2018 Training Data", "Annotations")
    print("将XML标注转换为掩码图像...")
    convert_all_annotations(annotations_dir)
    
    # 步骤2: 加载和预处理数据集
    print("加载和预处理数据集...")
    train_loader, val_loader, test_loader, datasets = load_monuseg_dataset(
        args.data_dir, 
        img_size=args.img_size, 
        augmentation=not args.no_aug,
        batch_size=args.batch_size
    )
    
    if train_loader is None:
        print("数据加载失败，请检查数据路径和预处理步骤")
        return
    
    train_dataset, val_dataset, test_dataset = datasets
    
    # 获取数据集信息
    print("数据集信息:")
    get_dataset_info()
    
    # 可视化数据样本
    print("可视化数据样本...")
    visualize_samples(train_dataset)
    
    # 可视化数据增强效果
    if not args.no_aug:
        print("可视化数据增强效果...")
        visualize_augmentations(train_dataset)
    
    # 创建模型
    print("创建模型...")
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
    
    # 打印参数量
    num_params = count_parameters(model)
    print(f"模型参数量: {num_params:,}")
    
    if not args.only_eval:
        # 步骤3: 训练模型
        print("开始训练模型...")
        from torch.optim import Adam
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        from utils import DiceLoss
        
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
        criterion = DiceLoss()
        
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
            save_dir=args.save_dir
        )
        
        # 绘制学习曲线
        plot_learning_curves(history, args.save_dir)
        
        # 使用最佳模型进行评估
        model_path = os.path.join(args.save_dir, "best_model.pth")
    else:
        # 仅评估模式，加载指定模型
        if args.model_path is None:
            print("评估模式下需要指定模型路径")
            return
        
        model_path = args.model_path
        from utils import load_model
        model, _ = load_model(model, model_path, device)
        print(f"加载模型: {model_path}")
    
    # 步骤4: 评估模型
    print("开始评估模型...")
    metrics = evaluate_model(model, test_loader, device, args.save_dir)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(metrics, args.save_dir)
    
    print("完成! 所有结果已保存至 {}".format(args.save_dir))

if __name__ == "__main__":
    main() 