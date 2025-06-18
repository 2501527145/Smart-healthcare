import os
import glob
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor

def parse_xml_to_mask(xml_path, img_path):
    """
    解析XML标注文件并生成掩码图像
    
    Args:
        xml_path: XML标注文件路径
        img_path: 原始图像路径，用于获取尺寸
        
    Returns:
        mask: 生成的掩码图像数组
    """
    # 获取图像尺寸
    img = Image.open(img_path)
    width, height = img.size
    
    # 创建空白掩码
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 解析XML文件
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 找到所有细胞核标注
        regions = root.findall('.//Region')
        
        for region in regions:
            # 获取每个区域的所有点
            vertices = region.findall('.//Vertex')
            points = []
            
            for vertex in vertices:
                x = float(vertex.get('X'))
                y = float(vertex.get('Y'))
                points.append((x, y))
            
            # 绘制多边形
            if len(points) > 2:  # 至少需要三个点才能构成多边形
                draw.polygon(points, fill=255)
        
        return np.array(mask)
    
    except Exception as e:
        print(f"处理文件 {xml_path} 时出错: {e}")
        return np.zeros((height, width), dtype=np.uint8)

def process_annotation(xml_path, output_dir):
    """处理单个XML标注文件并保存掩码"""
    try:
        # 获取图像名称
        img_name = os.path.basename(xml_path).split('.')[0]
        # 获取对应的图像路径
        img_dir = os.path.dirname(os.path.dirname(xml_path)) + "/Tissue Images"
        img_path = os.path.join(img_dir, img_name + ".tif")
        
        if not os.path.exists(img_path):
            print(f"未找到对应图像: {img_path}")
            return False
        
        # 生成掩码
        mask = parse_xml_to_mask(xml_path, img_path)
        
        # 保存掩码
        mask_path = os.path.join(output_dir, img_name + "_mask.png")
        cv2.imwrite(mask_path, mask)
        
        return True
    
    except Exception as e:
        print(f"处理 {xml_path} 时出错: {e}")
        return False

def convert_all_annotations(annotations_dir, output_dir=None):
    """
    转换所有XML标注为掩码图像
    
    Args:
        annotations_dir: 包含XML标注的目录
        output_dir: 输出掩码的目录，默认与标注相同
    
    Returns:
        成功转换的掩码数量
    """
    if output_dir is None:
        output_dir = annotations_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有XML文件
    xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))
    print(f"找到 {len(xml_files)} 个XML标注文件")
    
    # 并行处理所有标注
    success_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda x: process_annotation(x, output_dir), xml_files))
        success_count = sum(results)
    
    print(f"成功转换 {success_count}/{len(xml_files)} 个掩码")
    return success_count

if __name__ == "__main__":
    # 设置目录
    data_dir = "data"
    annotations_dir = os.path.join(data_dir, "MoNuSeg 2018 Training Data", "MoNuSeg 2018 Training Data", "Annotations")
    
    # 转换所有标注
    convert_all_annotations(annotations_dir) 