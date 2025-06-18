# 医学图像分割项目 - MoNuSeg细胞核分割

本项目是基于深度学习的医学图像分割实现，使用DeepLabv3+模型对MoNuSeg细胞核分割数据集进行训练和测试。

## 项目结构

```
.
├── data/                  # 数据集目录
│   ├── MoNuSeg 2018 Training Data/     # 训练数据
│   └── MoNuSegTestData/                # 测试数据
├── results/               # 结果保存目录
├── src/                   # 源代码
│   ├── data_preprocessing.py  # 数据预处理模块
│   ├── xml_to_mask.py         # XML标注转换为掩码
│   ├── models.py              # 模型定义
│   ├── utils.py               # 工具函数
│   ├── train.py               # 训练脚本
│   ├── evaluate.py            # 评估脚本
│   └── main.py                # 主运行脚本
└── README.md              # 说明文档
```

## 环境要求

本项目需要以下Python库：

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- matplotlib
- scikit-learn
- pillow
- tqdm
- opencv-python

可以使用以下命令安装依赖：

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow tqdm opencv-python
```

## 数据集

项目使用MoNuSeg (Multi-Organ Nucleus Segmentation) 细胞核分割数据集，该数据集包含H&E染色的组织图像和对应的细胞核标注。

- 数据类型：医学显微镜图像 (H&E染色的组织切片)
- 分辨率：原始图像约为1000x1000像素
- 任务：细胞核分割

## 运行步骤

### 1. 准备数据

将MoNuSeg数据集放在`data`目录下，确保目录结构如下：

```
data/
├── MoNuSeg 2018 Training Data/
│   └── MoNuSeg 2018 Training Data/
│       ├── Tissue Images/
│       └── Annotations/
└── MoNuSegTestData/
    └── MoNuSegTestData/
```

### 2. 运行完整流程

```bash
python src/main.py --data-dir data --epochs 30 --device cuda
```

参数说明：
- `--data-dir`: 数据集目录
- `--model`: 模型名称，默认为"deeplabv3plus"
- `--backbone`: 骨干网络，默认为"resnet50"
- `--epochs`: 训练轮数，默认为30
- `--batch-size`: 批大小，默认为4
- `--lr`: 学习率，默认为0.001
- `--img-size`: 图像大小，默认为512
- `--save-dir`: 结果保存目录，默认为"results"
- `--no-aug`: 不使用数据增强
- `--device`: 训练设备，默认为"cuda"
- `--only-eval`: 只进行评估不训练
- `--model-path`: 模型路径，用于评估

### 3. 单独运行各个步骤

#### 标注转换

将XML格式的标注转换为掩码图像：

```bash
python src/xml_to_mask.py
```

#### 训练模型

```bash
python src/train.py --data-dir data --epochs 50
```

#### 评估模型

```bash
python src/evaluate.py --data-dir data --model-path results/best_model.pth
```

## 模型

本项目使用DeepLabv3+模型进行细胞核分割。DeepLabv3+的主要特点：

- 采用了空洞卷积以获得更大的感受野，同时保持分辨率
- 结合了空洞空间金字塔池化(ASPP)模块，可以捕获多尺度上下文
- 采用编码器-解码器结构，结合低层特征进行精细分割
- 使用跳跃连接融合不同层级的特征，提高边界细节

## 结果

训练完成后，结果将保存在`results`目录下，包括：

- 训练好的模型
- 损失函数和IoU曲线
- 学习率变化曲线
- 测试集评估结果
- 预测结果可视化
- 混淆矩阵

## 引用

如果使用了本项目代码，请引用：

```
@article{MoNuSeg2018,
  title={A dataset and a technique for generalized nuclear segmentation for computational pathology},
  author={Kumar, Neeraj and Verma, Ruchika and Sharma, Sanuj and Bhargava, Surabhi and Vahadane, Abhishek and Sethi, Amit},
  journal={IEEE transactions on medical imaging},
  year={2017}
}

@article{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  journal={ECCV},
  year={2018}
}
``` 