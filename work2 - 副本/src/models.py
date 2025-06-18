import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def get_norm_layer(planes, norm_type='batch', num_groups=32):
    """
    根据指定类型获取归一化层
    
    Args:
        planes: 通道数
        norm_type: 归一化类型，'batch'或'group'
        num_groups: GroupNorm的组数，仅当norm_type='group'时使用
    
    Returns:
        归一化层
    """
    if norm_type == 'batch':
        return nn.BatchNorm2d(planes)
    elif norm_type == 'group':
        # 确保组数不超过通道数且能被通道数整除
        if planes < num_groups:
            num_groups = planes
        else:
            # 找到能被通道数整除的最大组数
            while planes % num_groups != 0:
                num_groups -= 1
        return nn.GroupNorm(num_groups, planes)
    else:
        raise ValueError(f"不支持的归一化类型: {norm_type}")

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, norm_type='batch'):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            get_norm_layer(out_channels, norm_type),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        # 确保组数不超过通道数
        num_groups = min(32, out_channels)
        self.aspp_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # 使用GroupNorm代替BatchNorm，避免1x1特征图的批归一化问题
            # GroupNorm不依赖批次大小，所以可以处理单个样本
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.aspp_pooling(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256, norm_type='batch'):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            get_norm_layer(out_channels, norm_type),
            nn.ReLU()
        ))

        # 不同空洞率的卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, norm_type))

        # 全局平均池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 项目融合
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            get_norm_layer(out_channels, norm_type),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet50', output_stride=16, norm_type='batch'):
        super(DeepLabV3Plus, self).__init__()
        
        self.norm_type = norm_type
        
        # 选择骨干网络
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
        else:
            raise ValueError("不支持的骨干网络: {}".format(backbone))
        
        # 替换stride，实现output_stride
        if output_stride == 16:
            self.backbone.layer4[0].conv2.stride = (1, 1)
            self.backbone.layer4[0].downsample[0].stride = (1, 1)
        elif output_stride == 8:
            self.backbone.layer3[0].conv2.stride = (1, 1)
            self.backbone.layer3[0].downsample[0].stride = (1, 1)
            self.backbone.layer4[0].conv2.stride = (1, 1)
            self.backbone.layer4[0].downsample[0].stride = (1, 1)
        
        # 低层特征提取（用于跳跃连接）
        self.low_level_features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1
        )
        
        # 高层特征提取
        self.high_level_features = nn.Sequential(
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        )
        
        # ASPP模块
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise ValueError("output_stride只能是8或16")
        
        self.aspp = ASPP(2048, dilations, norm_type=norm_type)
        
        # 低层特征转换
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            get_norm_layer(48, norm_type),
            nn.ReLU()
        )
        
        # 解码器（特征融合）
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            get_norm_layer(256, norm_type),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            get_norm_layer(256, norm_type),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # 提取特征
        low_level_feat = self.low_level_features(x)
        high_level_feat = self.high_level_features(low_level_feat)
        
        # ASPP处理
        x = self.aspp(high_level_feat)
        
        # 上采样回原始分辨率的1/4
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        
        # 处理低层特征
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # 特征融合
        x = torch.cat((x, low_level_feat), dim=1)
        
        # 解码器处理并上采样回原始尺寸
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x

def create_model(model_name, num_classes=1, **kwargs):
    """
    创建模型工厂函数
    
    Args:
        model_name: 模型名称
        num_classes: 类别数量
        **kwargs: 其他参数
    
    Returns:
        model: 创建的模型
    """
    if model_name == 'deeplabv3plus':
        # 创建一个新的kwargs字典，移除batch_size参数
        model_kwargs = kwargs.copy()
        
        # 如果批量大小为1，则使用GroupNorm
        if 'batch_size' in model_kwargs and model_kwargs['batch_size'] == 1:
            model_kwargs['norm_type'] = 'group'
        
        # 从传递给模型的参数中移除batch_size
        if 'batch_size' in model_kwargs:
            del model_kwargs['batch_size']
            
        return DeepLabV3Plus(num_classes=num_classes, **model_kwargs)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

def get_model_info(model_name):
    """获取模型信息"""
    if model_name == 'deeplabv3plus':
        return {
            "name": "DeepLabv3+",
            "description": "DeepLabv3+是一种用于语义分割的高级深度学习模型",
            "advantages": [
                "采用了空洞卷积以获得更大的感受野，同时保持分辨率",
                "结合了空洞空间金字塔池化(ASPP)模块，可以捕获多尺度上下文",
                "采用编码器-解码器结构，结合低层特征进行精细分割",
                "使用跳跃连接融合不同层级的特征，提高边界细节"
            ]
        }
    else:
        return {"name": model_name, "description": "未知模型"} 