import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os

def set_chinese_font():
    """
    设置matplotlib中文字体
    """
    # 获取操作系统类型
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统常用中文字体
        font_list = ['SimHei', 'Microsoft YaHei', 'STXihei', 'SimSun']
    elif system == 'Darwin':  # macOS
        # macOS系统常用中文字体
        font_list = ['STHeiti', 'Heiti TC', 'Songti SC', 'PingFang SC']
    else:  # Linux或其他系统
        # Linux系统常用中文字体
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UMing CN']
    
    # 遍历字体列表，找到第一个可用的中文字体
    font_found = False
    for font_name in font_list:
        font_path = fm.findfont(fm.FontProperties(family=font_name))
        if os.path.exists(font_path) and font_name.lower() not in font_path.lower():
            plt.rcParams['font.family'] = font_name
            print(f"使用中文字体: {font_name}")
            font_found = True
            break
    
    # 如果没有找到中文字体，使用matplotlib默认配置
    if not font_found:
        # 设置matplotlib使用Unicode字符集，可以显示中文但可能不美观
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("未找到中文字体，使用默认配置")
    
    return font_found 