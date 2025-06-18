@echo off
echo 开始评估优化后的医学图像分割模型...

:: 设置PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%cd%

:: 运行评估脚本
python src/evaluate_optimized.py ^
    --data-dir data ^
    --model-path results/optimized/best_iou_model.pth ^
    --batch-size 4 ^
    --img-size 512 ^
    --save-dir results/evaluation ^
    --original-metrics

echo 评估完成！结果已保存到 results/evaluation 目录
pause 