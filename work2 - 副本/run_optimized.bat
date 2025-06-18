@echo off
echo 开始运行优化后的医学图像分割训练...

:: 设置PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%cd%

:: 运行优化后的训练脚本
python src/optimized_train.py ^
    --data-dir data ^
    --model deeplabv3plus ^
    --backbone resnet50 ^
    --epochs 100 ^
    --batch-size 4 ^
    --lr 0.0005 ^
    --weight-decay 1e-4 ^
    --img-size 512 ^
    --save-dir results/optimized ^
    --strong-aug ^
    --loss combo ^
    --scheduler cosine ^
    --grad-accum 2 ^
    --mixed-precision ^
    --seed 42

echo 训练完成！
pause 