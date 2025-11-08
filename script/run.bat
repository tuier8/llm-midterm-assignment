@echo off
REM ============================================================
REM IWSLT2017 英译中翻译任务 - 完整运行脚本 (Windows版本)
REM 从创建conda环境到运行Transformer模型训练
REM ============================================================

setlocal enabledelayedexpansion

REM 配置参数
set ENV_NAME=llm-translation
set PYTHON_VERSION=3.10

echo ============================================================
echo 步骤 1: 检查conda是否安装
echo ============================================================
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: conda未安装或未添加到PATH
    echo 请先安装Anaconda或Miniconda: https://www.anaconda.com/products/distribution
    pause
    exit /b 1
)
conda --version
echo ✓ Conda已安装

echo.
echo ============================================================
echo 步骤 2: 创建conda环境 ^(%ENV_NAME%^)
echo ============================================================
REM 检查环境是否已存在
conda env list | findstr /b "%ENV_NAME% " >nul 2>&1
if %errorlevel% equ 0 (
    echo 环境 '%ENV_NAME%' 已存在
    set /p choice="是否删除并重新创建? (y/n): "
    if /i "!choice!"=="y" (
        echo 正在删除环境 '%ENV_NAME%'...
        call conda env remove -n %ENV_NAME% -y
        echo 正在创建新环境...
        call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
    ) else (
        echo 使用现有环境 '%ENV_NAME%'
    )
) else (
    echo 正在创建新环境 '%ENV_NAME%'...
    call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
)

echo.
echo ============================================================
echo 步骤 3: 激活conda环境
echo ============================================================
echo 激活环境: %ENV_NAME%
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo 错误: 无法激活环境
    pause
    exit /b 1
)
python --version
echo ✓ 环境已激活

echo.
echo ============================================================
echo 步骤 4: 安装PyTorch ^(CPU版本^)
echo ============================================================
echo 正在安装PyTorch...
REM 安装CPU版本的PyTorch（适合无GPU环境）
call conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
python -c "import torch; print('✓ PyTorch已安装:', torch.__version__)"

echo.
echo ============================================================
echo 步骤 5: 安装项目依赖
echo ============================================================
if exist "requirements.txt" (
    echo 从requirements.txt安装依赖...
    pip install -r requirements.txt
) else (
    echo requirements.txt未找到，手动安装依赖...
    pip install numpy>=1.24.0 matplotlib>=3.7.0 tqdm>=4.65.0
    pip install jieba>=0.42.1 sacrebleu>=2.3.1
)
echo ✓ 依赖包已安装

echo.
echo ============================================================
echo 步骤 6: 验证安装
echo ============================================================
echo 检查关键依赖包...
python -c "import sys; import torch; import numpy as np; import matplotlib; import tqdm; import jieba; import sacrebleu; print('✓ PyTorch:', torch.__version__); print('✓ NumPy:', np.__version__); print('✓ Matplotlib:', matplotlib.__version__); print('✓ Jieba: 已安装'); print('✓ SacreBLEU: 已安装'); print('✓ Python:', sys.version.split()[0])"

echo.
echo ============================================================
echo 步骤 7: 检查数据集
echo ============================================================
set DATA_PATH=datasets\iwslt2017\data\2017-01-trnted\texts\en\zh\en-zh.zip
if exist "%DATA_PATH%" (
    echo ✓ 数据集文件存在: %DATA_PATH%
    for %%F in ("%DATA_PATH%") do echo   文件大小: %%~zF 字节
) else (
    echo ⚠ 警告: 数据集文件不存在: %DATA_PATH%
    echo   请确保已下载IWSLT2017数据集
)

echo.
echo ============================================================
echo 步骤 8: 运行训练脚本
echo ============================================================
echo 开始训练Transformer模型...
echo.

REM 运行训练脚本
if exist "src\train_translation.py" (
    python src\train_translation.py
    
    echo.
    echo ============================================================
    echo 训练完成！
    echo ============================================================
    echo 查看结果:
    echo   - 模型检查点: results\translation\best_model.pt
    echo   - 训练结果: results\translation\results_summary.json
    echo   - 可视化图表: results\translation\training_curves.png
    echo.
    
    REM 如果结果文件存在，显示摘要
    if exist "results\translation\results_summary.json" (
        echo 训练结果摘要:
        type results\translation\results_summary.json
    )
) else (
    echo 错误: 训练脚本不存在: src\train_translation.py
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 全部完成！
echo ============================================================
echo 环境信息:
echo   - Conda环境: %ENV_NAME%
python --version
python -c "import torch; print('  - PyTorch:', torch.__version__)"
echo.
echo 激活环境命令: conda activate %ENV_NAME%
echo 再次运行训练: python src\train_translation.py
echo ============================================================
pause

