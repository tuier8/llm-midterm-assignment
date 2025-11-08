#!/bin/bash

# ============================================================
# IWSLT2017 英译中翻译任务 - 完整运行脚本
# 从创建conda环境到运行Transformer模型训练
# ============================================================

set -e  # 遇到错误立即退出

# 配置参数
ENV_NAME="llm-translation"
PYTHON_VERSION="3.10"

echo "============================================================"
echo "步骤 1: 检查conda是否安装"
echo "============================================================"
if ! command -v conda &> /dev/null
then
    echo "错误: conda未安装或未添加到PATH"
    echo "请先安装Anaconda或Miniconda: https://www.anaconda.com/products/distribution"
    exit 1
fi
echo "✓ Conda已安装: $(conda --version)"

echo ""
echo "============================================================"
echo "步骤 2: 创建conda环境 (${ENV_NAME})"
echo "============================================================"
# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 '${ENV_NAME}' 已存在"
    read -p "是否删除并重新创建? (y/n): " choice
    if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
        echo "正在删除环境 '${ENV_NAME}'..."
        conda env remove -n ${ENV_NAME} -y
        echo "正在创建新环境..."
        conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    else
        echo "使用现有环境 '${ENV_NAME}'"
    fi
else
    echo "正在创建新环境 '${ENV_NAME}'..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

echo ""
echo "============================================================"
echo "步骤 3: 激活conda环境"
echo "============================================================"
echo "激活环境: ${ENV_NAME}"
# 初始化conda（如果需要）
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}
echo "✓ 当前环境: $(conda info --envs | grep '*')"
echo "✓ Python版本: $(python --version)"

echo ""
echo "============================================================"
echo "步骤 4: 安装PyTorch (CPU版本)"
echo "============================================================"
echo "正在安装PyTorch..."
# 安装CPU版本的PyTorch（适合无GPU环境）
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
echo "✓ PyTorch已安装: $(python -c 'import torch; print(torch.__version__)')"

echo ""
echo "============================================================"
echo "步骤 5: 安装项目依赖"
echo "============================================================"
if [ -f "requirements.txt" ]; then
    echo "从requirements.txt安装依赖..."
    pip install -r requirements.txt
else
    echo "requirements.txt未找到，手动安装依赖..."
    pip install numpy>=1.24.0 matplotlib>=3.7.0 tqdm>=4.65.0
    pip install jieba>=0.42.1 sacrebleu>=2.3.1
fi
echo "✓ 依赖包已安装"

echo ""
echo "============================================================"
echo "步骤 6: 验证安装"
echo "============================================================"
echo "检查关键依赖包..."
python -c "
import sys
import torch
import numpy as np
import matplotlib
import tqdm
import jieba
import sacrebleu

print('✓ PyTorch:', torch.__version__)
print('✓ NumPy:', np.__version__)
print('✓ Matplotlib:', matplotlib.__version__)
print('✓ Jieba: 已安装')
print('✓ SacreBLEU: 已安装')
print('✓ Python:', sys.version.split()[0])
"

echo ""
echo "============================================================"
echo "步骤 7: 检查数据集"
echo "============================================================"
DATA_PATH="datasets/iwslt2017/data/2017-01-trnted/texts/en/zh/en-zh.zip"
if [ -f "$DATA_PATH" ]; then
    echo "✓ 数据集文件存在: $DATA_PATH"
    # 显示文件大小
    if command -v du &> /dev/null; then
        SIZE=$(du -h "$DATA_PATH" | cut -f1)
        echo "  文件大小: $SIZE"
    fi
else
    echo "⚠ 警告: 数据集文件不存在: $DATA_PATH"
    echo "  请确保已下载IWSLT2017数据集"
fi

echo ""
echo "============================================================"
echo "步骤 8: 运行训练脚本"
echo "============================================================"
echo "开始训练Transformer模型..."
echo ""

# 运行训练脚本
if [ -f "src/train_translation.py" ]; then
    python src/train_translation.py
    
    echo ""
    echo "============================================================"
    echo "训练完成！"
    echo "============================================================"
    echo "查看结果:"
    echo "  - 模型检查点: results/translation/best_model.pt"
    echo "  - 训练结果: results/translation/results_summary.json"
    echo "  - 可视化图表: results/translation/training_curves.png"
    echo ""
    
    # 如果结果文件存在，显示摘要
    if [ -f "results/translation/results_summary.json" ]; then
        echo "训练结果摘要:"
        cat results/translation/results_summary.json
    fi
else
    echo "错误: 训练脚本不存在: src/train_translation.py"
    exit 1
fi

echo ""
echo "============================================================"
echo "全部完成！"
echo "============================================================"
echo "环境信息:"
echo "  - Conda环境: ${ENV_NAME}"
echo "  - Python: $(python --version)"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "激活环境命令: conda activate ${ENV_NAME}"
echo "再次运行训练: python src/train_translation.py"
echo "============================================================"

