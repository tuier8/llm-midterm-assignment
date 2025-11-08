# 运行脚本使用说明

本目录包含两个自动化脚本，用于快速搭建环境并运行英译中翻译任务训练。

## 脚本文件

- `run.bat` - Windows系统运行脚本
- `run.sh` - Linux/Mac系统运行脚本

## 使用方法

### Windows系统

1. 打开命令提示符（CMD）或PowerShell
2. 切换到项目根目录
3. 运行脚本：

```cmd
script\run.bat
```

### Linux/Mac系统

1. 打开终端
2. 切换到项目根目录
3. 添加执行权限并运行：

```bash
chmod +x script/run.sh
./script/run.sh
```

## 脚本功能

脚本将自动完成以下步骤：

### 1. 检查conda安装
验证conda是否已安装并添加到系统PATH

### 2. 创建conda环境
- 环境名称：`llm-translation`
- Python版本：3.10
- 如果环境已存在，会提示是否重新创建

### 3. 激活环境
自动激活新创建的conda环境

### 4. 安装PyTorch
安装CPU版本的PyTorch（适合无GPU环境）

如果需要GPU版本，可以手动修改脚本中的安装命令：
```bash
# CPU版本（默认）
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# GPU版本（CUDA 11.8）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### 5. 安装项目依赖
从`requirements.txt`安装以下依赖包：
- numpy>=1.24.0
- matplotlib>=3.7.0
- tqdm>=4.65.0
- jieba>=0.42.1（中文分词）
- sacrebleu>=2.3.1（BLEU评估）
- datasets>=2.14.0（数据集加载，可选）

### 6. 验证安装
检查所有关键依赖包是否正确安装

### 7. 检查数据集
验证IWSLT2017数据集文件是否存在

### 8. 运行训练
自动执行`src/train_translation.py`开始训练

## 预期输出

训练完成后，结果将保存在`results/translation/`目录：

```
results/translation/
├── best_model.pt           # 最佳模型检查点
├── results_summary.json    # 训练结果摘要
└── training_curves.png     # 训练曲线图
```

## 训练配置

默认配置（快速验证模式）：

```
训练数据：10,000条
批大小：32
序列长度：64
训练轮数：15 epochs
学习率：0.0001

模型维度：256
注意力头数：4
前馈维度：512
层数：3
```

## 预计训练时间

- **CPU训练**：约2-4小时（取决于CPU性能）
- **GPU训练**：约20-40分钟（取决于GPU性能）

## 常见问题

### Q1: conda命令未找到
**A:** 请先安装Anaconda或Miniconda：
- Anaconda: https://www.anaconda.com/products/distribution
- Miniconda: https://docs.conda.io/en/latest/miniconda.html

安装后，确保conda已添加到系统PATH。

### Q2: 环境已存在如何处理？
**A:** 脚本会提示是否删除并重新创建。选择：
- `y` - 删除旧环境并创建新环境
- `n` - 使用现有环境继续

### Q3: 数据集文件不存在
**A:** 确保数据集文件位于正确路径：
```
datasets/iwslt2017/data/2017-01-trnted/texts/en/zh/en-zh.zip
```

### Q4: 训练速度很慢
**A:** CPU训练速度较慢是正常的。建议：
1. 使用GPU版本的PyTorch（如果有GPU）
2. 减小批大小或训练数据量
3. 减少训练轮数

### Q5: 内存不足
**A:** 可以修改`src/train_translation.py`中的参数：
```python
BATCH_SIZE = 16  # 减小批大小
MAX_LEN = 32     # 减小序列长度
MAX_SAMPLES = 5000  # 减少训练数据
```

## 手动运行

如果不想使用自动化脚本，可以手动执行：

```bash
# 1. 创建并激活环境
conda create -n llm-translation python=3.10 -y
conda activate llm-translation

# 2. 安装PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行训练
python src/train_translation.py
```

## 后续使用

环境创建成功后，后续只需：

```bash
# 激活环境
conda activate llm-translation

# 运行训练
python src/train_translation.py
```

## 查看结果

训练完成后查看结果：

```bash
# Windows
type results\translation\results_summary.json

# Linux/Mac
cat results/translation/results_summary.json
```

查看训练曲线图：
```bash
# 使用图片查看器打开
results/translation/training_curves.png
```

## 技术支持

如果遇到问题，请检查：
1. conda是否正确安装
2. Python版本是否为3.8+
3. 数据集文件是否存在
4. 网络连接是否正常（下载依赖包）
5. 磁盘空间是否充足（至少2GB）

