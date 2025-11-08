# LLM中期作业 - IWSLT2017英译中翻译任务

本项目实现了基于Transformer的Seq2Seq模型，在IWSLT2017数据集上进行英译中翻译任务训练。

## 项目结构

```
llm-midterm-assignment/
├── src/                          # 源代码目录
│   ├── transformer_components.py # Transformer核心组件
│   ├── transformer_model.py      # Seq2Seq模型实现
│   ├── translation_data_utils.py # 数据处理工具
│   └── train_translation.py      # 训练脚本
├── datasets/                     # 数据集目录
│   └── iwslt2017/               # IWSLT2017数据集
├── results/                      # 训练结果目录
│   └── translation/             # 翻译任务结果
├── script/                       # 运行脚本
│   ├── run.sh                   # Linux/Mac运行脚本
│   └── run.bat                  # Windows运行脚本
└── requirements.txt              # Python依赖包

```

## 快速开始

### 方法一：使用自动化脚本（推荐）

#### Windows系统

```bash
# 在项目根目录下运行
script\run.bat
```

#### Linux/Mac系统

```bash
# 在项目根目录下运行
chmod +x script/run.sh
./script/run.sh
```

脚本将自动完成以下步骤：
1. 检查conda安装
2. 创建conda环境（llm-translation）
3. 安装PyTorch和依赖包
4. 验证安装
5. 运行训练脚本

### 方法二：手动安装

#### 1. 创建conda环境

```bash
conda create -n llm-translation python=3.10 -y
conda activate llm-translation
```

#### 2. 安装PyTorch（CPU版本）

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

如果有GPU，可以安装CUDA版本：
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

#### 3. 安装依赖包

```bash
pip install -r requirements.txt
```

#### 4. 运行训练

```bash
python src/train_translation.py
```

## 模型配置

### 训练配置（快速验证模式）

- **训练数据**：前10,000条（限制样本数以适应CPU训练）
- **批大小**：32
- **序列长度**：64
- **训练轮数**：15 epochs
- **学习率**：0.0001

### 模型架构

- **模型类型**：Transformer Seq2Seq（编码器-解码器）
- **模型维度**：256
- **注意力头数**：4
- **前馈网络维度**：512
- **编码器/解码器层数**：3
- **Dropout**：0.1

### 数据处理

- **英文分词**：空格分词（小写化）
- **中文分词**：jieba分词
- **特殊Token**：`<pad>`, `<unk>`, `<sos>`, `<eos>`
- **词汇表大小**：最大10,000（源语言和目标语言各自独立）

## 训练输出

训练完成后，结果将保存在 `results/translation/` 目录下：

1. **best_model.pt** - 最佳模型检查点（基于BLEU分数）
2. **results_summary.json** - 训练结果摘要
3. **training_curves.png** - 训练曲线可视化图表

### 结果摘要示例

```json
{
  "final_train_loss": 2.3456,
  "final_val_loss": 2.5678,
  "final_bleu_score": 15.23,
  "best_bleu_score": 16.45,
  "best_val_loss": 2.4567
}
```

## 评估指标

- **损失函数**：交叉熵损失（忽略padding）
- **评估指标**：BLEU分数（使用SacreBLEU）
- **评估频率**：每5个epoch计算一次BLEU分数

## 项目特性

### 核心功能

✅ 完整的Transformer Seq2Seq架构
✅ 编码器-解码器结构
✅ 多头自注意力机制
✅ 多头交叉注意力机制
✅ 位置编码
✅ 残差连接和Layer Normalization
✅ Teacher Forcing训练策略
✅ 贪婪解码推理
✅ BLEU评估

### 数据处理

✅ IWSLT2017数据集自动加载
✅ Jieba中文分词
✅ 动态padding
✅ 词汇表构建（支持最小词频和最大词汇表大小）
✅ 自动格式检测（XML/纯文本）

### 训练优化

✅ 梯度裁剪
✅ 学习率调度（Adam优化器）
✅ 自动保存最佳模型
✅ 训练进度可视化

## 环境要求

- **Python**: 3.8+
- **PyTorch**: 2.0.0+
- **CUDA**: 可选（支持CPU训练）
- **内存**: 建议8GB+
- **磁盘空间**: 约2GB（包括数据集和模型）

## 依赖包

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
jieba>=0.42.1
sacrebleu>=2.3.1
datasets>=2.14.0
```

## 常见问题

### 1. 训练速度慢

CPU训练速度较慢是正常的。如果有GPU，建议安装CUDA版本的PyTorch。

### 2. 内存不足

可以减小批大小（BATCH_SIZE）或序列长度（MAX_LEN）。

### 3. 数据集加载失败

确保数据集文件存在于正确路径：
```
datasets/iwslt2017/data/2017-01-trnted/texts/en/zh/en-zh.zip
```

### 4. BLEU分数较低

这是正常的，因为：
- 使用了小规模模型（适应CPU训练）
- 训练数据有限（10,000条）
- 训练轮数较少（15 epochs）

完整训练需要更大的模型、更多数据和更长时间。

## 扩展训练

如果想进行完整训练，可以修改 `src/train_translation.py` 中的参数：

```python
# 完整训练配置
MAX_SAMPLES = None  # 使用全部数据（约200k条）
BATCH_SIZE = 64
MAX_LEN = 128
NUM_EPOCHS = 50
D_MODEL = 512
NUM_HEADS = 8
D_FF = 2048
NUM_LAYERS = 6
```

## 许可证

本项目仅用于学习和研究目的。

## 致谢

- IWSLT2017数据集
- PyTorch团队
- Jieba分词工具
- SacreBLEU评估工具

