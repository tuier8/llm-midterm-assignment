"""
翻译任务数据处理工具
用于IWSLT2017英译中数据集的预处理和批处理
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import jieba
import os


# 特殊token
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def english_tokenizer(text):
    """
    英文分词器：简单的空格分词
    """
    return text.lower().split()


def chinese_tokenizer(text):
    """
    中文分词器：使用jieba分词
    """
    return list(jieba.cut(text))


def build_vocab_from_tokens(tokens_list, min_freq=2, max_size=None):
    """
    从token列表构建词汇表
    
    Args:
        tokens_list: token序列的列表
        min_freq: 最小词频
        max_size: 词汇表最大大小
    
    Returns:
        vocab: token到索引的映射
        idx2token: 索引到token的映射
    """
    # 统计词频
    token_freq = {}
    for tokens in tokens_list:
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1
    
    # 过滤低频词并排序
    tokens = [token for token, freq in token_freq.items() if freq >= min_freq]
    tokens = sorted(tokens)  # 排序以保证确定性
    
    # 限制词汇表大小
    if max_size is not None:
        tokens = tokens[:max_size - 4]  # 减去特殊token的数量
    
    # 构建词汇表
    special_tokens = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    vocab = {token: idx for idx, token in enumerate(special_tokens + tokens)}
    idx2token = {idx: token for token, idx in vocab.items()}
    
    return vocab, idx2token


def load_iwslt2017_data(data_dir='datasets/iwslt2017', max_samples=None):
    """
    加载IWSLT2017英译中数据集
    
    Args:
        data_dir: 数据集目录
        max_samples: 限制样本数量（用于快速实验）
    
    Returns:
        train_data: 训练数据 [(en, zh), ...]
        val_data: 验证数据
        test_data: 测试数据
    """

    import zipfile
    
    zip_path = os.path.join(data_dir, 'data/2017-01-trnted/texts/en/zh/en-zh.zip')
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"找不到数据文件: {zip_path}")
    
    print(f"手动解析ZIP文件: {zip_path}")
    
    train_data = []
    val_data = []
    test_data = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 列出ZIP文件内容
        file_list = zip_ref.namelist()
        print(f"ZIP文件包含 {len(file_list)} 个文件")
        
        # 查找训练集文件
        train_en_file = None
        train_zh_file = None
        for fname in file_list:
            if 'train.tags' in fname and fname.endswith('.en'):
                train_en_file = fname
            if 'train.tags' in fname and fname.endswith('.zh'):
                train_zh_file = fname
        
        print(f"训练集英文文件: {train_en_file}")
        print(f"训练集中文文件: {train_zh_file}")
        
        # 读取训练集
        if train_en_file and train_zh_file:
            try:
                with zip_ref.open(train_en_file) as f_en, \
                     zip_ref.open(train_zh_file) as f_zh:
                    en_lines = f_en.read().decode('utf-8').strip().split('\n')
                    zh_lines = f_zh.read().decode('utf-8').strip().split('\n')
                    
                    print(f"英文行数: {len(en_lines)}, 中文行数: {len(zh_lines)}")
                    
                    # 检查前几行来判断文件格式
                    sample_lines = en_lines[:5]
                    print(f"训练集前5行示例:")
                    for i, line in enumerate(sample_lines):
                        print(f"  行{i}: {line[:100]}")  # 只显示前100个字符
                    
                    # 判断是XML格式还是纯文本格式
                    is_xml_format = any(line.startswith('<seg') for line in sample_lines)
                    print(f"文件格式: {'XML' if is_xml_format else '纯文本'}")
                    
                    for en_line, zh_line in zip(en_lines, zh_lines):
                        if is_xml_format:
                            # XML格式：提取<seg>标签中的内容
                            if en_line.startswith('<') and not en_line.startswith('<seg'):
                                continue
                            
                            if en_line.startswith('<seg'):
                                try:
                                    en_text = en_line.split('>')[1].split('<')[0].strip()
                                    zh_text = zh_line.split('>')[1].split('<')[0].strip()
                                    
                                    if en_text and zh_text:
                                        train_data.append((en_text, zh_text))
                                        if max_samples and len(train_data) >= max_samples:
                                            break
                                except IndexError:
                                    continue
                        else:
                            # 纯文本格式：直接使用（跳过XML标签行）
                            if en_line.startswith('<') or en_line.startswith('#') or not en_line.strip():
                                continue
                            
                            en_text = en_line.strip()
                            zh_text = zh_line.strip()
                            
                            if en_text and zh_text:
                                train_data.append((en_text, zh_text))
                                if max_samples and len(train_data) >= max_samples:
                                    break
                    
                    print(f"成功提取 {len(train_data)} 条训练数据")
            except Exception as e:
                print(f"读取训练集出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("未找到训练集文件！")
        
        # 读取验证集
        val_en_file = None
        val_zh_file = None
        for fname in file_list:
            if 'dev2010' in fname and fname.endswith('.en.xml'):
                val_en_file = fname
            if 'dev2010' in fname and fname.endswith('.zh.xml'):
                val_zh_file = fname
        
        print(f"验证集英文文件: {val_en_file}")
        print(f"验证集中文文件: {val_zh_file}")
        
        if val_en_file and val_zh_file:
            try:
                with zip_ref.open(val_en_file) as f_en, \
                     zip_ref.open(val_zh_file) as f_zh:
                    en_lines = f_en.read().decode('utf-8').strip().split('\n')
                    zh_lines = f_zh.read().decode('utf-8').strip().split('\n')
                    
                    for en_line, zh_line in zip(en_lines, zh_lines):
                        if en_line.startswith('<seg'):
                            try:
                                en_text = en_line.split('>')[1].split('<')[0].strip()
                                zh_text = zh_line.split('>')[1].split('<')[0].strip()
                                if en_text and zh_text:
                                    val_data.append((en_text, zh_text))
                            except IndexError:
                                continue
            except Exception as e:
                print(f"读取验证集出错: {e}")
    
    print(f"加载完成: 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条")
    
    # 如果训练集为空，使用验证集的一部分作为训练集
    if len(train_data) == 0 and len(val_data) > 0:
        print("警告: 训练集为空，将使用验证集的前80%作为训练集")
        split_idx = int(len(val_data) * 0.8)
        train_data = val_data[:split_idx]
        val_data = val_data[split_idx:]
        print(f"重新分配: 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条")
    
    return train_data, val_data, test_data


class TranslationDataset(Dataset):
    """
    翻译数据集
    """
    def __init__(self, data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, max_len=128):
        """
        Args:
            data: [(src_text, tgt_text), ...]
            src_vocab: 源语言词汇表
            tgt_vocab: 目标语言词汇表
            src_tokenizer: 源语言分词器
            tgt_tokenizer: 目标语言分词器
            max_len: 最大序列长度
        """
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        
        # 分词
        src_tokens = self.src_tokenizer(src_text)[:self.max_len - 2]  # 留出<sos>和<eos>的位置
        tgt_tokens = self.tgt_tokenizer(tgt_text)[:self.max_len - 2]
        
        # 转换为索引，添加<sos>和<eos>
        src_indices = [SOS_IDX] + [
            self.src_vocab.get(token, UNK_IDX) for token in src_tokens
        ] + [EOS_IDX]
        
        tgt_indices = [SOS_IDX] + [
            self.tgt_vocab.get(token, UNK_IDX) for token in tgt_tokens
        ] + [EOS_IDX]
        
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)


def collate_fn(batch):
    """
    批处理函数，动态padding
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Padding
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    
    return src_batch, tgt_batch


def build_translation_vocabs(train_data, src_tokenizer, tgt_tokenizer, 
                              src_min_freq=2, tgt_min_freq=2,
                              src_max_size=10000, tgt_max_size=10000):
    """
    为源语言和目标语言构建词汇表
    
    Args:
        train_data: 训练数据 [(src_text, tgt_text), ...]
        src_tokenizer: 源语言分词器
        tgt_tokenizer: 目标语言分词器
        src_min_freq: 源语言最小词频
        tgt_min_freq: 目标语言最小词频
        src_max_size: 源语言词汇表最大大小
        tgt_max_size: 目标语言词汇表最大大小
    
    Returns:
        src_vocab, src_idx2token, tgt_vocab, tgt_idx2token
    """
    print("正在构建词汇表...")
    
    # 分词
    src_tokens_list = [src_tokenizer(src_text) for src_text, _ in train_data]
    tgt_tokens_list = [tgt_tokenizer(tgt_text) for _, tgt_text in train_data]
    
    # 构建词汇表
    src_vocab, src_idx2token = build_vocab_from_tokens(
        src_tokens_list, min_freq=src_min_freq, max_size=src_max_size
    )
    tgt_vocab, tgt_idx2token = build_vocab_from_tokens(
        tgt_tokens_list, min_freq=tgt_min_freq, max_size=tgt_max_size
    )
    
    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    
    return src_vocab, src_idx2token, tgt_vocab, tgt_idx2token


def create_translation_dataloaders(train_data, val_data, 
                                    src_vocab, tgt_vocab,
                                    src_tokenizer, tgt_tokenizer,
                                    batch_size=32, max_len=128,
                                    num_workers=0):
    """
    创建训练和验证数据加载器
    
    Args:
        train_data: 训练数据
        val_data: 验证数据
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        src_tokenizer: 源语言分词器
        tgt_tokenizer: 目标语言分词器
        batch_size: 批大小
        max_len: 最大序列长度
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader
    """
    # 创建数据集
    train_dataset = TranslationDataset(
        train_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, max_len
    )
    val_dataset = TranslationDataset(
        val_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, max_len
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return train_loader, val_loader

