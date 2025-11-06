"""
数据处理工具
用于文本数据的预处理和批处理
"""

import torch
from torch.utils.data import Dataset, DataLoader
import re


class SimpleTextDataset(Dataset):
    """
    简单的文本数据集
    用于语言建模任务
    """
    def __init__(self, text, vocab, seq_len=128):
        self.seq_len = seq_len
        self.vocab = vocab
        
        # 分词并转换为索引
        tokens = self.tokenize(text)
        self.data = [vocab.get(token, vocab['<unk>']) for token in tokens]
        
    def tokenize(self, text):
        """简单的分词器"""
        # 转小写并按空格分割
        text = text.lower()
        tokens = re.findall(r'\w+|[.,!?;]', text)
        return tokens
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        # 输入序列
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        # 目标序列（下一个词）
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def build_vocab(text, min_freq=2):
    """
    构建词汇表
    Args:
        text: 输入文本
        min_freq: 最小词频
    Returns:
        vocab: 词到索引的映射
        idx2word: 索引到词的映射
    """
    # 分词
    text = text.lower()
    tokens = re.findall(r'\w+|[.,!?;]', text)
    
    # 统计词频
    word_freq = {}
    for token in tokens:
        word_freq[token] = word_freq.get(token, 0) + 1
    
    # 过滤低频词
    words = [word for word, freq in word_freq.items() if freq >= min_freq]
    words = sorted(words)  # 排序以保证确定性
    
    # 添加特殊token
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    vocab = {token: idx for idx, token in enumerate(special_tokens + words)}
    idx2word = {idx: token for token, idx in vocab.items()}
    
    return vocab, idx2word


def get_sample_text():
    """
    获取用于实验的示例文本
    使用多篇简短故事作为训练数据
    """
    text = """
    Once upon a time, there was a brave knight who lived in a castle. 
    The knight loved to explore the forest near his home. One day, he found a magical sword.
    The sword had special powers that could protect the kingdom from evil.
    
    In a small village, there lived a wise old wizard. The wizard knew many spells and potions.
    People from far and wide came to seek his advice. He helped everyone with kindness.
    
    A young prince dreamed of becoming a great hero. He trained every day with his sword and shield.
    His father, the king, was very proud of him. The prince wanted to make the kingdom safe.
    
    There was a beautiful princess who loved reading books. She had a large library in her tower.
    Every night, she read stories about adventures and magic. She wished to go on an adventure too.
    
    One sunny morning, the knight met the wizard in the forest. They talked about the dark force 
    that threatened the land. Together, they made a plan to protect the kingdom.
    
    The prince joined them on their quest. The princess also wanted to help. She brought her knowledge
    from all the books she had read. The four heroes set out on their journey.
    
    They traveled through mountains and crossed rivers. They faced many challenges along the way.
    But they never gave up. Their friendship made them stronger.
    
    Finally, they reached the dark castle where the evil lived. Using their combined powers,
    they defeated the darkness. The kingdom was saved and peace returned to the land.
    
    The knight, wizard, prince, and princess became legends. Their story was told for generations.
    Children learned about courage, wisdom, and friendship from their tale.
    
    The king held a great celebration in their honor. There was music, dancing, and feasting.
    Everyone in the kingdom was happy and grateful. The heroes had brought hope to all.
    
    Years passed, and the kingdom flourished. New knights trained in the castle. Young wizards
    studied magic and spells. Princes and princesses learned to be brave and kind.
    
    The forest remained a place of wonder and mystery. Sometimes, people still found magical items
    hidden among the trees. Each discovery reminded them of the great adventure.
    
    The old wizard continued to help people with his wisdom. The knight protected the borders of
    the kingdom. The prince became a fair and just king. The princess opened schools to teach reading.
    
    Their legacy lived on through their actions and teachings. The kingdom became known throughout
    the world as a place of peace and learning. Many visitors came to see its beauty.
    
    At night, stars shone brightly over the castle. The moon illuminated the forest paths.
    Everything was peaceful and calm. The land had been transformed by the heroes' sacrifice.
    
    Stories were written about their adventures. Bards sang songs about their bravery. Painters
    created beautiful artwork depicting their quest. Their memory would never be forgotten.
    
    New generations grew up hearing these tales. They were inspired to be brave and kind.
    The values of the heroes became part of the kingdom's culture. Peace and prosperity continued.
    
    Sometimes, when the wind blew through the trees, people said they could hear the whispers of
    the ancient magic. The forest seemed to hold secrets from long ago. It was a reminder that
    magic still existed in the world, waiting for those brave enough to seek it.
    """
    return text


def create_dataloaders(vocab, batch_size=32, seq_len=64):
    """
    创建训练和验证数据加载器
    """
    text = get_sample_text()
    
    # 划分训练集和验证集（80-20分割）
    split_idx = int(len(text) * 0.8)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # 创建数据集
    train_dataset = SimpleTextDataset(train_text, vocab, seq_len)
    val_dataset = SimpleTextDataset(val_text, vocab, seq_len)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

