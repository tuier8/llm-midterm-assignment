"""
完整的Transformer模型实现
用于语言建模任务
"""

import torch
import torch.nn as nn
from transformer_components import (
    PositionalEncoding, 
    EncoderLayer,
    MultiHeadSelfAttention,
    PositionwiseFeedForward
)


class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    包含多个编码器层的堆叠
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=5000):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # 编码器层堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len) - 输入token索引
            mask: (batch_size, 1, 1, seq_len) or None
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # 词嵌入 + 缩放
        x = self.embedding(x) * (self.d_model ** 0.5)
        
        # 添加位置编码
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 通过编码器层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        return x


class TransformerLanguageModel(nn.Module):
    """
    基于Transformer的语言模型
    用于下一个词预测任务
    """
    def __init__(self, vocab_size, d_model=256, num_heads=8, d_ff=1024, 
                 num_layers=4, dropout=0.1, max_len=512):
        super(TransformerLanguageModel, self).__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len
        )
        
        # 输出层: 将隐藏状态映射到词汇表
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len)
            mask: (batch_size, 1, 1, seq_len) or None
        Returns:
            output: (batch_size, seq_len, vocab_size)
        """
        # 编码
        enc_output = self.encoder(x, mask)
        
        # 预测下一个词
        output = self.fc_out(enc_output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """
        生成因果mask（causal mask），用于自回归生成
        确保位置i只能看到位置<=i的信息
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).type(torch.bool)
        return mask


class TransformerLM_NoAttention(nn.Module):
    """
    消融实验: 移除注意力机制的模型
    用于验证注意力机制的重要性
    """
    def __init__(self, vocab_size, d_model=256, d_ff=1024, num_layers=4, dropout=0.1, max_len=512):
        super(TransformerLM_NoAttention, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # 只使用FFN层
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                PositionwiseFeedForward(d_model, d_ff, dropout),
                nn.LayerNorm(d_model)
            )
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = self.embedding(x) * (self.d_model ** 0.5)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for ffn_layer in self.ffn_layers:
            x = ffn_layer(x) + x  # 残差连接
        
        output = self.fc_out(x)
        return output


class TransformerLM_NoPositionalEncoding(nn.Module):
    """
    消融实验: 移除位置编码的模型
    用于验证位置编码的重要性
    """
    def __init__(self, vocab_size, d_model=256, num_heads=8, d_ff=1024, 
                 num_layers=4, dropout=0.1, max_len=512):
        super(TransformerLM_NoPositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 不使用位置编码
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = self.embedding(x) * (self.d_model ** 0.5)
        # 跳过位置编码
        x = self.dropout(x)
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        output = self.fc_out(x)
        return output


class TransformerLM_NoResidual(nn.Module):
    """
    消融实验: 移除残差连接的模型
    用于验证残差连接的重要性
    """
    def __init__(self, vocab_size, d_model=256, num_heads=8, d_ff=1024, 
                 num_layers=4, dropout=0.1, max_len=512):
        super(TransformerLM_NoResidual, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # 自定义编码器层（无残差连接）
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.ffn_layers = nn.ModuleList([
            PositionwiseFeedForward(d_model, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm_layers1 = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        
        self.norm_layers2 = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = self.embedding(x) * (self.d_model ** 0.5)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for i in range(len(self.attention_layers)):
            # 不使用残差连接
            x = self.norm_layers1[i](self.attention_layers[i](x, mask))
            x = self.norm_layers2[i](self.ffn_layers[i](x))
        
        output = self.fc_out(x)
        return output

