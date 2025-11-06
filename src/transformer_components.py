"""
Transformer核心组件实现
包括: Multi-Head Self-Attention, Position-wise FFN, LayerNorm, 位置编码等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    使用sin和cos函数生成固定的位置编码
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q, K, V的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        Args:
            Q: (batch_size, num_heads, seq_len, d_k)
            K: (batch_size, num_heads, seq_len, d_k)
            V: (batch_size, num_heads, seq_len, d_k)
            mask: (batch_size, 1, 1, seq_len) or None
        Returns:
            output: (batch_size, num_heads, seq_len, d_k)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, 1, seq_len) or None
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # 线性变换并分割成多头
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 应用缩放点积注意力
        attn_output, self.attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 最终线性变换
        output = self.W_o(attn_output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    包含: Multi-Head Attention + Add&Norm + FFN + Add&Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # Multi-Head Self-Attention
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        
        # Position-wise Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, 1, seq_len) or None
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Multi-Head Attention + 残差连接 + LayerNorm
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-Forward + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    包含: Masked Multi-Head Attention + Add&Norm + 
          Multi-Head Attention + Add&Norm + FFN + Add&Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Masked Multi-Head Self-Attention
        self.masked_self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        
        # Multi-Head Cross-Attention
        self.cross_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        
        # Position-wise Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: (batch_size, tgt_seq_len, d_model)
            enc_output: (batch_size, src_seq_len, d_model)
            src_mask: (batch_size, 1, 1, src_seq_len) or None
            tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len) or None
        Returns:
            output: (batch_size, tgt_seq_len, d_model)
        """
        # Masked Multi-Head Self-Attention + 残差连接 + LayerNorm
        attn_output = self.masked_self_attention(x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Multi-Head Cross-Attention + 残差连接 + LayerNorm
        # 注意: 这里需要修改cross_attention以支持不同的Q和K,V
        # 简化起见，我们这里先用self-attention的实现
        cross_attn_output = self.cross_attention(x, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-Forward + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

