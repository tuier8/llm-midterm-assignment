"""
训练脚本
用于训练Transformer语言模型并进行消融实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tqdm import tqdm

from transformer_model import (
    TransformerLanguageModel,
    TransformerLM_NoAttention,
    TransformerLM_NoPositionalEncoding,
    TransformerLM_NoResidual
)
from data_utils import build_vocab, create_dataloaders, get_sample_text


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(x)  # (batch_size, seq_len, vocab_size)
        
        # 计算损失
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    """
    评估模型
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def calculate_perplexity(loss):
    """
    计算困惑度
    """
    return np.exp(loss)


def train_model(model, train_loader, val_loader, num_epochs, device, 
                learning_rate=0.0001, model_name="transformer"):
    """
    完整的训练流程
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []
    
    print(f"\n训练模型: {model_name}")
    print(f"参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(num_epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_perplexity = calculate_perplexity(train_loss)
        train_perplexities.append(train_perplexity)
        
        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_perplexity = calculate_perplexity(val_loss)
        val_perplexities.append(val_perplexity)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train PPL: {train_perplexity:.2f} - "
              f"Val Loss: {val_loss:.4f}, Val PPL: {val_perplexity:.2f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_perplexities': train_perplexities,
        'val_perplexities': val_perplexities
    }


def run_ablation_study(vocab_size, train_loader, val_loader, device, num_epochs=50):
    """
    运行消融实验
    """
    results = {}
    
    # 超参数设置
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 3
    dropout = 0.1
    
    # 1. 完整模型
    print("\n" + "="*50)
    print("实验 1: 完整 Transformer 模型")
    print("="*50)
    model_full = TransformerLanguageModel(
        vocab_size, d_model, num_heads, d_ff, num_layers, dropout
    ).to(device)
    results['full_model'] = train_model(
        model_full, train_loader, val_loader, num_epochs, device, model_name="Full Transformer"
    )
    
    # 2. 无注意力机制
    print("\n" + "="*50)
    print("实验 2: 移除多头注意力机制")
    print("="*50)
    model_no_attn = TransformerLM_NoAttention(
        vocab_size, d_model, d_ff, num_layers, dropout
    ).to(device)
    results['no_attention'] = train_model(
        model_no_attn, train_loader, val_loader, num_epochs, device, model_name="No Attention"
    )
    
    # 3. 无位置编码
    print("\n" + "="*50)
    print("实验 3: 移除位置编码")
    print("="*50)
    model_no_pe = TransformerLM_NoPositionalEncoding(
        vocab_size, d_model, num_heads, d_ff, num_layers, dropout
    ).to(device)
    results['no_positional_encoding'] = train_model(
        model_no_pe, train_loader, val_loader, num_epochs, device, model_name="No Positional Encoding"
    )
    
    # 4. 无残差连接
    print("\n" + "="*50)
    print("实验 4: 移除残差连接")
    print("="*50)
    model_no_res = TransformerLM_NoResidual(
        vocab_size, d_model, num_heads, d_ff, num_layers, dropout
    ).to(device)
    results['no_residual'] = train_model(
        model_no_res, train_loader, val_loader, num_epochs, device, model_name="No Residual"
    )
    
    return results


def plot_results(results, save_dir='results'):
    """
    绘制实验结果图表
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 训练损失对比
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        label_map = {
            'full_model': '完整模型',
            'no_attention': '无注意力',
            'no_positional_encoding': '无位置编码',
            'no_residual': '无残差连接'
        }
        plt.plot(result['train_losses'], label=label_map.get(name, name), linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('训练损失对比', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        label_map = {
            'full_model': '完整模型',
            'no_attention': '无注意力',
            'no_positional_encoding': '无位置编码',
            'no_residual': '无残差连接'
        }
        plt.plot(result['val_losses'], label=label_map.get(name, name), linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('验证损失对比', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"保存损失对比图: {save_dir}/loss_comparison.png")
    
    # 2. 困惑度对比
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        label_map = {
            'full_model': '完整模型',
            'no_attention': '无注意力',
            'no_positional_encoding': '无位置编码',
            'no_residual': '无残差连接'
        }
        plt.plot(result['train_perplexities'], label=label_map.get(name, name), linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Perplexity', fontsize=12)
    plt.title('训练困惑度对比', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        label_map = {
            'full_model': '完整模型',
            'no_attention': '无注意力',
            'no_positional_encoding': '无位置编码',
            'no_residual': '无残差连接'
        }
        plt.plot(result['val_perplexities'], label=label_map.get(name, name), linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Perplexity', fontsize=12)
    plt.title('验证困惑度对比', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/perplexity_comparison.png', dpi=300, bbox_inches='tight')
    print(f"保存困惑度对比图: {save_dir}/perplexity_comparison.png")
    
    # 3. 最终性能对比柱状图
    plt.figure(figsize=(10, 6))
    
    models = []
    final_val_ppl = []
    label_map = {
        'full_model': '完整模型',
        'no_attention': '无注意力',
        'no_positional_encoding': '无位置编码',
        'no_residual': '无残差连接'
    }
    
    for name, result in results.items():
        models.append(label_map.get(name, name))
        final_val_ppl.append(result['val_perplexities'][-1])
    
    x = np.arange(len(models))
    bars = plt.bar(x, final_val_ppl, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    plt.xlabel('模型配置', fontsize=12)
    plt.ylabel('最终验证困惑度', fontsize=12)
    plt.title('消融实验：最终验证困惑度对比', fontsize=14, fontweight='bold')
    plt.xticks(x, models, rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上显示数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/final_performance.png', dpi=300, bbox_inches='tight')
    print(f"保存最终性能对比图: {save_dir}/final_performance.png")
    
    plt.close('all')


def save_results(results, vocab, save_dir='results'):
    """
    保存实验结果
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存数值结果
    summary = {}
    for name, result in results.items():
        summary[name] = {
            'final_train_loss': float(result['train_losses'][-1]),
            'final_val_loss': float(result['val_losses'][-1]),
            'final_train_perplexity': float(result['train_perplexities'][-1]),
            'final_val_perplexity': float(result['val_perplexities'][-1]),
            'best_val_loss': float(min(result['val_losses'])),
            'best_val_perplexity': float(min(result['val_perplexities']))
        }
    
    with open(f'{save_dir}/results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n保存实验结果摘要: {save_dir}/results_summary.json")
    
    # 打印摘要表格
    print("\n" + "="*80)
    print("消融实验结果摘要")
    print("="*80)
    print(f"{'模型配置':<25} {'最终训练损失':<15} {'最终验证损失':<15} {'最终验证困惑度':<15}")
    print("-"*80)
    
    label_map = {
        'full_model': '完整模型',
        'no_attention': '无注意力机制',
        'no_positional_encoding': '无位置编码',
        'no_residual': '无残差连接'
    }
    
    for name, metrics in summary.items():
        model_name = label_map.get(name, name)
        print(f"{model_name:<25} {metrics['final_train_loss']:<15.4f} "
              f"{metrics['final_val_loss']:<15.4f} {metrics['final_val_perplexity']:<15.2f}")
    
    print("="*80)


def main():
    """
    主函数
    """
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 构建词汇表
    text = get_sample_text()
    vocab, idx2word = build_vocab(text, min_freq=1)
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")
    
    # 创建数据加载器
    batch_size = 16
    seq_len = 32
    train_loader, val_loader = create_dataloaders(vocab, batch_size, seq_len)
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # 运行消融实验
    num_epochs = 50
    results = run_ablation_study(vocab_size, train_loader, val_loader, device, num_epochs)
    
    # 保存和可视化结果
    save_results(results, vocab)
    plot_results(results)
    
    print("\n实验完成!")


if __name__ == '__main__':
    main()

