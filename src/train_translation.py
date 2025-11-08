"""
英译中翻译任务训练脚本
使用Transformer Seq2Seq模型在IWSLT2017数据集上训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tqdm import tqdm
import sacrebleu

from transformer_model import TransformerSeq2Seq
from translation_data_utils import (
    load_iwslt2017_data,
    build_translation_vocabs,
    create_translation_dataloaders,
    english_tokenizer,
    chinese_tokenizer,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX
)


def create_mask(src, tgt, pad_idx=PAD_IDX):
    """
    创建源序列和目标序列的mask
    
    Args:
        src: (batch_size, src_seq_len)
        tgt: (batch_size, tgt_seq_len)
        pad_idx: padding token的索引
    
    Returns:
        src_mask: (batch_size, 1, 1, src_seq_len)
        tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
    """
    # 源序列padding mask
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_len)
    
    # 目标序列padding mask
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_seq_len)
    
    # 目标序列causal mask（下三角矩阵）
    tgt_seq_len = tgt.size(1)
    tgt_causal_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device)).bool()
    tgt_causal_mask = tgt_causal_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_seq_len, tgt_seq_len)
    
    # 组合padding mask和causal mask
    tgt_mask = tgt_padding_mask & tgt_causal_mask
    
    return src_mask, tgt_mask


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}')
    
    for batch_idx, (src, tgt) in enumerate(progress_bar):
        src, tgt = src.to(device), tgt.to(device)
        
        # Teacher forcing: 输入是tgt[:-1], 目标是tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # 创建mask
        src_mask, tgt_mask = create_mask(src, tgt_input)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)  # (batch_size, tgt_seq_len-1, tgt_vocab_size)
        
        # 计算损失（忽略padding）
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    """
    评估模型
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask, tgt_mask = create_mask(src, tgt_input)
            
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def greedy_decode(model, src, src_mask, max_len, tgt_vocab_size, device, sos_idx=SOS_IDX, eos_idx=EOS_IDX):
    """
    贪婪解码生成翻译
    
    Args:
        model: Transformer模型
        src: (batch_size, src_seq_len)
        src_mask: (batch_size, 1, 1, src_seq_len)
        max_len: 最大生成长度
        tgt_vocab_size: 目标词汇表大小
        device: 设备
        sos_idx: <sos> token索引
        eos_idx: <eos> token索引
    
    Returns:
        output: (batch_size, max_len)
    """
    model.eval()
    batch_size = src.size(0)
    
    # 编码源序列
    enc_output = model.encode(src, src_mask)
    
    # 初始化解码器输入为<sos>
    tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    
    # 逐个生成token
    for _ in range(max_len - 1):
        # 创建目标mask
        _, tgt_mask = create_mask(src, tgt)
        
        # 解码
        output = model.decode(tgt, enc_output, src_mask, tgt_mask)
        
        # 获取最后一个位置的预测
        next_token_logits = output[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
        
        # 添加到输出序列
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # 如果所有序列都生成了<eos>，则停止
        if (next_token == eos_idx).all():
            break
    
    return tgt


def calculate_bleu(model, val_loader, src_vocab, tgt_vocab, tgt_idx2token, device, max_samples=None):
    """
    计算BLEU分数
    
    Args:
        model: Transformer模型
        val_loader: 验证数据加载器
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        tgt_idx2token: 目标语言索引到token的映射
        device: 设备
        max_samples: 最大评估样本数（为了加速）
    
    Returns:
        bleu_score: BLEU分数
    """
    model.eval()
    
    hypotheses = []  # 模型生成的翻译
    references = []  # 参考翻译
    
    sample_count = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(val_loader, desc='计算BLEU'):
            src = src.to(device)
            
            # 创建源mask
            src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
            
            # 贪婪解码
            output = greedy_decode(model, src, src_mask, max_len=100, 
                                   tgt_vocab_size=len(tgt_vocab), device=device)
            
            # 转换为文本
            for i in range(output.size(0)):
                # 生成的翻译
                hyp_indices = output[i].cpu().tolist()
                hyp_tokens = []
                for idx in hyp_indices:
                    if idx == EOS_IDX:
                        break
                    if idx not in [PAD_IDX, SOS_IDX]:
                        hyp_tokens.append(tgt_idx2token.get(idx, '<unk>'))
                hyp_text = ''.join(hyp_tokens)  # 中文不需要空格
                hypotheses.append(hyp_text)
                
                # 参考翻译
                ref_indices = tgt[i].cpu().tolist()
                ref_tokens = []
                for idx in ref_indices:
                    if idx == EOS_IDX:
                        break
                    if idx not in [PAD_IDX, SOS_IDX]:
                        ref_tokens.append(tgt_idx2token.get(idx, '<unk>'))
                ref_text = ''.join(ref_tokens)
                references.append([ref_text])  # sacrebleu需要列表形式
                
                sample_count += 1
                if max_samples and sample_count >= max_samples:
                    break
            
            if max_samples and sample_count >= max_samples:
                break
    
    # 计算BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    
    # 打印一些示例
    print("\n翻译示例:")
    for i in range(min(3, len(hypotheses))):
        print(f"生成: {hypotheses[i]}")
        print(f"参考: {references[i][0]}")
        print()
    
    return bleu.score


def train_model(model, train_loader, val_loader, src_vocab, tgt_vocab, 
                tgt_idx2token, num_epochs, device, learning_rate=0.0001,
                save_dir='results/translation'):
    """
    完整的训练流程
    """
    os.makedirs(save_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    train_losses = []
    val_losses = []
    bleu_scores = []
    
    best_bleu = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pt')
    
    print(f"\n开始训练...")
    print(f"参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # 计算BLEU（每5个epoch计算一次以节省时间）
        if epoch % 5 == 0 or epoch == num_epochs:
            bleu_score = calculate_bleu(model, val_loader, src_vocab, tgt_vocab, 
                                        tgt_idx2token, device, max_samples=500)
            bleu_scores.append(bleu_score)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {bleu_score:.2f}")
            
            # 保存最佳模型
            if bleu_score > best_bleu:
                best_bleu = bleu_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'bleu_score': bleu_score,
                }, best_model_path)
                print(f"保存最佳模型 (BLEU: {bleu_score:.2f})")
        else:
            bleu_scores.append(bleu_scores[-1] if bleu_scores else 0.0)
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'bleu_scores': bleu_scores,
        'best_bleu': best_bleu
    }


def plot_results(results, save_dir='results/translation'):
    """
    绘制训练结果图表
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    epochs = range(1, len(results['train_losses']) + 1)
    
    # 损失曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['train_losses'], label='训练损失', linewidth=2, color='#2E86AB')
    plt.plot(epochs, results['val_losses'], label='验证损失', linewidth=2, color='#A23B72')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('训练和验证损失', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # BLEU分数曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['bleu_scores'], label='BLEU分数', linewidth=2, color='#F18F01')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('BLEU Score', fontsize=12)
    plt.title('BLEU分数变化', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"保存训练曲线图: {save_dir}/training_curves.png")
    plt.close()


def save_results(results, save_dir='results'):
    """
    保存训练结果
    """
    os.makedirs(save_dir, exist_ok=True)
    
    summary = {
        'final_train_loss': float(results['train_losses'][-1]),
        'final_val_loss': float(results['val_losses'][-1]),
        'final_bleu_score': float(results['bleu_scores'][-1]),
        'best_bleu_score': float(results['best_bleu']),
        'best_val_loss': float(min(results['val_losses'])),
    }
    
    with open(f'{save_dir}/results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n保存实验结果摘要: {save_dir}/results_summary.json")
    
    # 打印摘要
    print("\n" + "="*60)
    print("训练结果摘要")
    print("="*60)
    print(f"最终训练损失: {summary['final_train_loss']:.4f}")
    print(f"最终验证损失: {summary['final_val_loss']:.4f}")
    print(f"最佳验证损失: {summary['best_val_loss']:.4f}")
    print(f"最终BLEU分数: {summary['final_bleu_score']:.2f}")
    print(f"最佳BLEU分数: {summary['best_bleu_score']:.2f}")
    print("="*60)


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
    
    # 超参数设置
    MAX_SAMPLES = 10000  # 限制训练样本数（快速验证）
    BATCH_SIZE = 32
    MAX_LEN = 64
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.0001
    
    # 模型参数（小规模以适应CPU训练）
    D_MODEL = 256
    NUM_HEADS = 4
    D_FF = 512
    NUM_LAYERS = 3
    DROPOUT = 0.1
    
    # 加载数据
    print("\n" + "="*60)
    print("步骤 1: 加载数据")
    print("="*60)
    train_data, val_data, test_data = load_iwslt2017_data(max_samples=MAX_SAMPLES)
    
    # 构建词汇表
    print("\n" + "="*60)
    print("步骤 2: 构建词汇表")
    print("="*60)
    src_vocab, src_idx2token, tgt_vocab, tgt_idx2token = build_translation_vocabs(
        train_data, 
        english_tokenizer, 
        chinese_tokenizer,
        src_min_freq=2,
        tgt_min_freq=2,
        src_max_size=10000,
        tgt_max_size=10000
    )
    
    # 创建数据加载器
    print("\n" + "="*60)
    print("步骤 3: 创建数据加载器")
    print("="*60)
    train_loader, val_loader = create_translation_dataloaders(
        train_data, val_data,
        src_vocab, tgt_vocab,
        english_tokenizer, chinese_tokenizer,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        num_workers=0
    )
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # 初始化模型
    print("\n" + "="*60)
    print("步骤 4: 初始化模型")
    print("="*60)
    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        max_len=MAX_LEN * 2
    ).to(device)
    
    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("\n" + "="*60)
    print("步骤 5: 训练模型")
    print("="*60)
    results = train_model(
        model, train_loader, val_loader,
        src_vocab, tgt_vocab, tgt_idx2token,
        num_epochs=NUM_EPOCHS,
        device=device,
        learning_rate=LEARNING_RATE
    )
    
    # 保存和可视化结果
    print("\n" + "="*60)
    print("步骤 6: 保存结果")
    print("="*60)
    save_results(results)
    plot_results(results)
    
    print("\n训练完成!")


if __name__ == '__main__':
    main()

