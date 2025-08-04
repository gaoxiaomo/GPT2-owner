import math
import os
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from DataLoader import DataLoaderLite
import time
import inspect
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from hellaswag import render_example, iterate_examples


def get_lr(it):
    # warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    elif it > max_steps:
        return min_Lr
    # consine
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # 1--0
    return min_Lr + (max_lr - min_Lr) * coeff

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key query value for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.MYGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.triu(torch.ones(config.block_size,config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self,x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, num_head, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, num_head, T, head_size)
        # attention计算
        # 原始版本
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        # 加速版本
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


#前馈网络FFN的实现：
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.MYGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x



# transformer的主结构
# 和之前的那个论文（Attention is all your need)好像有不太一样的地方，是归一化的位置，这个在进入层之前，Pre-LN——更稳定
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) #FFN

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257   # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 endpoint
    n_layers: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享 /*TODO：paper to read maybe*/
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 对于残差连接的权重矩阵放缩，训练稳定，防止梯度消失/爆炸
            std = 0.02
            if hasattr(module, 'MYGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type.startswith('cuda')
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        # 从huggingface加载参数
        assert  model_type in {'gpt2', 'gpt2-large', 'gpt2-medium', 'gpt2-xl'}
        print('Loading pretrained model from {}'.format(model_type))

        # 根据模型决定参数
        config_args = {
            'gpt2': dict(n_layers=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium': dict(n_layers=24, n_head=16, n_embd=1024), # 350M
            'gpt2-large': dict(n_layers=36, n_head=20, n_embd=1280), # 774M
            'gpt2-xl': dict(n_layers=48, n_head=25, n_embd=1600) # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # huggingface的模型加载
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model.state_dict()

        # 复制参数
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # openAI [in_dim,out_dim] 与torch的nn.linear相反 ?
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].clone())
            # 是不是huggingface那对conv1d更新了torch，现在不需要转置了？这样些不对，直接copy是对的
            # if any(k.endswith(w) for w in transposed):
            #     # special treatment for the Conv1D weights we need to transpose
            #     # assert sd_hf[k].shape[::-1] == sd[k].shape
            #     with torch.no_grad():
            #         sd[k].copy_(sd_hf[k].t().clone())
            # else:
            #     # vanilla copy over the other parameters
            #     assert sd_hf[k].shape == sd[k].shape
            #     with torch.no_grad():
            #         sd[k].copy_(sd_hf[k])

        return model


def get_most_likely_row(tokens, mask, logits):
    """
    在HellaSwag的4个选项中找出模型认为最可能的那个

    Args:
        tokens: 输入token序列 (4, seq_len) - 4个选项的完整序列
        mask: 掩码 (4, seq_len) - 标记哪些位置是选项部分（需要计算损失的部分）
        logits: 模型输出 (4, seq_len, vocab_size) - 每个位置的词汇概率分布

    Returns:
        pred_norm: 最可能选项的索引 (0-3)
    """

    # 将logits和tokens进行偏移，实现"给定前面的词，预测下一个词"
    shift_logits = (logits[..., :-1, :]).contiguous()  # 去掉最后一个位置的预测
    shift_tokens = (tokens[..., 1:]).contiguous()  # 去掉第一个位置的token

    # 举例：原序列 [A, B, C, D, E]
    # shift_logits: 对应位置[A, B, C, D]的预测分布
    # shift_tokens: 对应目标[B, C, D, E]
    # 即用A预测B，用B预测C，以此类推

    # 展平张量以便计算交叉熵损失
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # (4*seq_len, vocab_size)
    flat_shift_tokens = shift_tokens.view(-1)  # (4*seq_len,)

    # 计算每个位置的交叉熵损失（困惑度的组成部分）
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    # reduction='none'表示不求平均，保留每个位置的单独损失

    # 重新整形回原来的维度
    shift_losses = shift_losses.view(tokens.size(0), -1)  # (4, seq_len-1)

    # 对mask也进行相同的偏移操作
    shift_mask = (mask[..., 1:]).contiguous()  # (4, seq_len-1)

    # 将损失与mask相乘，只保留选项部分的损失
    masked_shift_losses = shift_losses * shift_mask  # (4, seq_len-1)
    # 情境部分的损失被置零，选项部分的损失保留

    # 计算每个选项的平均损失
    sum_loss = masked_shift_losses.sum(dim=1)  # (4,) - 每个选项的总损失
    avg_loss = sum_loss / shift_mask.sum(dim=1)  # (4,) - 每个选项的平均损失

    # 损失越低表示模型认为这个选项越可能
    pred_norm = avg_loss.argmin().item()
    return pred_norm

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
print(f"Using device: {device}")

enc = tiktoken.get_encoding("gpt2")
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B = 32
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total batch size: {total_batch_size}")
    print(f"=> grad_accum_steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_process=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_process=ddp_world_size, split="val")

# 设置矩阵乘法的精度 这个是第二档，使用较高精度（可能启用 TF32，但更谨慎）
torch.set_float32_matmul_precision('high')

# 变成2幂的优雅数字
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

max_lr = 6e-4
min_Lr = max_lr * 0.1
warmup_steps = 715 #375e6(对齐GPT3的预热） /2**19
max_steps = 19073 #10e9(数据集大概的tokens)/2**19

# optimizer /*TODO*/AdamW详细公式
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, betas=(0.9,0.95), device_type=device)

use_compile = False # torch.compile interferes with HellaSwag eval and Generation. 好像是bug，没明白compile之后就不能运行了，先false了，作者那边好像也鸽了，明天再试吧/*TODO*/fix
if use_compile:
    # 编译加速 减少GPU的读写，一次计算； 部分操作跳过python解释器
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # 变成了三层结构，必须通过module访问

# 普通训练版本——学习中
# for step in range(max_steps):
#     t0 = time.time()
#
#     optimizer.zero_grad()
#     loss_accum = 0.0
#     for micro_step in range(grad_accum_steps):
#         x, y = train_loader.next_batch()
#         x, y = x.to(device), y.to(device)
#         # 对于部分操作自动切换，且范围和F32一致，不需要梯度放缩器
#         with torch.autocast(device_type=device, dtype=torch.bfloat16):
#             logits, loss = model(x, y)
#         # 对于梯度的贡献是分摊的
#         loss = loss / grad_accum_steps
#         loss_accum += loss.detach()
#         # 直接改变DDP内部的变量，实现通信梯度平均，没有使用官方的语法糖
#         if ddp:
#             model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
#         loss.backward()
#     if ddp:
#         dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
#     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     lr = get_lr(step)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     optimizer.step()
#     torch.cuda.synchronize()
#     t1 = time.time()
#     dt = (t1-t0)*1000
#     tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size)/(t1-t0)
#     if master_process:
#         print(f"step: {step} | loss_accum: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

# 日志
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

# 评估+训练——完整版
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # 验证评估 与上面的训练类似
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()  # 重置验证数据加载器，确保从头开始

        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20

            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)

                # 使用混合精度加速推理，bfloat16在现代GPU上更高效
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)

                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

            # 定期保存模型检查点 (每5000步或最后一步)
            if step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),  # 模型参数
                    'config': raw_model.config,  # 模型配置
                    'step': step,  # 当前训练步数
                    'val_loss': val_loss_accum.item()  # 验证损失
                }
                torch.save(checkpoint, checkpoint_path)

    # HellaSwag评估
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0  # 正确预测的数量
        num_total = 0  # 总样本数量

        for i, example in enumerate(iterate_examples("val")):
            # 分布式处理：每个进程只处理属于自己的样本，这样可以并行处理，提高评估速度
            if i % ddp_world_size != ddp_rank:
                continue

            # 格式转换
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # 获取模型预测
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                # 根据mask找出最可能的选择
                pred_norm = get_most_likely_row(tokens, mask, logits)

            num_total += 1
            num_correct_norm += int(pred_norm == label)  # 预测正确则加1

        # 分布式汇总：将所有进程的统计结果合并
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)  # 汇总总数
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)  # 汇总正确数
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()

        acc_norm = num_correct_norm / num_total  # 计算准确率
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # 文本生成展示
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4  # 生成4个不同的序列进行对比
        max_length = 32  # 最大生成长度

        # 设置起始提示词
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        # 复制成4份，用于生成不同的序列
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)

        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)  # 不同进程使用不同种子

        while xgen.size(1) < max_length:
            # 前向传播获取下一个token的概率分布
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)  # (B, T, vocab_size)

                # 只取最后一个位置的logits，这是下一个token的预测
                logits = logits[:, -1, :]  # (B, vocab_size)
                # 转换为概率分布
                probs = F.softmax(logits, dim=-1)

                # Top-K采样 (K=50)：从概率最高的50个token中采样
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # 从top-k中按概率随机选择
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                xgen = torch.cat((xgen, xcol), dim=1)

        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # 主要训练步骤——与学习版一致
    model.train()  # 切换到训练模式
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        # 前向传播，使用混合精度训练加速
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize()

    # 监控日志
    t1 = time.time()
    dt = t1 - t0

    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(
            f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

# 销毁分布式进程组，释放资源
if ddp:
    destroy_process_group()