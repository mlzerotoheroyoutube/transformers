"""Transformer Architecture: code from the ML Zero to Hero explainer.

Video: https://youtu.be/bPHNUWcCBp0
"""

# A modern decoder-only Transformer from scratch: pre-norm, RMSNorm, RoPE, GQA, SwiGLU, tied embeddings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))
    def forward(self, x):                                   # x: (B, T, d)
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def rope_cache(T, d_head, base=10000.0):
    inv_freq = 1.0 / base ** (torch.arange(0, d_head, 2).float() / d_head)   # (d_head/2,)
    ang = torch.outer(torch.arange(T).float(), inv_freq)                    # (T, d_head/2)
    return ang.cos(), ang.sin()

def apply_rope(x, cos, sin):                                # x: (B, H, T, d_head)
    x1, x2 = x[..., 0::2], x[..., 1::2]                     # pair up dims (2i, 2i+1)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)

class Attention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads):
        super().__init__()
        self.h, self.kv, self.dh = n_heads, n_kv_heads, d // n_heads
        self.wq = nn.Linear(d, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d, n_kv_heads * self.dh, bias=False)
        self.wv = nn.Linear(d, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(n_heads * self.dh, d, bias=False)
    def forward(self, x, cos, sin):
        B, T, d = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)    # (B, H, T, dh)
        k = self.wk(x).view(B, T, self.kv, self.dh).transpose(1, 2)   # (B, KV, T, dh)
        v = self.wv(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.kv                                        # GQA: share each K/V head
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)          # (B, H, T, T)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        y = att @ v                                                    # (B, H, T, dh)
        return self.wo(y.transpose(1, 2).reshape(B, T, d))             # (B, T, d)

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1, self.w3 = nn.Linear(d, d_ff, bias=False), nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, d_ff):
        super().__init__()
        self.n1, self.attn = RMSNorm(d), Attention(d, n_heads, n_kv_heads)
        self.n2, self.mlp = RMSNorm(d), SwiGLU(d, d_ff)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)     # pre-norm residual
        return x + self.mlp(self.n2(x))

class TinyLM(nn.Module):
    def __init__(self, vocab=1000, d=256, n_layers=4, n_heads=8, n_kv_heads=2, d_ff=688, max_T=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, n_kv_heads, d_ff) for _ in range(n_layers))
        self.norm, self.head = RMSNorm(d), nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight                   # weight tying
        cos, sin = rope_cache(max_T, d // n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for p in self.parameters():                          # GPT-style small init
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)
    def forward(self, idx, targets=None):                    # idx: (B, T) int64
        T = idx.shape[1]
        x = self.emb(idx)                                    # (B, T, d)
        for blk in self.blocks:
            x = blk(x, self.cos[:T], self.sin[:T])
        logits = self.head(self.norm(x))                     # (B, T, vocab)
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

model = TinyLM()
print(f"params: {sum(p.numel() for p in model.parameters()):,}")   # 3,027,200
idx = torch.randint(0, 1000, (2, 64))
logits, loss = model(idx[:, :-1], idx[:, 1:])
print(logits.shape, f"initial loss {loss.item():.3f} vs ln(1000) = {math.log(1000):.3f}")

# overfit one batch to show it learns
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
for step in range(100):
    _, loss = model(idx[:, :-1], idx[:, 1:])
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
print(f"loss after 100 steps: {loss.item():.3f}")              # ≈ 0.1

# --------------------------------------------------------------------------

# Library version: the original 'base' encoder-decoder Transformer with torch.nn.Transformer
import torch
import torch.nn as nn
torch.manual_seed(0)

model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                       dim_feedforward=2048, dropout=0.1, batch_first=True, norm_first=False)
n_core = sum(p.numel() for p in model.parameters())
n_emb = 37000 * 512                       # shared source/target BPE vocab, tied with output layer
print(f"core: {n_core/1e6:.1f}M  + embeddings: {n_emb/1e6:.1f}M  = {(n_core+n_emb)/1e6:.1f}M")  # ≈ 63M (paper: ~65M)

src = torch.randn(2, 10, 512)             # (B, S, d) already-embedded source
tgt = torch.randn(2, 7, 512)              # (B, T, d) shifted-right target
causal = nn.Transformer.generate_square_subsequent_mask(7)   # (T, T) float mask, -inf above diagonal
out = model(src, tgt, tgt_mask=causal)
print(out.shape)                          # torch.Size([2, 7, 512])
