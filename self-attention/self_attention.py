"""Self-Attention: code from the ML Zero to Hero explainer.

Video: https://youtu.be/J4vO9iuOFlQ
"""

# From scratch in NumPy: single-head self-attention with an optional causal mask
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)          # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def self_attention(X, Wq, Wk, Wv, causal=False):
    """X: (n, d_model). Returns output (n, d_v) and attention weights (n, n)."""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                  # (n, d_k), (n, d_k), (n, d_v)
    scores = Q @ K.T / np.sqrt(K.shape[-1])          # (n, n)
    if causal:
        scores = np.where(np.tril(np.ones_like(scores)) == 1, scores, -np.inf)
    A = softmax(scores, axis=-1)                     # each row sums to 1
    return A @ V, A

rng = np.random.default_rng(0)
n, d_model, d_k = 4, 8, 8
X = rng.normal(size=(n, d_model))
Wq, Wk, Wv = (rng.normal(size=(d_model, d_k)) / np.sqrt(d_model) for _ in range(3))
out, A = self_attention(X, Wq, Wk, Wv, causal=True)
print(out.shape, A.round(2), A.sum(-1), sep="\n")    # lower-triangular rows summing to 1

# Permutation equivariance (no positional info): permuting tokens permutes outputs
perm = rng.permutation(n)
out_p, _ = self_attention(X[perm], Wq, Wk, Wv)
out_f, _ = self_attention(X, Wq, Wk, Wv)
print("equivariant:", np.allclose(out_p, out_f[perm]))   # True

# --------------------------------------------------------------------------

# PyTorch module, checked against the fused kernel, plus the sqrt(d_k) experiment
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_k, bias=False)   # fused projection
        self.d_k = d_k
    def forward(self, x, causal=True):                        # x: (B, n, d_model)
        q, k, v = self.qkv(x).chunk(3, dim=-1)                # each (B, n, d_k)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)     # (B, n, n)
        if causal:
            n = x.shape[1]
            s = s.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), float("-inf"))
        return s.softmax(-1) @ v                              # (B, n, d_k)

attn = SelfAttention(d_model=64, d_k=64)
x = torch.randn(2, 16, 64)
y = attn(x)
q, k, v = attn.qkv(x).chunk(3, dim=-1)
y_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # fused kernel (Flash/mem-efficient on GPU)
print(y.shape, torch.allclose(y, y_ref, atol=1e-5))               # torch.Size([2, 16, 64]) True

# Why scale by sqrt(d_k): unscaled logits have variance ≈ d_k and the softmax saturates
for d in (16, 256, 4096):
    qq, kk = torch.randn(10000, d), torch.randn(10000, d)
    print(d, f"var(q·k)={((qq * kk).sum(-1)).var():.0f}", f"var(q·k/√d)={((qq * kk).sum(-1) / math.sqrt(d)).var():.2f}")
