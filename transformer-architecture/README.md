# Transformer Architecture

**Level:** Intermediate  ·  **Topic:** Transformers

▶️ **Watch the video:** https://youtu.be/bPHNUWcCBp0

> A Transformer is a stack of identical blocks, each of which lets every token gather information from every other token (attention) and then thinks about it alone (a small MLP), with residual connections and normalisation holding the stack together. It replaced recurrence with parallel matrix multiplies, which is why it scales, and it is the backbone of essentially every frontier language, vision and multimodal model in 2026.

## The intuition
Picture a class reading a sentence where each student holds one word card. In round one, every student stands up, looks at all the other cards, and decides which ones matter for *their* word. The student holding "it" in "The dog chased the ball because it was bouncing" checks the room, notices "ball" fits better than "dog", and writes a note to themselves: "it = ball". That step is called **attention**.

Then every student sits down and thinks privately about their note for a moment, updating their understanding. That private step is a small calculator called a **feed-forward network**.

A Transformer repeats these two steps many times: GPT-3 does it 96 times. Each round adds understanding on top of the last, like layers of tracing paper, instead of erasing the old picture. That "add, don't replace" idea is a **residual connection**.

Tiny numbers: suppose "it" gives scores 1 to "dog", 3 to "ball" and 0 to "bouncing". Turning scores into shares that add up to 100% gives about 11% dog, 84% ball, 4% bouncing. So "it" mostly copies information from "ball".

The big trick is that all students do their looking **at the same time**. Older models (RNNs) read one word after another, like passing a note down a line. Doing it all at once is what lets Transformers use thousands of graphics chips efficiently.

## Core notes
- A block = **token mixing** (multi-head self-attention) + **channel mixing** (FFN), each wrapped in a residual connection and a norm.
- Shapes never change inside the stack: every block maps $(B,T,d)\to(B,T,d)$, which is what makes deep stacking trivial.
- Per-layer parameters $\approx 12d^2$ (attention $4d^2$, MLP $8d^2$); whole model $N \approx 12Ld^2 + Vd$.
- Training compute $\approx 6ND$ FLOPs for $N$ parameters and $D$ tokens; inference $\approx 2N$ FLOPs per token (plus the attention term) — see Scaling Laws.
- Three families: encoder-only (BERT), decoder-only (GPT Series), encoder–decoder (T5 and Text-to-Text); see Encoder vs Decoder Transformers.
- Order information is injected by Positional Encoding (2017) or Rotary Position Embeddings (modern default).
- Pre-norm + residuals give a clean gradient path, which is why 100+ layer models train; see Residual Connections and Vanishing and Exploding Gradients.
- Autoregressive inference is memory-bound; the KV Cache, GQA and FlashAttention exist to fix that.
- The same architecture runs on image patches (Vision Transformer), audio frames (Speech Recognition) and diffusion latents (Diffusion Transformers).
- Quadratic cost in $T$ motivates Efficient Attention and Long Context techniques.

## The math
**Block (pre-norm).** For residual stream $x^{(\ell)}\in\mathbb{R}^{T\times d}$:
$$
\begin{aligned}
a^{(\ell)} &= x^{(\ell)} + \mathrm{MHA}\big(\mathrm{Norm}(x^{(\ell)})\big) \\
x^{(\ell+1)} &= a^{(\ell)} + \mathrm{FFN}\big(\mathrm{Norm}(a^{(\ell)})\big)
\end{aligned}
$$
Post-LN (2017) instead uses $x^{(\ell+1)} = \mathrm{Norm}\big(a + \mathrm{FFN}(a)\big)$ with $a = \mathrm{Norm}(x + \mathrm{MHA}(x))$.

**Attention** (one head $i$, causal mask $M_{ts} = 0$ if $s\le t$ else $-\infty$):
$$
\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_h}} + M\right)V_i, \qquad \mathrm{MHA}(u) = [\mathrm{head}_1;\dots;\mathrm{head}_h]\,W_O
$$
with $Q_i = uW_Q^{(i)}$, $K_i = uW_K^{(i)}$, $V_i = uW_V^{(i)}$, $W^{(i)}\in\mathbb{R}^{d\times d_h}$, $W_O\in\mathbb{R}^{hd_h\times d}$.

**RMSNorm and SwiGLU.** With learned gain $g\in\mathbb{R}^d$ and $\mathrm{SiLU}(z)=z\,\sigma(z)$:
$$
\mathrm{RMSNorm}(x) = g\odot\frac{x}{\sqrt{\tfrac{1}{d}\sum_j x_j^2+\epsilon}}, \qquad \mathrm{FFN}(x) = \big(\mathrm{SiLU}(xW_1)\odot xW_3\big)W_2
$$

**Objective.** Autoregressive negative log-likelihood over tokens $y_1,\dots,y_T$:
$$
\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T-1}\log p_\theta(y_{t+1}\mid y_{\le t}), \qquad p_\theta(\cdot\mid y_{\le t}) = \mathrm{softmax}\big(x_t^{(L)}W_U\big)
$$

**Parameter count.** Per layer with $h_{kv}$ KV heads and a SwiGLU MLP:
$$
P_{\text{layer}} = \underbrace{2\,d\,h d_h}_{W_Q,\,W_O} + \underbrace{2\,d\,h_{kv} d_h}_{W_K,\,W_V} + \underbrace{3\,d\,d_{ff}}_{W_1,W_2,W_3} + \underbrace{2d}_{\text{norms}}
$$
For MHA with a $4d$ GELU MLP this is $12d^2$, so $N \approx 12Ld^2$ non-embedding parameters. Check: GPT-3 with $L=96$, $d=12288$ gives $12\cdot 96\cdot 12288^2 \approx 1.74\times10^{11}$.

**Compute.** Forward FLOPs per token (Kaplan et al., 2020), with $T$ the context length:
$$
C_{\text{fwd}} \approx 2N + 2LTd, \qquad C_{\text{train}} \approx 6ND \;\;(\text{forward} + 2\times \text{backward})
$$
The ratio of attention-score FLOPs to dense FLOPs is $\frac{4LTd}{24Ld^2} = \frac{T}{6d}$ (non-causal), so attention dominates only when $T > 6d$ — about 24k tokens for $d=4096$.

**Memory.** Attention scores need $O(BhT^2)$ if materialised (avoided by FlashAttention); the KV Cache needs $2\,L\,h_{kv}\,d_h\,T\,B$ values at inference.

## Run the code
```bash
pip install torch
python transformer_architecture.py
```

## Check yourself
<details><summary>Walk through the tensor shapes of one decoder block for input (B, T, d) with h heads.</summary>

The input $(B,T,d)$ is normed (same shape), then projected to $Q,K,V$ each $(B,T,d)$ for MHA. Each is reshaped to $(B,T,h,d_h)$ and transposed to $(B,h,T,d_h)$ with $d_h=d/h$. Scores $QK^\top/\sqrt{d_h}$ are $(B,h,T,T)$; after the causal mask and softmax, multiplying by $V$ gives $(B,h,T,d_h)$. Heads are transposed back and concatenated to $(B,T,d)$, projected by $W_O$, and added to the residual. The FFN expands to $(B,T,d_{ff})$ and contracts back to $(B,T,d)$, followed by another residual add. Every block preserves $(B,T,d)$.
</details>

<details><summary>How many parameters and FLOPs are in a Transformer layer, and when does attention dominate the compute?</summary>

With MHA and a $4d$ MLP, attention projections hold $4d^2$ parameters and the MLP $8d^2$, so about $12d^2$ per layer. Forward FLOPs per token are roughly $2\times$ parameters, i.e. $24d^2$, plus $4Td$ for $QK^\top$ and $AV$ over a context of $T$ tokens. The attention term exceeds the dense term when $4Td > 24d^2$, i.e. $T > 6d$. For a $d=4096$ model that is about 24k tokens, so at ordinary context lengths the MLP and projections dominate; at very long context attention does.
</details>

<details><summary>Pre-LN vs post-LN: what is the difference and why did the field move to pre-LN?</summary>

Post-LN (the 2017 design) normalises after the residual addition, $x \leftarrow \mathrm{LN}(x + f(x))$, so the residual path passes through every norm. Xiong et al. (2020) showed that at initialisation post-LN has large gradients near the output layers, which is why it needs a careful learning-rate warmup and becomes unstable when deep. Pre-LN computes $x \leftarrow x + f(\mathrm{LN}(x))$, leaving an identity path from input to output and giving well-behaved gradients without warmup. The cost is that the residual stream norm grows with depth and later layers can contribute less, so some models add extra norms (e.g. on sublayer outputs) or use DeepNorm-style scaling to recover post-LN quality.
</details>

<details><summary>Why are Transformers easier to train at scale than RNNs?</summary>

An RNN's hidden state at step $t$ depends on step $t-1$, so training over a sequence of length $T$ requires $T$ sequential steps and gradients flow through $T$ Jacobian products, inviting vanishing or exploding gradients. Self-attention connects any two positions in one step (path length $O(1)$) and computes all positions in parallel with large matrix multiplies, which map well onto GPUs and TPUs. Residual connections and normalisation keep gradients well conditioned across depth. The price is $O(T^2)$ attention cost and, at inference, a growing KV cache instead of a fixed-size state.
</details>

<details><summary>List the main architectural changes between the 2017 Transformer and a Llama-3-style model, and the reason for each.</summary>

Decoder-only instead of encoder–decoder, because a single next-token objective scales simply and supports prompting. Pre-norm RMSNorm instead of post-LayerNorm, for stability and lower cost. SwiGLU MLPs with $d_{ff}\approx\tfrac{8}{3}d$ instead of ReLU $4d$, for better quality at equal FLOPs. RoPE inside attention instead of added sinusoidal embeddings, for relative-position behaviour and context extension. GQA instead of MHA, to shrink the KV cache for fast inference. No biases and no dropout in pretraining, plus bf16 training with FlashAttention kernels.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.