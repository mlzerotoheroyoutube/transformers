# Self-Attention

**Level:** Intermediate  ·  **Topic:** Transformers

▶️ **Watch the video:** https://youtu.be/J4vO9iuOFlQ

> Self-attention lets every position in a sequence build its new representation as a weighted average of all positions' values, with weights given by a softmax over query–key dot products: $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$. It is a learned, content-dependent mixing operator with $O(1)$ path length between tokens and $O(n^2)$ cost, and it is the core of the [[Transformer Architecture]].

## The intuition
Think of a video site's search. You type a **query** ("funny cat"). Every video has a **title** (call it a **key**) and the actual **content** (the **value**). The site compares your query with each title, gives better matches a bigger share of your attention, and you end up watching mostly the best-matching content.

In self-attention, *every word in a sentence does this search at the same time, over the same sentence*. Each word makes its own query ("what am I looking for?"), its own key ("what do I contain?") and its own value ("what will I hand over if picked?").

Take "I sat on the river bank". The word "bank" is confusing: money bank or river bank? Its query compares itself with every key. Say the match scores are: "I" 0, "sat" 1, "on" 0, "the" 0, "river" 3, "bank" 1. We turn scores into percentages that add to 100% (bigger scores get much bigger shares): "river" gets about 70%, "sat" and "bank" about 10% each, and the three small words about 3% each. The new meaning of "bank" is then 70% river-ish information mixed with a little of the rest. Now the computer "knows" it is a riverbank.

Nobody hand-writes these queries and keys. The model learns them from billions of sentences, the same way you learn which words matter by reading a lot.

## Core notes
- Formula: $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$; each output is a weighted average of values, weights sum to 1 per query.
- $Q, K, V$ are three learned linear projections of the **same** input $X$; that is what makes it *self*-attention.
- Scaling by $\sqrt{d_k}$: unit-variance $q,k$ give $\mathrm{Var}(q^\top k)=d_k$; dividing restores variance 1 and avoids softmax saturation.
- Masks are added to logits before softmax: $-\infty$ above the diagonal for causal LMs, $-\infty$ at padding positions.
- Cost: $O(n^2 d)$ time, $O(n^2)$ memory for the score matrix if materialised — the reason for FlashAttention and Efficient Attention.
- Permutation-equivariant without Positional Encoding or Rotary Position Embeddings: it sees a set, not a sequence.
- Path length between any two tokens is 1, versus $O(n)$ for Recurrent Neural Networks; all positions compute in parallel.
- In practice it runs as several heads in parallel: Multi-Head Attention.
- At inference, past keys and values are cached (KV Cache) so each new token costs $O(n\,d)$.
- Attention maps are useful for debugging and Mechanistic Interpretability but are not explanations by themselves.

## The math
**Definition.** For $X\in\mathbb{R}^{n\times d}$, $W_Q,W_K\in\mathbb{R}^{d\times d_k}$, $W_V\in\mathbb{R}^{d\times d_v}$ and mask $M\in\{0,-\infty\}^{n\times n}$:
$$
\mathrm{Attn}(X) = \mathrm{softmax}\!\left(\frac{XW_Q (XW_K)^\top}{\sqrt{d_k}} + M\right) XW_V
$$
Row $i$ of the output is $y_i = \sum_{j} a_{ij} v_j$ with $a_{ij} = \dfrac{\exp(q_i^\top k_j/\sqrt{d_k})\,[M_{ij}=0]}{\sum_{l}\exp(q_i^\top k_l/\sqrt{d_k})\,[M_{il}=0]}$.

**Why $\sqrt{d_k}$.** If the components of $q$ and $k$ are independent with mean 0 and variance 1,
$$
\mathbb{E}[q^\top k] = 0,\qquad \mathrm{Var}(q^\top k) = \sum_{m=1}^{d_k}\mathrm{Var}(q_m k_m) = d_k
$$
so logits have standard deviation $\sqrt{d_k}$. For $d_k=128$ that is $\approx 11$, pushing softmax toward one-hot where its Jacobian is nearly zero. Dividing by $\sqrt{d_k}$ restores unit variance.

**Softmax Jacobian** (why saturation kills gradients). For $a=\mathrm{softmax}(s)$:
$$
\frac{\partial a}{\partial s} = \operatorname{diag}(a) - a a^\top
$$
which tends to the zero matrix as $a$ approaches a one-hot vector.

**Permutation equivariance.** For a permutation matrix $P$ and $M=0$: $\mathrm{softmax}(PQK^\top P^\top)PV = P\,\mathrm{softmax}(QK^\top)P^\top P V = P\,\mathrm{Attn}(X)$, since row-wise softmax commutes with permuting rows and columns.

**Low-rank logits.** $S = X (W_Q W_K^\top) X^\top / \sqrt{d_k}$, and $\operatorname{rank}(W_QW_K^\top)\le d_k$, so $\operatorname{rank}(S)\le d_k$ before the softmax.

**Kernel view.** With $\kappa(q,k)=\exp(q^\top k/\sqrt{d_k})$,
$$
y_i = \frac{\sum_j \kappa(q_i,k_j)\,v_j}{\sum_j \kappa(q_i,k_j)}
$$
a Nadaraya–Watson estimator. Replacing $\kappa$ with $\phi(q)^\top\phi(k)$ lets the sums be reassociated as $\phi(q_i)^\top\big(\sum_j \phi(k_j)v_j^\top\big)$, giving linear-time attention (Efficient Attention).

**Complexity.** Time $O(n^2 d_k + n d d_k)$; memory $O(n^2 + nd)$ naive, $O(nd)$ with tiling. Per decoding step with a KV Cache: $O(n d_k)$ time and $O(n d_k)$ cache reads.

## Run the code
```bash
pip install numpy torch
python self_attention.py
```

## Check yourself
<details><summary>Why do we divide the dot products by √d_k?</summary>

If query and key components are roughly independent with zero mean and unit variance, $q^\top k$ is a sum of $d_k$ such products, so its variance is $d_k$. Large logits push the softmax toward a one-hot output, where its Jacobian $\operatorname{diag}(a)-aa^\top$ is close to zero, so gradients vanish and training stalls. Dividing by $\sqrt{d_k}$ returns the logits to unit variance at initialisation. It is equivalent to a fixed softmax temperature; some models learn it or replace it with QK-normalisation plus a learned scale.
</details>

<details><summary>What are the time and memory complexities of self-attention, and how do practical systems cope?</summary>

For sequence length $n$ and head width $d_k$, computing $QK^\top$ and $AV$ costs $O(n^2 d_k)$ FLOPs per head and storing the score matrix costs $O(n^2)$ memory. FlashAttention tiles the computation with an online softmax so the $n\times n$ matrix never reaches HBM, making memory linear while FLOPs stay quadratic. To reduce FLOPs one uses sliding-window or sparse patterns, linear/kernelised attention, or hybrids with state-space layers. At inference a KV cache turns each new token's cost into $O(n d_k)$ but makes cache size the bottleneck, which GQA/MQA and cache quantisation address.
</details>

<details><summary>Why use separate Q, K and V projections instead of attending with X directly?</summary>

With $Q=K=V=X$, the score $x_i^\top x_j$ is symmetric and every token attends most strongly to itself, since $x_i^\top x_i$ is usually the largest. Separate $W_Q$ and $W_K$ give an asymmetric learned bilinear form $x_i^\top W_Q W_K^\top x_j$, so "what I look for" can differ from "what I advertise". A separate $W_V$ decouples what is retrieved from how it is matched. Together with multiple heads, this lets different heads implement different relations (syntax, coreference, copying).
</details>

<details><summary>Is self-attention order-aware? What follows from your answer?</summary>

No. Without positional information it is permutation-equivariant: shuffling input tokens shuffles outputs identically, so "dog bites man" and "man bites dog" yield the same multiset of outputs. Order must be injected, either by adding position embeddings to the input (Positional Encoding) or by making the score depend on relative offset (relative biases, ALiBi, Rotary Position Embeddings). A causal mask also leaks some positional information, which is why decoder-only models without explicit positions (NoPE) still work to a degree. Equivariance is an advantage for set-structured inputs such as point clouds.
</details>

<details><summary>Your attention implementation produces NaNs only in padded batches. What is the likely cause and fix?</summary>

A query row whose keys are all masked has every logit equal to $-\infty$, and softmax of all $-\infty$ is $0/0$ = NaN. This happens for padding queries, or for left-padded sequences combined with a causal mask. Fixes: use a large finite negative (e.g. `torch.finfo(dtype).min`) instead of $-\infty$, ensure every row has at least one valid key, and mask those positions out of the loss. Also check that mask polarity matches the API, because PyTorch SDPA and `nn.MultiheadAttention` use opposite boolean conventions.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.