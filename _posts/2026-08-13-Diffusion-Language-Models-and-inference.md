---
title: "Diffusion Language Model and their Inference"
date: 2026-08-13
categories: [Technical Blog, Machine Learning]
tags: [Technical Blog, Language Models]
description: "Diffsuion Language Models and their Inference"
---


# Diffusion Language Model and their inference
A few months ago, I read the [jax-ml](https://jax-ml.github.io/scaling-book/) and wanted to write something from it then I came across a blog by inception AI where they introduced some sort of diffusion Language models as an alternative to the usual autoregressive LLMS. I became particularly interested in the inference framework of these DLMs (Diffusion Language models), given that in a vanilla DLLM Query and Key (QKV) are unavailable.

# A Brief Math Overview: From Diffusion to DLMs to dLLM Inference

## Gaussian diffusion to discrete diffusion

All continuous diffusion models such as [DDPMs](https://arxiv.org/abs/2006.11239) use a forward process that gradually destroys information by adding Gaussian noise. If $x_0$ is a clean data sample, the marginal distribution at timestep $t$ can be written as

$$
q(x_t \mid x_0)
=
\mathcal{N}\!\left(
    x_t;
    \sqrt{\bar{\alpha}_t}\,x_0,
    (1-\bar{\alpha}_t)I
\right),
$$

where

$$
\alpha_t = 1-\beta_t,
\qquad
\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s,
$$

and $\beta_t$ controls the amount of noise added at step $t$. Equivalently, we can sample $x_t$ directly as

$$
x_t
=
\sqrt{\bar{\alpha}_t}\,x_0
+
\sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

 $\bar{\alpha}_t$ measures how much signal from $x_0$ survives after $t$ noising steps. For a more detailed derivation of continuous diffusion, see [Weng (2021)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). 

Given its dicrete nature, text is different. A token is not a point in a continuous Euclidean space, its a discrete categorical object. So instead of adding Gaussian noise, discrete diffusion defines a Markov corruption process over categorical states.

A general discrete transition can be written as

$$
q(z_t\mid z_{t-1})
=
\operatorname{Cat}(z_t; Q_t z_{t-1}),
$$

where $Q_t$ is a transition matrix over the vocabulary. This idea is the building block of [D3PMs](https://arxiv.org/abs/2107.03006).

For the masked-diffusion case, a particularly useful marginal is the one used by [masked diffusion language models](https://arxiv.org/abs/2406.07524):

$$
q(z_t\mid x)
=
\operatorname{Cat}\!\left(
    z_t;
    \alpha_t x + (1-\alpha_t)\pi
\right),
$$

where $x$ is the clean token represented as a one-hot vector, $\pi$ is the terminal noise distribution, and $\alpha_t$ is now the probability that the clean token survives at noise level $t$. See [Remasking Discrete Diffusion Models with Inference-Time Scaling, Wang et al](https://arxiv.org/abs/2503.00307)


> **Notation note.** In the continuous DDPM equation above, $\bar{\alpha}_t$ is the cumulative signal-retention coefficient. In masked discrete diffusion papers, it is common to write the corresponding marginal survival probability directly as $\alpha_t$. The symbols look slightly different across papers, but they play similar roles. I also had minor problems because a lot of papers mix up $\beta$ and $t$. They arent quite the same but pretty similar concepts. They are proportional to each other.

> The notation abuse in a lot of ML papers is a lot. I had to settle on about 7 papers to focus on, I also used notation that I felt suited me better.

For timestep $s<t$, the transition between noisy states is

$$
q(z_t\mid z_s)
=
\operatorname{Cat}\!\left(
    z_t;
    \alpha_{t\mid s}z_s + (1-\alpha_{t\mid s})\pi
\right),
$$

with

$$
\alpha_{t\mid s}=\frac{\alpha_t}{\alpha_s}.
$$

This follows from consistency of the marginals:

$$
\alpha_{t\mid s}\alpha_s = \alpha_t.
$$

So the continuous Gaussian interpolation

$$
\mathcal{N}\!\left(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right)
$$

has a discrete analogue of the form

$$
\operatorname{Cat}\!\left(
\alpha_t x_0 + (1-\alpha_t)\pi
\right).
$$

---

## Tokens as one-hot vectors

This section borrows from [Simple and Effective Masked Diffusion Language Models, Sahoo et al](https://arxiv.org/abs/2406.07524).  

Suppose the vocabulary contains $K$ possible states. Represent one token as a one-hot vector

$$
x\in\mathcal{V},
\qquad
\mathcal{V}
=
\left\{
    x\in\{0,1\}^K:
    \sum_{i=1}^{K}x_i=1
\right\}.
$$

A model prediction, however, is generally not one-hot. It is a probability vector in the simplex
> Note: A probablity simplex is the set of all valid probablity vectors over the vocabulary bank
$$
\Delta^K
=
\left\{
    p\in\mathbb{R}^K:
    p_i\ge 0,
    \sum_{i=1}^{K}p_i=1
\right\}.
$$

For example, 

$$
x_0 = (0,1,0,0,0),
\qquad
\alpha_t=0.5,
$$

and let the last vocabulary entry be the special mask state

$$
m=(0,0,0,0,1).
$$

Then masked diffusion gives

$$
q(z_t\mid x_0)
=
\operatorname{Cat}\!\left(z_t;0.5x_0+0.5m\right),
$$

so

$$
0.5x_0+0.5m
=
(0,0.5,0,0,0.5).
$$

The token is therefore either still equal to the original token or has become `[MASK]`.

---

## Masked diffusion

Masked diffusion chooses the terminal distribution to be a point mass on the special mask token:

$$
\pi=m.
$$

The forward marginal becomes

$$
q(z_t\mid x_0)
=
\operatorname{Cat}\!\left(
    z_t;
    \alpha_t x_0+(1-\alpha_t)m
\right).
$$

Equivalently,

$$
z_t=
\begin{cases}
x_0, & \text{with probability }\alpha_t,\\[4pt]
m, & \text{with probability }1-\alpha_t.
\end{cases}
$$

Once a token is masked in the forward process, it stays masked at later, noisier times. This is why masking is also called an absorbing-state diffusion process.

For a sequence of $S$ tokens,

$$
x_0^{1:S}
=
\left(x_0^{(1)},\ldots,x_0^{(S)}\right),
$$

the forward corruption is usually applied independently across positions:

$$
q\!\left(z_t^{1:S}\mid x_0^{1:S}\right)
=
\prod_{i=1}^{S}
q\!\left(z_t^{(i)}\mid x_0^{(i)}\right).
$$

[LLaDA](https://arxiv.org/abs/2502.09992) uses a simple schedule: the continuous time variable $t\in[0,1]$ is itself the masking probability. In our notation,

$$
\alpha_t=1-t.
$$

Thus, for each token position $i$,

$$
q\!\left(z_t^{(i)}\mid x_0^{(i)}\right)
=
\begin{cases}
1-t, & z_t^{(i)}=x_0^{(i)},\\[4pt]
t, & z_t^{(i)}=m.
\end{cases}
$$

At $t=0$ the sequence is clean, while at $t=1$ it is fully masked.

---

## Reverse posterior

Consider a reverse transition from time $t$ to an earlier, cleaner time $s$, where

$$
0\le s<t\le 1.
$$

We want

$$
q(z_s\mid z_t,x_0).
$$

There are only two cases.

### Case 1: the token is already visible

Suppose

$$
z_t\neq m.
$$

Under absorbing-state masking, the only way a token can be visible at time $t$ is if it has never been masked. Therefore

$$
z_t=x_0,
$$

and at every earlier time $s<t$ it must be the same token. Hence

$$
q(z_s\mid z_t,x_0)
=
\delta_{z_t}(z_s),
\qquad z_t\neq m,
$$

where $\delta_{z_t}$ is a point mass at $z_t$.

In short: **already-visible tokens are copied backward unchanged**.

### Case 2: the token is masked

Now suppose

$$
z_t=m.
$$

At time $s$, the token could either already have been visible or still have been masked.

The probability that it was clean at $s$ but masked by $t$ is

$$
P(z_s=x_0,z_t=m\mid x_0)
=
\alpha_s-\alpha_t.
$$

The probability that it is masked at time $t$ is

$$
P(z_t=m\mid x_0)=1-\alpha_t.
$$

Using Bayes' rule,

$$
P(z_s=x_0\mid z_t=m,x_0)
=
\frac{\alpha_s-\alpha_t}{1-\alpha_t}.
$$

Similarly,

$$
P(z_s=m\mid z_t=m,x_0)
=
\frac{1-\alpha_s}{1-\alpha_t}.
$$

Therefore

$$
q(z_s\mid z_t=m,x_0)
=
\operatorname{Cat}\!\left(
    z_s;
    \frac{\alpha_s-\alpha_t}{1-\alpha_t}x_0
    +
    \frac{1-\alpha_s}{1-\alpha_t}m
\right).
$$

Putting the two cases together,

$$
q(z_s\mid z_t,x_0)
=
\begin{cases}
\delta_{z_t}(z_s),
& z_t\neq m,\\[8pt]
\operatorname{Cat}\!\left(
    z_s;
    \dfrac{\alpha_s-\alpha_t}{1-\alpha_t}x_0
    +
    \dfrac{1-\alpha_s}{1-\alpha_t}m
\right),
& z_t=m.
\end{cases}
$$

This is the key posterior behind masked diffusion language models.

---

## Exact posterior to a language model

During generation, $x_0$ is unknown.

So we train a Transformer to predict a distribution over possible clean tokens. At position $i$,

$$
p_\theta\!\left(x_0^{(i)}\mid z_t,t\right)
\in\Delta^K.
$$

For example, the network could output

$$
p_\theta\!\left(x_0^{(i)}\mid z_t,t\right)
=
\begin{cases}
0.7, & x_0^{(i)}=\texttt{cat},\\
0.2, & x_0^{(i)}=\texttt{dog},\\
0.1, & x_0^{(i)}=\texttt{clean}.
\end{cases}
$$

The learned reverse transition averages the exact posterior over the model's uncertainty about the clean token:

$$
p_\theta(z_s^{(i)}\mid z_t)
=
\sum_{x_0^{(i)}}
q\!\left(z_s^{(i)}\mid z_t^{(i)},x_0^{(i)}\right)
\,
p_\theta\!\left(x_0^{(i)}\mid z_t,t\right).
$$

If $z_t^{(i)}=m$, define

$$
r_{s,t}
=
\frac{\alpha_s-\alpha_t}{1-\alpha_t}.
$$

Then, for any vocabulary token $v\neq m$,

$$
p_\theta(z_s^{(i)}=v\mid z_t)
=
r_{s,t}\,
p_\theta\!\left(x_0^{(i)}=v\mid z_t,t\right),
$$

while

$$
p_\theta(z_s^{(i)}=m\mid z_t)
=
1-r_{s,t}
=
\frac{1-\alpha_s}{1-\alpha_t}.
$$

Using the example above, if $r_{s,t}=0.4$, then

$$
P(\texttt{cat})=0.4\times0.7=0.28,
$$

$$
P(\texttt{dog})=0.4\times0.2=0.08,
$$

$$
P(\texttt{clean})=0.4\times0.1=0.04,
$$

and

$$
P(\texttt{[MASK]})=0.6.
$$

So the reverse distribution is

$$
(0.28,0.08,0.04,0.60),
$$

which sums to one.

The Transformer supplies a distribution over the unknown clean token, while the diffusion posterior tells us how aggressively to move from the current noise level toward that prediction.

---

## Loss/Criterion objective

The diffusion model is trained by maximizing a variational lower bound on the data log-likelihood, or equivalently minimizing the negative ELBO (NELBO). A detailed introduction to ELBO can be found at [yunfans blog](https://yunfanj.com/blog/2021/01/11/ELBO.html).

Note: $\max_{\theta} \mathrm{ELBO} \Leftrightarrow \min_{\theta} \mathrm{NELBO}$

For a discretized diffusion process, a generic form is

$$
\mathcal{L}_{\mathrm{NELBO}}
=
\mathbb{E}_q\!\left[
    -\log p_\theta(x_0\mid z_0)
    +
    \sum_{k=1}^{K}
    D_{\mathrm{KL}}\!\left(
        q(z_{s_k}\mid z_{t_k},x_0)
        \,\|\,
        p_\theta(z_{s_k}\mid z_{t_k})
    \right)
\right],
$$

where $s_k<t_k$ are adjacent reverse-time points. A prior-matching term can also appear in the general ELBO, but for fully masked diffusion it vanishes when the terminal distribution is chosen to match the all-mask prior.

For masked diffusion, the KL terms simplify. Written as a positive cross-entropy loss, the discrete objective has the form

$$
\mathcal{L}_{K}
=
\mathbb{E}\!\left[
\sum_{k=1}^{K}
\frac{\alpha_{s_k}-\alpha_{t_k}}{1-\alpha_{t_k}}
\sum_{i=1}^{S}
\mathbf{1}\!\left[z_{t_k}^{(i)}=m\right]
\left(
    -\log p_\theta\!\left(x_0^{(i)}\mid z_{t_k},t_k\right)
\right)
\right].
$$

Only masked positions contribute to the loss.

Taking the continuous-time limit gives

$$
\mathcal{L}_{\infty}
=
\mathbb{E}\!\left[
\int_0^1
\frac{-\alpha_t'}{1-\alpha_t}
\sum_{i=1}^{S}
\mathbf{1}\!\left[z_t^{(i)}=m\right]
\left(
    -\log p_\theta\!\left(x_0^{(i)}\mid z_t,t\right)
\right)
\,dt
\right].
$$
[Remasking Discrete Diffusion Models with Inference-Time Scaling](https://arxiv.org/abs/2503.00307) goes into more derivation.

For the LLaDA schedule

$$
\alpha_t=1-t,
\qquad
\alpha_t'=-1,
$$

so

$$
\frac{-\alpha_t'}{1-\alpha_t}
=
\frac{1}{t}.
$$

This gives the LLaDA training objective

$$
\boxed{
\mathcal{L}(\theta)
=
-\mathbb{E}_{t,x_0,z_t}\!\left[
\frac{1}{t}
\sum_{i=1}^{S}
\mathbf{1}\!\left[z_t^{(i)}=m\right]
\log p_\theta\!\left(x_0^{(i)}\mid z_t\right)
\right]
}
$$

with

$$
t\sim\mathcal{U}(0,1).
$$

The factor $1/t$ isnt random. It comes from the continuous-time likelihood bound. It also compensates for the fact that small-$t$ examples contain fewer masked positions: the number of corrupted tokens is proportional to $t$, while the loss weight is proportional to $1/t$.

---

# Finally Inference

Suppose a prompt contains $C$ tokens and we allocate space for $G$ response tokens. The Transformer sequence length is

$$
S=C+G.
$$

Inference begins from

$$
z_1
=
[\text{prompt},\underbrace{m,\ldots,m}_{G\text{ response positions}}].
$$

The response region is initially fully masked.

At a reverse step from $t$ to $s<t$, the model produces distributions for all currently masked positions in one forward pass. Under the LLaDA schedule $\alpha_t=1-t$, the probability that a position masked at time $t$ becomes visible by time $s$ is

$$
\frac{\alpha_s-\alpha_t}{1-\alpha_t}
=
\frac{(1-s)-(1-t)}{t}
=
\frac{t-s}{t}.
$$

The fraction that should remain masked is therefore

$$
\frac{s}{t}.
$$

A simple practical LLaDA-style sampling iteration can be summarized as:

1. Run the Transformer and predict every masked token.
2. Assign each prediction a confidence score.
3. Keep the required number of high-confidence predictions so that the scheduled mask ratio decreases from $t$ to $s$.
4. Remask the lower-confidence predictions.
5. Repeat until the response is fully decoded or an EOS token terminates the answer.

The theory permits random remasking; LLaDA uses low-confidence remasking as an inference heuristic because it performs better in practice.
An autoregressive model commits one token per decoding iteration. A diffusion model can commit several. But a forward pass becomes expensive.

---

# Parallel inference and memory

Lets define

- $B$: batch size,
- $N$: number of Transformer layers,
- $d$: model width,
- $d_{\mathrm{ff}}$: feed-forward hidden width,
- $h_q$: number of query heads,
- $h_{kv}$: number of key/value heads,
- $d_h$: head dimension,
- $b$: bytes per stored scalar,
- $S$: current context length.

For simplicity, the main contrast here is between **autoregressive generation with reusable KV state** and **vanilla bidirectional diffusion generation, where the full sequence changes between denoising steps**.

## Autoregressive KV caching

In ordinary AR decoding, prefill processes the prompt once and stores the key and value projections from every layer. The KV cache contains two tensors, keys and values, so its memory is approximately

$$
\boxed{
M_{\mathrm{KV}}^{\mathrm{AR}}
=
2bBNS h_{kv}d_h
}
$$
> The 2 is to accomodate for key and query.

or simply

$$
M_{\mathrm{KV}}^{\mathrm{AR}}=\mathcal{O}(S)
$$

for fixed model architecture and batch size.

At each subsequent generation step, the active query length is only one token:

$$
q=1.
$$

The model computes activations for that new token and attends to the already-cached keys and values. The dominant per-layer activation terms are therefore roughly token-sized, for example

$$
M_{\mathrm{act}}^{\mathrm{AR}}
=
\mathcal{O}(B d_{\mathrm{ff}})
$$

for the feed-forward intermediates, while attention reads the $\mathcal{O}(S)$ KV cache.

A naive attention implementation would construct an attention-score workspace growing with context, i.e.

$$
\mathcal{O}(B h_q S),
$$

but fused kernels such as [FlashAttention](https://arxiv.org/abs/2205.14135) avoid materializing the full attention matrix in high-bandwidth memory. They reduce memory traffic and workspace, although they do not remove the underlying attention FLOPs. 

> We wont explore fused kernels here. \
>Important point is that old token representations do not need to be recomputed.

## Why vanilla dLLMs cannot reuse an AR-style KV cache

Now consider a diffusion denoising step $k$. Let

$$
z_{t_k}^{1:S}
=
\left(z_{t_k}^{(1)},\ldots,z_{t_k}^{(S)}\right)
$$

be the entire partially masked sequence, and write the hidden state of position $j$ at layer $\ell$ as

$$
h_j^{(\ell,k)}
=
f_\ell\!\left(z_{t_k}^{1:S}\right).
$$
>$f_l$  represents transformer computation.

Collecting all token states:

$$
H^{(\ell,k)}
=
\begin{bmatrix}
h_1^{(\ell,k)}\\
\vdots\\
h_S^{(\ell,k)}
\end{bmatrix}
\in\mathbb{R}^{S\times d}.
$$

The corresponding keys and values are

$$
K^{(\ell,k)}=H^{(\ell-1,k)}W_K^{(\ell)},
\qquad
V^{(\ell,k)}=H^{(\ell-1,k)}W_V^{(\ell)}.
$$

For DLLMs, we use bidirectional attention. If even one masked token changes between two denoising iterations,

$$
z_{t_{k+1}}\neq z_{t_k},
$$

that changed token can affect the contextual representation of every other token (remember that generation is in parallel!). In general,

$$
h_j^{(\ell,k+1)}\neq h_j^{(\ell,k)}
$$

for many positions $j$. Consequently,

$$
K^{(\ell,k+1)}\neq K^{(\ell,k)},
\qquad
V^{(\ell,k+1)}\neq V^{(\ell,k)}.
$$

So an AR-style KV cache is not reusable in vanilla full-sequence dLLM decoding.

Instead, each denoising iteration performs another forward pass over the full sequence. A useful schematic peak-activation model is

$$
M_{\mathrm{act}}^{\mathrm{dLLM}}
\approx
bB\left(
    c_1Sd
    +c_2Sd_{\mathrm{ff}}
    +c_3S(h_q+2h_{kv})d_h
\right)
+M_{\mathrm{attn}},
$$

where the constants $c_1,c_2,c_3$ depend on the implementation and $M_{\mathrm{attn}}$ is the attention-kernel workspace. $c_1,c_2,c_3$ correspond to the residuals, MLP and QKV respectively.

Ignoring constants,

$$
M_{\mathrm{act}}^{\mathrm{dLLM}}
=
\mathcal{O}(BSd),
$$

whereas the active-token activation footprint of one AR decoding step is roughly

$$
M_{\mathrm{act}}^{\mathrm{AR}}
=
\mathcal{O}(Bd).
$$

AR decoding still has an $\mathcal{O}(S)$ persistent KV cache, but that state is reusable. Vanilla diffusion repeatedly recomputes sequence-wide activations.

---

## Parallel prediction and compute

For one AR decoding step, a rough per-layer FLOP model is

$$
C_{\mathrm{AR,step}}
=
\mathcal{O}(d^2+Sd),
$$

where $d^2$ represents the projection/MLP work and $Sd$ represents attention against the existing context.

Generating $G$ tokens therefore costs approximately

$$
C_{\mathrm{AR}}
=
\mathcal{O}\!\left(
    NGd^2
    +
    Nd\sum_{j=1}^{G}(C+j)
\right).
$$

A vanilla dLLM full-sequence pass costs roughly

$$
C_{\mathrm{dLLM,pass}}
=
\mathcal{O}\!\left(
    NSd^2+NS^2d
\right).
$$

With $k$ denoising iterations,

$$
C_{\mathrm{dLLM}}
=
\mathcal{O}\!\left(
    kNSd^2+kNS^2d
\right).
$$

Let

$$
r=\frac{G}{k}
$$

be the average number of response tokens finalized per denoising iteration. If we look only at projection/MLP work and ignore attention, the ratio is

$$
\frac{C_{\mathrm{dLLM}}}{C_{\mathrm{AR}}}
\approx
\frac{kS}{G}
=
\frac{S}{r}.
$$

So merely predicting multiple tokens in parallel is not enough. The model must commit enough correct tokens per iteration to compensate for repeatedly evaluating a much larger active sequence.


The [JAX-ML Scaling Book](https://jax-ml.github.io/scaling-book/inference/) gives the arithmetic intensity of attention, in a simplified multi-head setting, as

$$
\text{AI}_{\mathrm{attn}}
\approx
\frac{ST}{S+T},
$$

where $T$ is the query length and $S$ is the key/value length.

For autoregressive generation,

$$
T=1,
\qquad
S\gg 1,
$$

so

$$
\frac{ST}{S+T}\approx 1.
$$

The operation performs relatively little arithmetic for the amount of memory that must be read, so generation attention is usually memory-bandwidth bound.

A full-sequence diffusion pass is much closer to prefill: many token positions are processed together, making the matrix multiplications larger and increasing hardware utilization. This is why a dLLM might achieve  good wall-clock throughput even while executing more FLOPs than an AR decoder.

---

# Addressing the memory and state-reuse problem

Recent dLLM inference improvement methods look like hybrids: preserve enough diffusion parallelism to decode several tokens at once, but introduce some causal structure to recover reusable state.

## Discrete Diffusion Forcing

[*Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing*](https://arxiv.org/abs/2508.09192) introduces **D2F**, which divides the sequence into blocks.

The attention structure is

- **causal between blocks**, and
- **bidirectional within each block**.

In this setup, once a block is complete, future blocks cannot modify its representation. Its key/value states can then be cached and reused.

At the same time, D2F does not require each preceding block to be completely denoised before later blocks begin making progress. The model is trained to predict later blocks from partially denoised predecessors.

If $A$ is the number of tokens in the currently active block pipeline and $S_{\mathrm{completed}}$ is the length of the completed prefix, a useful memory decomposition is

$$
M_{\mathrm{peak}}
\approx
M_{\mathrm{weights}}
+
M_{\mathrm{KV}}(S_{\mathrm{completed}})
+
M_{\mathrm{act}}(A),
$$

with

$$
A\ll S
$$

when only a small portion of the full context is actively being recomputed.

## Adaptive Parallel Decoding

Another approach is [*Accelerating Diffusion LLMs via Adaptive Parallel Decoding*](https://arxiv.org/abs/2506.00413), which introduces **Adaptive Parallel Decoding (APD)**.

APD imposes a left-to-right generation order and uses a small autoregressive model to decide how many dLLM proposals can be accepted in parallel. It also introduces systems optimisations aimed at the full-sequence recomputation.

One of them is a recompute KV window of width $W$. Tokens sufficiently far outside the active window are cached rather than recomputed every iteration. This can reduce a recomputed activation term from

$$
\mathcal{O}(BSd_{\mathrm{ff}})
$$

toward

$$
\mathcal{O}(BWd_{\mathrm{ff}}),
\qquad
W\ll S.
$$

---

# Conclusion

The mathematics of masked diffusion is neat:

$$
\text{clean token}
\longrightarrow
\text{masking process}
\longrightarrow
\text{weighted masked-token cross entropy}
\longrightarrow
\text{iterative parallel unmasking}.
$$

Much of the recent inference work is converging on hybrid designs: blockwise causal structure, partial KV reuse, active recomputation windows, adaptive token commitment etc.

Its going to be cool to follow how this problem would be solved an maybe one day they can be a sensible alternative to autoregressive models.

I started writing this blog the day I saw the tweet of lilian weng resigning from thinky machines. I was like, this name sounds familiar and I remember that I used to visit her blog a lot to learn about ML and diffusion. Given that I had also read the jax-ml book and was interested in DLMs, this just felt like the next path to pursue. I took a week break while writing due to travel but thankfully Im done.



---

# References

1. Jonathan Ho, Ajay Jain, and Pieter Abbeel. [**Denoising Diffusion Probabilistic Models.**](https://arxiv.org/abs/2006.11239) NeurIPS, 2020.

2. Lilian Weng. [**What are Diffusion Models?**](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) *Lil'Log*, 2021.

3. Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. [**Structured Denoising Diffusion Models in Discrete State-Spaces.**](https://arxiv.org/abs/2107.03006) NeurIPS, 2021.

4. Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, and Volodymyr Kuleshov. [**Simple and Effective Masked Diffusion Language Models.**](https://arxiv.org/abs/2406.07524) NeurIPS, 2024.

5. Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. [**Large Language Diffusion Models.**](https://arxiv.org/abs/2502.09992) NeurIPS, 2025.

6. JAX-ML. [**All About Transformer Inference.**](https://jax-ml.github.io/scaling-book/inference/) *How To Scale Your Model*, 2025.

7. JAX-ML. [**All the Transformer Math You Need to Know.**](https://jax-ml.github.io/scaling-book/transformers/) *How To Scale Your Model*, 2025.

8. Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. [**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.**](https://arxiv.org/abs/2205.14135) NeurIPS, 2022.

9. Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. [**Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models.**](https://arxiv.org/abs/2503.09573) 2025.

10. Daniel Israel, Guy Van den Broeck, and Aditya Grover. [**Accelerating Diffusion LLMs via Adaptive Parallel Decoding.**](https://arxiv.org/abs/2506.00413) NeurIPS, 2025.

11. Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, and Zhijie Deng. [**Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing.**](https://arxiv.org/abs/2508.09192) 2025.
