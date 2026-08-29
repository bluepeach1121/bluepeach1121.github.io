---
title: "Diffusion Language Model and their Inference"
date: 2026-08-29
categories: [Technical Blog, Machine Learning]
tags: [Technical Blog, Language Models]
description: "Diffusion Language Models and their Inference"
math: True
---


# Diffusion Language Model and their inference
A few months ago, I read the [jax-ml](https://jax-ml.github.io/scaling-book/) and wanted to write something from it then I came across a blog by inception AI where they introduced some sort of diffusion Language models as an alternative to the usual autoregressive LLMs. I started writing this blog the day I saw the tweet of Lilian Weng resigning from thinking machines. I remembered that I used to visit her blog a lot to learn about ML. Given that I had also read the jax-ml book and was interested in diffusion language model, this just felt like the next path to pursue. I took a about ten days off due to travel and changing apartments but thankfully I'm done.


# A Brief Math Overview: from Diffusion to DLMs to DLM Inference

A forward diffusion process samples a datapoint $x_0$ and gradually adds noise to corrupt the data in $T$ steps over timestep $t = 0$ to $t = 1$. We will define the final corrupted data as $z_T$.
Nearly all modern diffusion processes use a gaussian forward process (modelled as):

$$
q(x_t|x_0)=\mathcal N(x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)
$$

[Denoising Diffusion Probabilistic Models (DDPM), Ho et al](https://arxiv.org/pdf/2006.11239)

where
$\bar\alpha_t\to$  cumulative amount of signal preserved after $t$ timesteps
$\bar\alpha_t\to \prod_{s=1}^{t}\alpha_s$ ; and
$\alpha_t\to 1-\beta_t$; where $\beta_t$ is the noise variance.
> Note: There is a binomial form of diffusion models
>
> $$
> q(x_t \mid x_{t-1})
> =
> \mathcal{B}\left(
> x_t;
> (1-\beta_t)x_{t-1}
> +
> \frac{\beta_t}{2}
> \right)
> $$
> $$\mathcal{B}\rightarrow \text{bernoulli distribution}$$ 
> which we won't consider in this blog. It was introduced in
> [Deep Unsupervised Learning using Nonequilibrium Thermodynamics, Sohl-Dickstein et al](https://arxiv.org/abs/1503.03585) alongside the Gaussian variant, but its use case is largely limited to discrete binary data. 

Accordingly,

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon\ ;\ \epsilon\sim\mathcal N(0,I)
$$

Since text tokens aren't continuous, the diffusion equations are slightly reformulated to accommodate for the discrete nature of text. Rather than adding gaussian noise to a continuous value, we define a probability distribution over the discrete vocabulary. The continuous form can be reformulated as a Markov forward process: 

$$
q(z_t|z_{t-1})=Cat(z_t;Q_tz_{t-1})
$$


$z_t\rightarrow$  Corrupted discrete token at time $t$.
$Q_t\rightarrow$ A Transition matrix over the vocabulary. [Good Wikipedia link on stochastic matrix](https://en.wikipedia.org/wiki/Stochastic_matrix)
$Q_t​z_{t-1}$​ produces a probability vector over all possible tokens that $z_t$​ could transition to.
$Cat(.;p)\rightarrow$ [Categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution) with probability vector $p$.
$\pi\rightarrow$ Terminal noise distribution over the vocabulary.

By repeatedly applying the transition process during the forward process the probability mass moves from the original token to the terminal noise distribution $\pi$. This gives us

$$
q(z_t|x_0)=Cat(z_t;\alpha_tx_0+(1-\alpha_t)\pi).
$$

Here we follow the notation from the DDPM paper, where $\bar{\alpha}_t$ denotes the cumulative signal-preservation term:

$$
q(x_t \mid x_0)
=
\mathcal{N}
\left(
x_t;
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right).
$$

$\text{i.e.}$

$$
\mathcal{N}
\left(
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right)
\;\leftrightarrow\;
\operatorname{Cat}
\left(
\alpha_t x_0+(1-\alpha_t)\pi
\right).
$$

> Note: $\bar{\alpha}_t$ on the Gaussian side and $\alpha_t$ on the discrete side both represent cumulative signal preservation. Later in the blog, we switch fully to the [MDLM](https://arxiv.org/pdf/2406.07524) notation and use $\alpha_t$.

See [Simple and Effective Masked Diffusion Language Models (MDLM)](https://arxiv.org/abs/2406.07524) for further details.


 > Note the lack of square root in discrete diffusion form below:
 > 
> $$
> q(z_t|x_0)=Cat(z_t;\alpha_tx_0+(1-\alpha_t)\pi).
> $$
>
> This is because the coefficients are probabilities $Pr(z_t=x_0)=\alpha_t\ ||\Pr(z_t\sim\pi)=1-\alpha_t$ and they must sum to 1. i.e $(\alpha+(1-\alpha)=1)$
> $\therefore$ for $s<t$
> 
> $$
> q(z_t|z_s)=Cat(z_t;\alpha_{t|s}z_s+(1-\alpha_{t|s})\pi)
> $$
>
 > where $\alpha_{t|s}=\frac{\alpha_t}{\alpha_s}$
>[Lilian Weng's](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) blog has a clearer derivation of this.

Since
$$
q(z_s|x)=Cat(z_s;\alpha_sx+(1-\alpha_s)\pi)
$$

and

$$
q(z_t|z_s)=Cat(z_t;\alpha_{t|s}z_s+(1-\alpha_{t|s})\pi)
$$

Marginalising the probability  over $z_s$, we get

$$
q(z_t|x)=Cat(z_t;\alpha_{t|s}\alpha_sx+(1-\alpha_{t|s}\alpha_s)\pi)
$$

For consistency with the general equation form of

$$
q(z_t|x)=Cat(z_t;\alpha_tx+(1-\alpha_t)\pi)
$$

we need $\alpha_{t|s}\alpha_s=\alpha_t$

$$
\therefore\ \alpha_{t|s}=\frac{\alpha_t}{\alpha_s}
$$

---

### Example
Let $x_0$ be the original clean token before any data corruption. Let the vocabulary contain $V$ tokens, with $x_0$ represented as a $V$-dimensional one-hot vector:

$$
x_0 \in \{e_1,\ldots,e_V\}\subset\{0,1\}^V
$$

Define the set of all valid one-hot token vectors as

$$
\mathcal V
=
\left\{
x\in\{0,1\}^V:
\sum_{i=1}^{V}x_i=1
\right\}
=
\{e_1,\ldots,e_V\}.
$$

Assume

$$
x_0=[0, 1, 0, 0, 0];\ \alpha_t=0.5\ and \ \pi=m
$$

where $m$ = $\pi$ represents the terminal token.

$$
m =[0, 0, 0, 0, 1]
$$

$$
q(z_t|x_0)=Cat(z_t;\alpha_tx_0+(1-\alpha_t)m)
$$

$$
=0.5[0, 1, 0, 0, 0]+0.5[0, 0, 0, 0, 1]
$$

$$
=[0,0.5,0,0,0.5]
$$
---
>Note:
>
>$$
>z_t=\begin{cases}
>x_0; & probability\ \alpha_t\\
>m; & probability\ 1-\alpha_t
>\end{cases}
>$$

Since we know that a token masked during one forward process remain for the entirety of the forward process. For a sequence of $S$ tokens,

$$
x_0^{1:S}=(x_0^{(1)},\ldots,x_0^{(S)})
$$

the forward process becomes

$$
q(z_t^{1:S}|x_0^{1:S})=\prod_{i=1}^{S}q(z_t^{(i)}|x_0^{(i)})
$$

where $i\rightarrow$  token positions and $S$ is the total sequence length. For simplicity, we would be using [LLaDA](https://arxiv.org/abs/2502.09992) and LLaDA notations as a model for our DLM. In the paper, they used $\alpha_t=1-t$ but I find it more convenient to use

$$
\alpha_t=(1-\bar\beta_t)
$$

They are equivalent for now.

Consequently,

$$
q(z_t^{(i)}|x_0^{(i)})=
\begin{cases}
1-\bar\beta_t; & z_t^{(i)}=x_0^{(i)}\\
\bar\beta_t; & z_t^{(i)}=[MASK]
\end{cases}
$$

> Note: There's a lot of interchanging notation in diffusion paper math.  LLaDA uses a linear noise schedule $\bar{\beta}_t = t$ which I follow in the coming sections.

$$
{\LARGE\mathbf{\text{Simplified interpretation}}}
$$

$$
\underset{\scriptstyle t=0}{
  \bbox[12px,border:2px solid red;background:#e9eef2]{x_0}
}
\quad\longrightarrow\quad
\underset{\scriptstyle t=s}{
  \bbox[12px,border:2px solid red;background:#deda8c]{z_s}
}
\quad\longrightarrow\quad
\underset{\scriptstyle t=1}{
  \bbox[12px,border:2px solid red;background:#cf5d9d]{z_t}
}
$$

## The Reverse Posterior

Consider the backward step from $t$ to $s$; where $s<t$. Remember $z_t$ & $z_s$ are the token at noisier time $t$ and cleaner time $s<t$ respectively.  $\alpha_u\rightarrow$ probability that the original token is still visible at time $u$.

$$
\therefore\ q(z_u=x_0|x_0)=\alpha_u\ \text{ and }\ q(z_u=m|x_0)=1-\alpha_u
$$

### Case 1: $\{z_t\ne m\}$
Suppose at timestep $t$ our current token $z_t\ne m$ ; i.e it is still not covered with a $[MASK]$ token,

$$
z_t=x_0
$$

or 

$$
z_t=z_s=x_0
$$ 

with probability of 1. Hence

$$
q(z_s|z_t,x_0)=\delta_{z_t}(z_s);\ z_t\ne m.
$$

where

$$
\delta_{z_t}(z_s)=\begin{cases}
1 & z_s=z_t\\
0 & z_s\ne z_t
\end{cases}
$$

$\delta_{z_t}$ is a just point mass at $z_t$

In short; already visible tokens just gets copied.

### Case 2:  $z_t=m$
At time $s$; the token might be revealed to $x_0$ (its original token) or remained masked.
Let

$$
P(z_u=x_0|x_0)=\alpha_u
$$

$$
P(z_u=m|x_0)=1-\alpha_u
$$

Using Bayes' Theorem and conditional probability

Probability of $z_s$ being visible at timestep $s$

$$
P(z_s=x_0|z_t=m,x_0)
=\frac{P(z_s=x_0,z_t=m|x_0)}{P(z_t=m|x_0)}
=\frac{\alpha_s-\alpha_t}{1-\alpha_t}
$$

>Note the numerator is the probability that the token was visible at $s$ but masked by $t$.
and

$$
P(z_s=m|z_t=m,x_0)
=1-P(z_s=x_0|z_t=m,x_0)
=\frac{1-\alpha_s}{1-\alpha_t}
$$

or more formally (I guess)

$$
P(z_s=m,z_t=m|x_0)=P(z_s=m|x_0)=1-\alpha_s
$$

$$
\therefore\ P(z_s=m|z_t=m,x_0)=\frac{1-\alpha_s}{1-\alpha_t}
$$

Hence

$$
q(z_s|z_t=m,x_0)=Cat\left(\frac{\alpha_s-\alpha_t}{1-\alpha_t}x_0+\frac{1-\alpha_s}{1-\alpha_t}m\right)
$$

$\frac{\alpha_s-\alpha_t}{1-\alpha_t}$ $\rightarrow$ prob of unmasking between $t$ and $s$

$\frac{1-\alpha_s}{1-\alpha_t}$ $\rightarrow$ probability of staying masked.

## Slight Detour:

## Diffusion Models as an approximate subset of Continuous Normalising Flows

For a diffusion model, we've established that

$$
x_{t+1}=\sqrt{1-\beta_t}\,x_t+\sqrt{\beta_t}\,\epsilon_t
\;;\; 
\epsilon\sim\mathcal{N}(0,1)
$$

as a variance-preserving forward process.
For a small $\beta_t$

$$
\sqrt{1-\beta_t}\approx 1-\frac{1}{2}\beta_t
$$

$$
\therefore\quad
x_{t+1}\approx x_t-\frac{1}{2}\beta_t x_t+\sqrt{\beta_t}\epsilon_t
$$

For a continuous process, this becomes

$$
dx_t=-\frac{1}{2}\beta(t)x_t\,dt+\sqrt{\beta(t)}\,dW_t
$$

a standard form is:

$$
dx_t=f(x_t,t)dt+g(t)dW_t
$$

where

$$
f(x,t)=-\frac{1}{2}\beta(t)x
\quad\text{and}\quad
g(t)=\sqrt{\beta(t)}
$$

$dW_t$ is an infinitesimal increment of Brownian Motion (To represent random noise).

$$
dW_t\approx\sqrt{dt}\,\epsilon
\qquad
\epsilon\sim\mathcal{N}(0,1)
$$

Subsequent steps eventually take the form of:

$$

x_{t+\Delta t}

\approx

\underbrace{x_t}_{\text{current noisy state}}

-

\underbrace{\frac{1}{2}\beta(t)x_t\Delta t}_{\text{degradation of current signal}}

+

\underbrace{\sqrt{\beta(t)\Delta t}\,\epsilon}_{\text{noise}}

$$

The reverse process requires

$$
s_\theta(x,t)\approx\nabla_x\log p_t(x)
$$

where:
$p \rightarrow$ probability density of noisy state $x$ at time $t$. We don't need to know $p$, the gradient is enough and the reverse process becomes

$$
dx_t=
\left[
f(x_t,t)-g(t)^2s_\theta(x_t,t)
\right]dt
+
g(t)d\bar{W}_t
$$

For our diffusion model, we get (reverse-time SDE)

$$
dx_t=
\left[
-\frac{1}{2}\beta(t)x_t-\beta(t)s_\theta(x_t,t)
\right]dt
+
\sqrt{\beta(t)}\,d\bar{W}_t
$$

$$
{\LARGE\mathbf{\text{From Reverse Time SDE to Probablility Flow ODE}}}
$$

> Starting from the reverse-time SDE,
>
> $$
> dx_t
> =
> \left[
> f(x_t,t)-g(t)^2\nabla_x\log p_t(x_t)
> \right]dt
> +
> g(t)\,d\bar W_t,
> $$
>
> The probability density evolves according to the [Fokker-Planck equation](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation)(comparing coefficients and plugging them in from Wikipedia)
>
> $$
> \frac{\partial p_t(x)}{\partial t}
> =
> -\nabla_x\cdot
> \left[
> \left(
> f(x,t)-g(t)^2\nabla_x\log p_t(x)
> \right)p_t(x)
> \right]
> -
> \frac{1}{2}g(t)^2\Delta_x p_t(x).
> $$
>
> Using
>
> $$
> p_t(x)\nabla_x\log p_t(x)
> =
> \nabla_x p_t(x),
> $$
>Identity : $\nabla \log p = \frac{\nabla p}{p}$
> we combine the two score-dependent terms:
>
> $$
> \frac{\partial p_t(x)}{\partial t}
> =
> -\nabla_x\cdot
> \left[
> \left(
> f(x,t)
> -
> \frac{1}{2}g(t)^2\nabla_x\log p_t(x)
> \right)
> p_t(x)
> \right].
> $$
>
> This has the continuity equation form
>
> $$
> \frac{\partial p_t(x)}{\partial t}
> =
> -\nabla_x\cdot
> \left[
> v_t(x)p_t(x)
> \right],
> $$
>
> with
>
> $$
> v_t(x)
> =
> f(x,t)
> -
> \frac{1}{2}g(t)^2\nabla_x\log p_t(x).
> $$
>
> Therefore the same marginal densities can be generated by the deterministic ODE
>
> $$
> \frac{dx_t}{dt}
> =
> f(x_t,t)
> -
> \frac{1}{2}g(t)^2\nabla_x\log p_t(x_t).
> $$
>
> Replacing the true score with
>
> $$
> s_\theta(x_t,t)
> \approx
> \nabla_x\log p_t(x_t),
> $$
>
> gives
>
> $$
> \boxed{
> \frac{dx_t}{dt}
> =
> f(x_t,t)
> -
> \frac{1}{2}g(t)^2s_\theta(x_t,t)
> }.
> $$

Putting in our initial $f(x_t, t$) expression, the corresponding probability flow ODE is $$ \frac{dx}{dt} = -\frac{1}{2}\beta(t)x_t - \frac{1}{2}\beta(t)s_\theta(x_t,t) $$ From here we can define a velocity field $$ v_t(x) = -\frac{1}{2}\beta(t) \left[ x+s_\theta(x,t) \right] $$

Notice

$$
\frac{dx}{dt}
=
\dot{x}_t
=
v_t(x_t)

$$
which is the general form for a continuous normalising flow.

> Note: The probability flow ODE wasnt obtained by just deleting the stochastic term from the reverse time SDE. Doing that would give us
>
> $$
> \frac{dx_t}{dt}
> =
> f(x_t,t)-g(t)^2\nabla_x\log p_t(x_t).
> $$
> 
> Instead, the derivation yields
> 
> $$
> \frac{dx_t}{dt}
> =
> f(x_t,t)
> -
> \frac{1}{2}g(t)^2\nabla_x\log p_t(x_t),
> $$
> 
> which gives a deterministic process with the same marginal densities $p_t$ as the diffusion SDE.

>For flow-matching, no score is learnt because a flow-matching model directly learns $v_\theta$ (the vector field).

---

## Back to DLMs
## Reverse Posterior to Language Models.

The exact reverse posterior was $q(z_s^{(i)}|z_t^{(i)},x_0^{(i)})$; but during generation during Language Model generation; $x_0$ is unknown.

A transformer predicts a probability distribution of $x_0$

$$
x_\theta^{(i)}(z^{1:S},t)\in\Delta^V
$$

$\theta$ = model parameters
$\Delta^V$ = probability simplex over the vocabulary
$x_\theta^{(i)}$ = predicted clean-token distribution at position $i$.
>Note :A probability simplex is the set of all valid probability vectors over the vocabulary.
>
>$$
>\Delta^V=\{p\in\mathbb R^V:p_v\ge0,\sum_{v=1}^{V}p=1\}
>$$
---

Previously, our clean token $x_0$ was represented by a one-hot vector e.g. $[0, 0, 1, 0, 0]$. But during generation our model prediction $x_\theta$ might have the form $[0.03,0.07,0.80,0.05,0.05]$ $\leftarrow$ (sums to 1).
>Note: A one-hot vector is located at the vertex of the probability. 

> A design choice is $\langle x_\theta, m \rangle = 0$ i.e. $x_\theta$ assigns zero probability to $[MASK]$. This is done by substituting the logit index corresponding to the [MASK] token with $-\infty$. This prevents the model from predicting $[MASK]$ as the final token.

$\therefore$ the  updated learned reverse transition for $z_t = m$ becomes

$$
p_\theta(z_s^{(i)}|z_t^{1:S}):=q(z_s^{(i)}|z_t^{(i)},x_0=x_\theta^{(i)})
\simeq Cat\left(
\frac{\alpha_s-\alpha_t}{1-\alpha_t}x_\theta^{(i)}(z_t^{1:S},t)
+\frac{1-\alpha_s}{1-\alpha_t}m
\right)
$$

For a masked position: $p_\theta(z_s^{(i)}=M|z_t)=\frac{1-\alpha_s}{1-\alpha_t}$

while for a vocabulary token $v\ne[MASK]$

$$
p_\theta(z_s^{(i)}=v|z_t)=\left(\frac{\alpha_s-\alpha_t}{1-\alpha_t}\right)x_\theta^{(i)}(v|z_t,t)
$$

Equivalently, we can describe the process as averaging the exact posterior over every possible clean token.

$$
p_\theta(z_s^{(i)}|z_t)=\sum_{x_0^{(i)}}q(z_s^{(i)}|z_t^{(i)},x_0^{(i)})p_\theta(x_0^{(i)}|z_t,t)
$$

The transformer converts the reverse posterior in a language model by supplying the missing distribution over $x_0$ where $p_\theta(x_0^{(i)}|z_t,t)$ is the transformer's predicted probability that the original token at position $i$ was $x_0^{(i)}$

### Example

let $p_\theta(x_0^{(i)}|z_t,t)=$

$$
\begin{cases}
0.7, & x_0^{(i)}=cat\\
0.2, & x_0^{(i)}=dog\\
0.1, & x_0^{(i)}=clean
\end{cases}
$$

if $z_t=[MASK]$

$$
z_s^{(i)}=\begin{cases}
x_0^{(i)}; & probability\ r\\
[MASK]; & probability\ 1-r
\end{cases}
$$

>$z_s^{i}$ represents the sample posterior.

where $r=\frac{\alpha_s-\alpha_t}{1-\alpha_t}$  

Suppose $r=0.4$

$$
P(z_s=cat)=0.4*0.7=0.28
$$

$$
P(z_s=dog)=0.4*0.2=0.08
$$

$$
P(z_s=clean)=0.4*0.1=0.04
$$

$$
P(z_s=[MASK])=0.6
$$

$p_\theta(z_s^{(i)}|z_t)=$

$$
\begin{cases}
0.28 & cat\\
0.08 & dog\\
0.04 & clean\\
0.6 & MASK
\end{cases}
$$

## Loss/Criterion Objective

> Note: in DDPM, $\beta_t$ denoted the per-step Gaussian noise variance. Here, $\bar\beta_t = 1-\alpha_t$ denotes the cumulative masking probability in discrete diffusion. For LLaDA's linear masking schedule (which we follow here), $\bar\beta_t=t$, so $\alpha_t=1-t$.
Without going in the details about ELBO, a good starting point is [yuanfanj's website](https://yunfanj.com/blog/2021/01/11/ELBO.html).

For discrete diffusion, the model is trained by minimising the NELBO. 

$$
\{L_{NELBO}=-ELBO\}
$$

$$
\max_\theta ELBO\ \Longleftrightarrow\ \min_\theta L_{NELBO}
$$

$$
L_{\mathrm{NELBO}}
=
\mathbb{E}_q\left[
\underbrace{
-\log p_\theta(x_0\mid z_0)
}_{\text{reconstruction term}}
+
\underbrace{
\sum_{k=1}^{K}
D_{\mathrm{KL}}\left(
q(z_{s_k}\mid z_{t_k},x_0)
\,\|\, 
p_\theta(z_{s_k}\mid z_{t_k})
\right)
}_{\text{KL divergence term}}
\right]
$$

where $k\rightarrow$ Number of discretized diffusion intervals.
Appendix section A 2.3 and A 2.4 in [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) goes more in depth. Im just picking necessary equations. For a masked diffusion, each KL term simplifies to a weighted token cross-entropy

$$
L_k=\sum_{k=1}^{K}E\left[\frac{\alpha_{s_k}-\alpha_{t_k}}{1-\alpha_{t_k}}\sum_{i=1}^{S}\mathbf 1[z_{t_k}^{(i)}=M]\left(-\log p_\theta(x_0^{(i)}|z_{t_k})\right)\right]
$$

taking the limit to infinity ($K \to -\infty$)

$$
L_\infty=E\left[\int_0^1\frac{-\alpha'_t}{1-\alpha_t}\sum_{i=1}^{S}\mathbf 1[z_t^{(i)}=m]\left(-\log p_\theta(x_0^{(i)}|z_t)\right)dt\right]
$$

since $\alpha_t$ is monotonically decreasing along $t$

$$
\alpha'_t<0
$$

$$
\alpha_t=(1-\bar\beta_t)
$$



$$
 \alpha'_t=\frac{d}{dt}(1-t)=-1
$$

$$
\alpha_s-\alpha_t
=
\alpha_{t-\Delta t}-\alpha_t
\overset{\text{1st-order Taylor}}{\approx}
\left(\alpha_t-\alpha_t'\Delta t\right)-\alpha_t
=
-\alpha_t'\Delta t
$$

$$
\frac{\alpha_s-\alpha_t}{1-\alpha_t}\approx\frac{-\alpha'_t}{1-\alpha_t}\Delta t\to\int_0^1\frac{-\alpha'_t}{1-\alpha_t}l(t)dt
$$

$$
\therefore\ \frac{-\alpha'_t}{1-\alpha_t}=\frac{1}{\bar{\beta}_t}
$$

. The paper gives it as $1/t$ because for LLaDA $\bar\beta_t=t$ .

From this formulation, we attain the LLaDA training objectives

$$
L(\theta):=-E_{t,x_0,z_t}\left[\frac{1}{\bar{\beta}_t}\sum_{i=1}^{S}\mathbf 1[z_t^{(i)}=m]\log p_\theta(x_0^{(i)}|z_t)\right]
$$

$1/\bar\beta$ becomes a loss weight term to prevent lightly corrupted timesteps from contributing almost no training signal.

For the linear masking schedule ( $\grave{a} \ la \ LLaDA$),

$$
\bar\beta_t=t,\qquad \bar\beta_t'=1
$$

so

$$
\frac{-\alpha_t'}{1-\alpha_t}
=
\frac{\bar\beta_t'}{\bar\beta_t}
=
\frac{1}{\bar\beta_t}
=
\frac{1}{t}.
$$

## Inference

Suppose a prompt has $C$ tokens and needs to generate $G$ response tokens, the total transformer sequence length is

$$
S=C+G
$$

before inference begin with

$$
z_1 = [\textit{PROMPT},\; \underbrace{[\textit{MASK}], [\textit{MASK}], \ldots, [\textit{MASK}]}_{\text{G response tokens}} ].
$$

$G$ could be selected, user specifies, predicted a max length could be allocated and the model predicts $[EOS]$ token then discards every other token after $[EOS]$ from step $t=t_k$ to $s=t_{k-1}$: a DLM predicts every masked position simultaneously.

The probability that a masked position becomes visible is

$$
u_{t\to s}=\frac{\alpha_s-\alpha_t}{1-\alpha_t}\approx r
$$

for $\alpha_t=1-t$

$$
u_{t\to s}=\frac{t-s}{t}\to1-\frac{s}{t}
$$

Fraction that remains masked is $s/t$

$$
{\LARGE\mathbf{\text{OverView of Sampling Process}}}
$$

$\rightarrow$ Predict all masked token. 
$\rightarrow$ Keep $1-s/t$ predictions
$\rightarrow$ Remask ~$s/t$
$\rightarrow$ Repeat until $t=0$

### Example 
```
Prompt: What year is this?
Response : [[MASK] [MASK] [MASK] [MASK] [MASK]]
```


1) Predict every masked token in parallel.

Say there are $M$ masked positions in our current sequence $z_t$. The transformer processes the entire sequence in a single forward pass and produces the logit vector for every masked position simultaneously.

For masked position $i$:

$$
\ell_i
=
(\ell_{i,1},\ell_{i,2},\ldots,\ell_{i,V})
\in\mathbb{R}^{V}
$$

where $V$ is the vocabulary size.

For all $M$ masked positions, these logits can be written as a matrix:

$$
L_t
=
\begin{bmatrix}
\ell_{1,1} & \ell_{1,2} & \cdots & \ell_{1,V}\\
\ell_{2,1} & \ell_{2,2} & \cdots & \ell_{2,V}\\
\vdots & \vdots & \ddots & \vdots\\
\ell_{M,1} & \ell_{M,2} & \cdots & \ell_{M,V}
\end{bmatrix}
\in\mathbb{R}^{M\times V}
$$


For example,

$$
[\text{MASK}], [\text{MASK}], [\text{MASK}], [\text{MASK}], [\text{MASK}]
$$

might produce

```
Response: This is the year 1887
```

2) Compute the confidence.
Softmax converts these logits to a categorical distribution over the vocabulary:

$$
p_\theta(x_i=v\mid z_t)
=
\frac{\exp(\ell_{i,v})}
{\sum_{u=1}^{V}\exp(\ell_{i,u})}
$$

The predicted token at position $i$ is the token with the highest probability:

$$
\hat{x}_i
=
\arg\max_{v\in V}
p_\theta(x_i=v\mid z_t)
$$

The confidence of this prediction is the probability assigned to the selected token:

$$
c_i
=
p_\theta(x_i=\hat{x}_i\mid z_t)
=
\max_{v\in V}p_\theta(x_i=v\mid z_t)
$$

For example:

| Prediction | Confidence $c_i$ |
|---|---:|
| This | $0.99$ |
| is | $0.98$ |
| the | $0.97$ |
| year | $0.94$ |
| 1887 | $0.31$ |

Since 1887 has the lowest confidence,

$$
c_{\text{1887}} = 0.31
$$

it is selected for remasking:

$$
\text{This is the year 1887}
\quad\rightarrow\quad
\text{This is the year [MASK]}
$$

3) Keep confident tokens and remask the lower confidence ones. LLaDA picks the lowest confidence predictions for remasking.

```
Response: This is the year [MASK]
```
> Note: Unlike our derivations above where positions are randomly selected for unmasking according to the reverse-transition probabilities. LLaDA uses a low confidence remasking strategy

4) Following a LLaDA style example. Repeat for $N$ denoising steps, until $t=0$. LLaDA uses a predefined generation length.

```
Response: This is the year 2026
```

## Parallel Inference and Memory

This blog by Sebastian Raschka [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) and the [jax-ml book](https://jax-ml.github.io/scaling-book/inference/) goes very in-depth into KV-Cache and its significance in inference. 

Let
$B\Rightarrow$ Batch size.
$N\Rightarrow$ Number of Transformer layer.
$d_h\Rightarrow$ Dimension of each attention head.
$d\Rightarrow$ Model hidden width
$d_{ff}$ $\Rightarrow$ Feed forward intermediate width.
$N\Rightarrow$ Number of transformer layer.
$h_{kv}\Rightarrow$ Number of key/value heads.
$b\Rightarrow$ Bytes per stored activation.
> Note: $b$ is dependent on datatype. $b$ is 4 for $FP32$, 2 for $FP16$ and 1 for $INT8$

$S\Rightarrow$ Current context length.
$h_q\Rightarrow$ No. of query head.

In Autoregressive KV cache

$$
\text{Memory}_{\,kv}^{AR}=2bBNS h_{kv}d_h
$$
>Note: The 2 accounts for keys and values.

$$
M_{kv}^{AR}=O(S).
$$

At each decoding iteration, the active query length is only $q=1$

Consequently, token activations are approximately

$$
M_{activation}^{AR}=O(Bqd_{ff})=O(Bd_{ff})
$$

and an unfused attention-score storage has size

$$
O(Bh_q q S)\simeq O(Bh_q S)
$$

> A fused kernel e.g. [FlashAttention, Tri Dao](https://arxiv.org/abs/2205.14135) avoids computing the full attention matrix in one chunk. Instead it processes it in tiles. But for simplicity sake, we'd stick to unfused attention.

## Vanilla diffusion decoding

In vanilla diffusion decoding every denoising iteration runs the transformer over the entire prompt-and-response $q=S$.

The current hidden state at position $j$ and layer $l$ is

$$
h_j^{(l,k)}=F_l(z_{t_k}^{1:S})_j
$$

 where F is the transformer computation and $k$ is the denoising step and $z$ is in the form:

$$
z_{t_k}^{(1:S)}=[z_1,z_2,...,z_S].
$$

some positions may be masked.

Then $F_l:z_{t_k}^{(1:S)}\to H^{(l,k)}$ where

$$
H^{(l,k)}=[h_1^{(l,k)},h_2^{(l,k)},...,h_S^{(l,k)}]^T\in\mathbb R^{S\times d}
$$

the key and value vectors are

$$
K_j^{(l,k)}=h_j^{(l-1,k)}W_k^{(l)}
$$

$$
V_j^{(l,k)}=h_j^{(l-1,k)}W_v^{(l)}
$$

Because attention here is bidirectional modifying one masked token can change the internal representation of every token,

$$
z_{t_{k+1}}\ne z_{t_k}\text{and}\ h_j^{(l,k+1)}\ne h_j^{(l,k)}
$$

for nearly all positions $j$ and this carries over to

$$
K_j^{(l,k+1)}\ne K_j^{(l,k)},\ V_j^{(l,k+1)}\ne V_j^{(l,k)}
$$

This is why AR-style KV caching doesn't directly apply to vanilla LLaDA. The LLaDA paper uses noncausal attention.

The approximate full sequence for activation memory

$$
M_{activation}^{dLLM}=bB[c_1Sd+c_2Sd_{ff}+c_3S(h_q+2h_{kv})d_h]+M_{attention}
$$

> $c_1d$  for residuals, $c_2d_{ff}$ for MLP, $c_3(h_q+2h_{kv})d_h$  for QKV

and the 

$$
M_{\mathrm{peak}}^{dLLM}(t)
\approx
M_{\mathrm{weights}}
+
\max\left\{
M_{\mathrm{FFN}},
M_{\mathrm{logits}}(t),
M_{\mathrm{attention}},
M_{\mathrm{activations}}
\right\}.
$$

At long sequence lengths, activation memory dominates completely.
Remember that one Auto-Regressive decoding step $q=1$ $\therefore$ $M_{activation}^{AR}\propto B(1)d$

But for diffusion models $M_{act}^{dLLM}\propto BSd$. So Diffusion Language Models has the advantage of parallelism over Auto-Regressive models but the price is full sequence computation.

> Note: AR decoding has a KV cache of size $O(S)$ which scales linearly with the context length, but it is a separate memory component.

## Parallel prediction and speed.

The approximate per layer cost for AR decoding step is

$$
C_{AR,step}=O(d^2+Sd)
$$

$d^2$  from projections and MLP. $Sd$ is from attending to cached context.

Generating $G$ tokens costs about

$$
C_{AR}=O\left(NGd^2+Nd\sum_{g=1}^{G}(C+g)\right)
$$

### Quick proof: Cost of generating $G$ tokens

Suppose the prompt has $C$ tokens. At generation step $g$, the model has generated approximately $g$ tokens, $\therefore$ the current context length is

$$
S_g \approx C+g.
$$

From the single-step autoregressive cost,

$$
C_{\mathrm{AR,step}}
=
O\left(Nd^2+NS_gd\right),
$$

the cost of generating token $g$ is

$$
C_{\mathrm{AR},g}
=
O\left(
Nd^2+Nd(C+g)
\right).
$$

To generate $G$ tokens, sum over all decoding steps:

$$
C_{\mathrm{AR}}
=
\sum_{g=1}^{G}
O\left(
Nd^2+Nd(C+g)
\right).
$$

---
But for dLLMs, a full pass costs:

$$
C_{dLLM,pass}=O(NSd^2+NS^2d)
$$

 then with $k$ denoising steps
 
$$
C_{dLLM}=O(kNSd^2+kNS^2d)
$$

let $r=G/k$ be the average number of tokens finalized per iteration. and lets consider only the projection and MLP term,

$$
\frac{C_{dLLM}}{C_{AR}}=\frac{kS}{G}=\frac{S}{r}
$$

We are processing $S$ tokens and unmasking $r$ position where $S>>r$. We see that predicting multiple tokens in parallel is not enough. The model might have to reduce the active activation size or unmask enough correct tokens per iteration to compensate for evaluation over the entire sequence. But DLMs can achieve pretty good wall-clock performance because full sequence matrix multiplication has higher GPU utilisation and [arithmetic intensity](https://jax-ml.github.io/scaling-book/transformers/)  than one-token AR-decoding

## Briefly addressing the problem

In the paper [Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing](https://arxiv.org/abs/2508.09192)

They introduced Discrete diffusion forcing partitions for that matrix blocks. Attention is causal between blocks and bidirectional within each block. Once a block is completed, the KV cache cant be changed by future blocks and can be cached. This also allows several blocks to be denoised in parallel.

$$
{\LARGE\mathbf{\text{Approximate Memory Profile}}}
$$

$$
M_{peak}^{block}=M_{weights}+M_{kv}(S_{completed})+M_{act}(A)
$$

where $A<<S$

This is like an AR-style reusable state + DLLM hybrid. In another paper, [Accelerating Diffusion LLMs via Adaptive Parallel Decoding](https://arxiv.org/abs/2506.00413) a left-to-right generation order is imposed and states outside a recomputation window of width $W$ are cached.

$$
M_{activation}^{APD}\approx O(BWd_{ff});\ W<<S
$$

> I didnt find a lot of papers that addressed the problem.  Likely because Diffusion Language Models are relatively new compared to Auto-regressive models.
# Conclusion

The entire process is pretty neat.

Clean Token $\Rightarrow$ Masking $\Rightarrow$ Masked Token Crossentropy $\Rightarrow$ Parallel Unmasking.

Much of the recent inference work is converging on hybrid designs: blockwise causal structure, partial KV reuse, dynamic token commitment etc.

Its going to be interesting to follow how this problem would be solved and maybe one day they can be a viable alternative to autoregressive models.

---

$$
{\Huge\mathbf{\text{RIP Dolly Parton}}}
$$

In Akure, Nigeria, on my way to my primary school, we used to have these Dolly Parton compilation CDs that I would listen to all the way through. I don't listen to a ton of country but once in a while I just run thru the Dolly Parton compilation albums. Link to AppleJack (one of my favourites:[LINK TO APPLEJACK](https://open.spotify.com/track/44H8pO6AOXOcsp3mikoTgq?si=f517b2575e2341b9))

