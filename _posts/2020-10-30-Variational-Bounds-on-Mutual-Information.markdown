---
title: "Variational Bounds on Mutual Information"
layout: post
date: 2020-10-30 13:35
image: /assets/images/markdown.jpg
headerImage: false
tag:
- Machine Learning
star: false
category: blog
author: MatthewWiesner
published: true
description: Review of [On Variational Bounds on Mutual Information](https://arxiv.org/pdf/1905.06922.pdf)
---

There has been incredible success using InfoNCE as a constrastive unsupervised pretraining objective for ASR in the past year. The objective function, originally presented in [Contrastive Predictive Coding (CPC)](https://arxiv.org/pdf/1807.03748.pdf), was described more theoretically in a subsequent [paper](https://arxiv.org/pdf/1905.06922.pdf).

Estimating the mutual information between two random variables is difficult. Training objectives that aim to maximize the mutual information between various quantities in sequence-to-sequence prediction tasks can be formulated by constructing lower bounds on the mutual information and maximizing them. Most of these bounds come from viewing the mutual information in terms of the KL-divergence

$$I\left(X; Y\right) = I\left(p\left(X\right); p\left(Y\right)\right) = D_{KL}\left(p\left(X, Y\right) || p\left(X\right)p\left(Y\right)\right)$$

## Upper Bound on Mutual Information
To upper bound the mutual information, we express the KL-divergence as an expectation, introduce a third distribution, $$q\left(Y\right)$$ and factor the joint distribution as $$p\left(X, Y\right) = p\left(Y|X\right)p\left(Y\right)$$.

$$\begin{align}
I\left(X; Y\right) &= \mathbb{E}_{p\left(X, Y\right)}\left[\log{\frac{p\left(Y|X\right)}{p\left(Y\right)}}\right] \\
&= \mathbb{E}_{p\left(X, Y\right)}\left[\log{\frac{p\left(Y|X\right)q\left(Y\right)}{p\left(Y\right)q\left(Y\right)}}\right] \\
&= \mathbb{E}_{p\left(X, Y\right)}\left[\log{\frac{p\left(Y|X\right)}{q\left(Y\right)}}\right] - D_{KL}\left(p\left(y\right) || q\left(y\right)\right) \\
&\leq \mathbb{E}_{p\left(X, Y\right)}\left[\log{\frac{p\left(Y|X\right)}{q\left(Y\right)}}\right] \mbox{Since KL-Divergence is non-negative} \\
&= \mathbb{E}_{p\left(X\right)}\left[D_{KL}\left(p\left(Y|X\right) || q\left(Y\right)\right)\right] \\
\end{align}$$

And now we have our first bound!

$$I\left(X; Y\right) \leq \mathbb{E}_{p\left(X\right)}\left[D_{KL}\left(p\left(Y|X\right) || q\left(Y\right)\right)\right] = R$$

This term can be thought of as a regularizer. Overfitting is reduced by forcing overconfident predictions to be smoothed with the prior distribution. 
$$R$$ stands for the rate of a model and is a limit on the information about the output $$Y$$ that is transmitted through the model from the input $$X$$. Similarly most regularizers can be interpretted as limiting the rate of model. An ideal regularizer limits the rate to the true mutual information between the random variables representing the inputs and desired outputs.

## Lower Bound on Mutual Information
To lower bound the mutual information, we factor in the opposite direction $$p\left(X, Y\right) = p\left(X | Y\right)p\left(Y\right)$$, introduce a third distribution $$q\left(X | Y\right)$$ and use the non-negativity of the KL-divergence as well as the differential entropy of a random variable to arrive at our lower bound. 

$$\begin{align}
I\left(X; Y\right) &= \mathbb{E}_{p\left(X, Y\right)}\left[\log{\frac{p\left(X|Y\right)q\left(X | Y\right)}{p\left(X\right)q\left(X|Y\right)}}\right] \\
&= \mathbb{E}_{p\left(X, Y\right)}\left[\log{\frac{q\left(X | Y\right)}{p\left(X\right)}}\right] + \mathbb{E}_{p\left(y\right)}\left[D_{KL}\left(p\left(X|Y\right) || q\left(X|Y\right)\right)\right] \\
&\geq \mathbb{E}_{p\left(X, Y\right)}\left[\log{\frac{q\left(X | Y\right)}{p\left(X\right)}}\right] \mbox{due to the non-negativity of KL-divergence} \\
&= \mathbb{E}_{p\left(X, Y\right)}\left[\log{q\left(X | Y\right)}\right] - \mathbb{E}_{p\left(X\right)}\left[\log{p\left(X\right)}\right] \\
&= \mathbb{E}_{p\left(X, Y\right)}\left[\log{q\left(X | Y\right)}\right] + h\left(X\right) \\
\end{align}$$

And now we have our second bound!

$$I\left(X; Y\right) \geq \mathbb{E}_{p\left(X, Y\right)}\left[q\left(X | Y\right)\right] + h\left(X\right) = I_{BA}$$

## Lower Bound on Mutual Information estimated with unormalized distributions

In general, computing normalized distributions as well as the differential entropy are intractable. For this reason it is important to find bounds of un-normalized distributions. By chosing a specific form for our un-normalized distribution and plugging it into the expresesion in our third-to-last step in our derivation for the lower bound on mutual information, we arrive at the following bound for un-normalized distributions.

Let 

$$q\left(X|Y\right) = \frac{p\left(X\right)e^{f\left(X, Y\right)}}{\mathbb{E}_{p\left(x\right)}\left[e^{f\left(X, Y\right)}\right]}$$

We then have that 
$$\begin{align}
\mathbb{E}_{p\left(X, Y\right)}\left[\log{\frac{q\left(X | Y\right)}{p\left(X\right)}}\right] &= \mathbb{E}_{p\left(X, Y\right)}\left[f\left(X, Y\right)\right] - \mathbb{E}_{p\left(Y\right)}\left[\log{\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]}\right]
\end{align}$$

So finally we have our third bound!!

$$I\left(X; Y\right) \geq \mathbb{E}_{p\left(X, Y\right)}\left[f\left(X, Y\right)\right] - \mathbb{E}_{p\left(Y\right)}\left[\log{\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]}\right] = I_{UBA}$$

## Donsker-Varadhan Bound on Mutual Information

The Donsker-Varadhan lower bound on mutual information is a well known bound that can be recovered by applying Jensen's Inequality on the expectation in the second term of the bound 

$$I\left(X; Y\right) \geq \mathbb{E}_{p\left(X, Y\right)}\left[f\left(X, Y\right)\right] - \mathbb{E}_{p\left(Y\right)}\left[\log{\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]}\right]$$


$$\begin{align}
\mathbb{E}_{p\left(Y\right)}\left[\log{\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]}\right] &\leq \log{\mathbb{E}_{p\left(Y\right)}\left[\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]\right]} \\
&\implies \mathbb{E}_{p\left(X, Y\right)}\left[f\left(X, Y\right)\right] - \mathbb{E}_{p\left(Y\right)}\left[\log{\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]}\right] \geq \mathbb{E}_{p\left(X, Y\right)}\left[f\left(X, Y\right)\right] - \log{\mathbb{E}_{p\left(Y\right)}\left[\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]\right]} \\
&\implies I\left(X; Y\right) \geq \mathbb{E}_{p\left(X, Y\right)}\left[f\left(X, Y\right)\right] - \log{\mathbb{E}_{p\left(Y\right)}\left[\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]\right]}
\end{align}$$

This is our fourth bound!!

$$I\left(X; Y\right) \geq \mathbb{E}_{p\left(X, Y\right)}\left[f\left(X, Y\right)\right] - \log{\mathbb{E}_{p\left(Y\right)}\left[\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]\right]} = I_{DV}$$

In Summary we have derived an upper bound and three lower bounds (or estimators) of Mutual Information. Their relationship is as follows.

$$R \geq I\left(X; Y\right) \geq I_{BA} \geq I_{UBA} \geq I_{DV}$$

## The [MINE](https://arxiv.org/pdf/1801.04062.pdf) Estimator for Mutual Information
MINE is an estimator for the mutual information parameterized by a neural network that is almost identical to InfoNCE. It uses the $$I_{DV}$$ bound from above and replaces the expectations with Monte-Carlo Estimates from minibatches of data.

<!--
Note when the minibatch consists of a single element,

$$\begin{align}\mathbb{E}_{p\left(Y\right)}\left[\log{\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]}\right] &= \mathbb{E}_{p\left(Y\right)}\left[\log{e^{f\left(X, Y\right)}}\right] \\
&= \mathbb{E}_{p\left(Y\right)}\left[f\left(X, Y\right)\right] \\
&= \mathbb{E}_{p\left(Y\right)}\left[\mathbb{E}_{p\left(X\right)}\left[f\left(X, Y\right)\right]\right]
\end{align}$$

In other words, 


The MINE objective can also be obtained by using Jensen's Inequatlity on the second term of the $$I_{UBA}$$ bound, but on the inner expectation over $$p\left(X\right)$$.

$$\begin{align}
\mathbb{E}_{p\left(Y\right)}\left[\log{\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]}\right] &\geq \mathbb{E}_{p\left(Y\right)}\left[\mathbb{E}_{p\left(X\right)}\left[f\left(X, Y\right)\right]\right] \
&\implies I_{UBA} \leq \mathbb{E}_{p\left(X, Y\right)}\left[f\left(X, Y\right)\right] - \mathbb{E}_{p\left(Y\right)}\left[\mathbb{E}_{p\left(X\right)}\left[f\left(X, Y\right)\right]\right] = I_{MINE}\\
\end{align}$$

So finally we have that 

$$begin{align}
I\left(X; Y\right) &\geq I_{UBA} \\
I_{MINE} &\geq I_{UBA} \\
I\left(X; Y\right) &\lessgtr I_{MINE} \\ 
\end{align}$$

-->


## Maximum Mutual Information and Pseudo-Labeling

In a [previous post](https://m-wiesner.github.io/LF-MMI) I gave a somewhat wrong explanation of why the MMI objective function actually did maximize the mutual information between random variables. Here is a better explanation for the particular case where we are working with an un-normalized neural estimator $$f\left(X, Y\right)$$.

$$\begin{align}
I\left(X; Y\right) &\geq I_{UBA} \\
&= \mathbb{E}_{p\left(X, Y\right)}\left[f\left(X, Y\right)\right] - \mathbb{E}_{p\left(X\right)}\left[\log{\mathbb{E}_{p\left(Y\right)}\left[e^{f\left(X, Y\right)}\right]}\right] \\
&= \mathbb{E}_{p\left(X, Y\right)}\left[\log{\frac{e^{f\left(x, y\right)}}{\mathbb{E}_{p\left(y\right)}e^{f\left(x, y\right)}}}\right] \\
\end{align}$$

This is exactly the MMI objective, where the $$\log{p\left(y\right)}$$ in the numerator gets removed. Since we do not optimize with respect to a fixed $$p\left(Y\right)$$ optimizing either objective is clearly the same as optimizing a lower-bound on the mutual information.

We can understand pseudo-labeling as factoring the expectation in the first term in terms of the posterior and marginal data likelihood. In this way you first sample unlabeled data, estimate a posterior distribution over output sequences, by producing a hypothesis lattice for instance, and then using this lattice for marginalization to compute the expectation over the posterior.

## Tractable Lower Bound

An easier to compute lower bound comes from the identity $$\log{x} \leq \frac{x}{a} + \log{a} + 1$$. Thereforethe

$$\begin{align}
\log{\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]} &\leq \frac{\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]}{a\left(Y\right)} + \log{a\left(Y\right)} + 1 \\
&\implies I_{UBA} \geq \mathbb{E}_{p\left(X, Y\right)}\left[f\left(X, Y\right)\right] - \mathbb{E}_{p\left(Y\right)}\left[\frac{\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]}{a\left(Y\right)} + \log{a\left(Y\right)} + 1\right] = I_{TUBA} \\
\end{align}$$ 

Since this relationship holds true for all values of $$a\left(Y\right)$$, we can set $$a\left(Y\right) = e \ \forall \ Y$$ at the expense of having a slightly looser bound on the mutual information. This gives us the bound 

$$\begin{align}
I\left(X, Y\right) &\geq I_{TUBA} \\
&\geq \mathbb{E}_{p\left(X, Y\right)} \left[ f\left(X, Y\right) \right] - \mathbb{E}_{p\left(Y\right)} \left[e^{-1} \mathbb{E}_{p\left(X\right)} \left[e^{f\left(X, Y\right)}\right]\right] \\
&= I_{NJW} \\
\end{align}$$
<!--
&\geq \mathbb{E}_{p\left(X, Y\right)}\left[f\left(X, Y\right)\right] - \mathbb{E}_{p\left(Y\right)}\left[e^{-1}\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)\right]\right] \\
&= I_{NJW} \\
\end{align}$$
-->

This is the bound used in the f-MINE variant of the MINE objective. When $$a\left(y\right)$$ is estimated by an exponential moving average, this corresponds to the heuristic used in MINE to reduce the bias of the MINE gradient. 

To summarize all of the bounds we've seen again

$$R \geq I\left(X; Y\right) \geq I_{BA} \geq I_{UBA} = F_{MMI} - \mathbb{E}_{p\left(X\right)}\left[\log{p\left(Y\right)}\right] \geq I_{TUBA} \geq I_{NJW}$$

## InfoNCE

The main insight of InfoNCE is that we can use independent samples from some other distribution to decrease the variance of our estimate of the mutual information.

$$I\left(X, Z; Y\right) = \mathbb{E}_{p\left(Z\right)}\left[I\left(X; Y\right)\right] = I\left(X, Y\right)$$

In InfoNCE the RV $$Z=\{X_2^{\prime} \ldots X_N^{\prime}\}$$ are $$N-1$$ samples from some other distribution over $$X$$, which we treat as negative examples of $$X$$. Often the neural network is called the critic in this literature as it is tasked with compared inputs $$X$$ to outputs $$Y$$. Setting the critic to 

$$\begin{align}
f^{\prime}\left(X, Y\right) &= 1 + \log{\frac{e^{f\left(X,  Y\right)}}{a\left(Y; X, Z\right)}} \\
&\implies I_{TUBA} = 1 + \mathbb{E}_{p\left(X,Y\right)}\left[\log{\frac{e^{f\left(X, Y\right)}}{a\left(Y; X, Z\right)}}\right] - \mathbb{E}_{p\left(Y\right)p\left(X\right)}\left[\frac{e^{f\left(X, Y\right)}}{a\left(Y; X, Z\right)}\right] \\
&=  1 + \mathbb{E}_{p\left(X,Y\right)p\left(Z\right)}\left[\log{\frac{e^{f\left(X, Y\right)}}{a\left(Y; X, Z\right)}}\right] - \mathbb{E}_{p\left(Y\right)p\left(X\right)p\left(Z\right)}\left[\frac{e^{f\left(X, Y\right)}}{a\left(Y; X, Z\right)}\right]
\end{align}$$

Note that the optimal critic for $$I_{UBA}$$ is 
$$f\left(X, Y\right) = 1 + \log{\frac{p\left(Y|X\right)}{p\left(Y\right)}}$$

So we are simply replacing 
$$\frac{p\left(Y|X\right)}{p\left(Y\right)} \to \frac{e^{f\left(X, Y\right)}}{a\left(Y\right)}$$

which are learned parameters and if trained to convergence we should recover the optimal critic. $$a\left(Y\right)$$ should optimally be the partition function for $$Y$$.

There are two final steps to get the InfoNCE objective. The inner two expectations of the last term of the above expression for $$I_{TUBA}$$ can be rewritten as 

$$\begin{align}
\mathbb{E}_{p\left(Y\right)p\left(X\right)p\left(Z\right)}\left[\frac{e^{f\left(X, Y\right)}}{a\left(Y; X, Z\right)}\right] &= \mathbb{E}_{p\left(Y\right)}\left[\frac{1}{K}\sum_{i=1}^K \mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[\frac{e^{f\left(X, Y\right)}}{a\left(Y; X, Z\right)}\right]\right] \\
\end{align}$$

<!--&= \mathbb{E}_{p\left(Y\right)}\left[\mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[\frac{1}{K}\sum_{i=1}^K \frac{e^{f\left(X, Y\right)}{a\left(Y; X, Z\right)}\right]\right] \\
-->

In other words, we can just rewrite the expectation in terms $$K$$ replicas of the expectation. How do we get $$K$$ replicas of the data? Since $$Z$$ are other examples inputs examples drawn *independently*, each example can also be considered as a draw from $$X$$. So we can simply swap one of the $$K-1$$ examples in $$Z$$ with the original example from $$p\left(X\right)$$. Since there are $$K-1$$ examples in $$Z$$ we can repeat this swapping procedure $$K-1$$ times, which in addition to using the original value $$X$$ drawn from $$p\left(X\right)$$ gives us $$K$$ replicas. In expectation this sum will be the same as the sum of the expectations.

The second step is to use $$Z$$ to form a Monte-Carlo approximation for $$a\left(Y; X, Z\right)$$, which can also be viewed as approximating the partition function.

$$a\left(Y; X, Z\right) = \frac{1}{K}\left(e^{f\left(X, Y\right)} + \sum_{i=2}^{K} e^{f\left(Z_i, Y\right)}\right)$$. 

Therefore ... 

$$\begin{align}
\mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[\frac{1}{K} \sum_{i=1}^K \frac{e^{f\left(X, Y\right)}}{a\left(Y; X, Z\right)\right] &=
\mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[\frac{1}{K}\left(e^{f\left(X, Y\right)} + \sum_{i=2}^K \frac{e^{f\left(Z_i, Y\right)}}{a\left(Y; X, Z\right)}\right)\right] \\
&= \mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[\frac{1}{K}\left(e^{f\left(X, Y\right)} + \sum_{i=2}^K \frac{e^{f\left(Z_i, Y\right)}\right)}{\frac{1}{K}\left(e^{f\left(X, Y\right)} + \sum_{i=2}^K \frac{e^{f\left(Z_i, Y\right)}\right)}\right] \\
&= \mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[ 1 \right] \\
&= 1 \\
&\implies I_{TUBA} = I_{NCE} = \mathbb{E}_{p\left(X, Y\right)p\left(Z\right)}\left[\log{\frac{e^{f\left(X, Y\right)}}{a\left(Y; X, Z\right)}}\right]
\end{align}$$

Since the second term in the bound is now a constant $$1$$ it cancels with the 1 in $$I_{TUBA}$$ and only the first expectation remains.

<!-- $$\begin{align}
I_{NCE} &= \mathbb{E}_{p\left(Y\right)}\left[\frac{1}{K}\sum_{k=1}^K \log{\frac{e^{f\left(X_k, Y\right)}}{\frac{1}{J}\sum_{j=1}^J e^{f\left(X_j, Y\right)}}}\right] \\
&= 
\end{align}$$

In the original CPC paper, the outer expectation was omitted because the join expectation was factored as 
$$p\left(X, Y\right) = p\left(Y | X) p\left(X\right) = p\left(X\right)$$ 

since the latent variable $$Y$$ was a deterministic function of $$X$$.
-->
