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
author: Matthew Wiesner
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

$$R \geq I\left(X; Y\right) \geq I_{BA} \geq I_{UBA} = F_{MMI} - \mathbb{E}_{p\left(Y\right)}\left[\log{p\left(Y\right)}\right] = F_{MMI} + H\left(Y\right) \geq I_{TUBA} \geq I_{NJW}$$

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

In other words, we can just rewrite the expectation in terms $$K$$ replicas of the expectation. How do we get $$K$$ replicas of the data? Since $$Z$$ are other input examples drawn *independently*, each example can also be considered as a draw from $$X$$. So we can simply swap one of the $$K-1$$ examples in $$Z$$ with the original example from $$p\left(X\right)$$. Since there are $$K-1$$ examples in $$Z$$ we can repeat this swapping procedure $$K-1$$ times, which in addition to using the original value $$X$$ drawn from $$p\left(X\right)$$ gives us $$K$$ replicas. In expectation this sum will be the same as the sum of the expectations.

The second step is to use $$Z$$ to form a Monte-Carlo approximation for $$a\left(Y; X, Z\right)$$, which can also be viewed as approximating the partition function.

$$a\left(Y; X, Z\right) = \frac{1}{K}\left(e^{f\left(X, Y\right)} + \sum_{i=2}^{K} e^{f\left(Z_i, Y\right)}\right)$$. 

Therefore ... 

$$\begin{align}
\mathbb{E}_{p\left(Y\right)}\left[\frac{1}{K}\sum_{i=1}^K \mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[\frac{e^{f\left(X, Y\right)}}{a\left(Y; X, Z\right)}\right]\right] &= \mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[\frac{1}{K} \sum_{i=1}^K \frac{e^{f\left(X, Y\right)}}{a\left(Y; X, Z\right)}\right] \\
&= \mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[\frac{\frac{1}{K}\left(e^{f\left(X, Y\right)} + \sum_{i=2}^K e^{f\left(Z_i, Y\right)}\right)}{a\left(Y; X, Z\right)}\right] \\
&= \mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[\frac{\frac{1}{K}\left(e^{f\left(X, Y\right)} + \sum_{i=2}^K e^{f\left(Z_i, Y\right)}\right)}{\frac{1}{K}\left(e^{f\left(X, Y\right)} + \sum_{i=2}^{K} e^{f\left(Z_i, Y\right)}\right)}\right] \\
&= \mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[ 1 \right] \\
&\implies I_{TUBA} = I_{NCE} = \mathbb{E}_{p\left(X, Y\right)p\left(Z\right)}\left[\log{\frac{e^{f\left(X, Y\right)}}{\frac{1}{K}\left(e^{f\left(X, Y\right)} + \sum_{i=2}^{K} e^{f\left(Z_i, Y\right)}\right)}}\right] \\
\end{align}$$

Since the second term in the bound is now a constant $$1$$ it cancels with the 1 in $$I_{TUBA}$$ and only the first expectation remains.

Also note that this lower bound is itself upper bounded by $$\log{K}$$.

$$ I_{NCE} = \log{K} + \mathbb{E}_{p\left(X, Y\right)p\left(Z\right)}\left[\log{\frac{e^{f\left(X, Y\right)}}{e^{f\left(X, Y\right)} + \sum_{i=2}^{K} e^{f\left(Z_i, Y\right)}}}\right] $$

And $$\mathbb{E}_{p\left(X, Y\right)p\left(Z\right)}\left[\log{\frac{e^{f\left(X, Y\right)}}{e^{f\left(X, Y\right)} + \sum_{i=2}^{K} e^{f\left(Z_i, Y\right)}}}\right]$$ is guaranteed to be negative, since the denominator is a sum of non-negative values that includes the numerator. Therefore, the smallest value this can take is $$0$$ leaving ...

$$I_{NCE} \leq \log{K}$$.

Therefore, if $$I\left(X; Y\right) \geq \log{K}$$ this estimator will drastically underestimate the mutual information. It also shows that it is critical to use a large number of negative samples to accurately estimate the mutual information. 

Approximating the expectation over $$p\left(X, Y\right)p\left(Z\right)$$ can be handled in many ways. In the original CPC paper, $$X$$ and $$Y$$ are particular values called $$ z_{t+k}, c_t$$ which correspond to learned latent, local encodings of speech frames, and a global context vector learned over these encodings. The are both deterministic functions of the *same* input $$X=\{x_1, x_2, \ldots, x_N \}$$. The expecation is then approximated with the Monte-Carlo estimate using the neural network outputs corresponding to a minibatch of inputs.

## Alternative Factorizations of the Expecation
We could factor the expectation in multiple ways. Let 

$$\begin{align}
I_{NCE} &= \mathbb{E}_{p\left(X, Y\right)p\left(Z\right)}\left[\log{\frac{e^{f\left(X, Y\right)}}{\frac{1}{K}\left(e^{f\left(X, Y\right)} + \sum_{i=2}^{K} e^{f\left(Z_i, Y\right)}\right)}}\right] \\
&= \mathbb{E}_{p\left(X, Y\right)p\left(Z\right)}\left[L_{NCE}\right] \\
&= \mathbb{E}_{p\left(Z\right)p\left(X\right)} \left[\sum_Y p\left(Y|X\right) L_{NCE}\right] \\
&= \mathbb{E}_{p\left(Z\right)p\left(X\right)} \left[\sum_Y p\left(Y|X\right) \log{\frac{e^{f\left(X, Y\right)}}{\frac{1}{K}\left(e^{f\left(X, Y\right)} + \sum_{i=2}^{K} e^{f\left(Z_i, Y\right)}\right)}} \right] \\
&= \mathbb{E}_{p\left(Z\right)p\left(X\right)} \left[\sum_Y p\left(Y|X\right) \left(f\left(X, Y\right) - \log{\frac{1}{K}\left(e^{f\left(X, Y\right)} + \sum_{i=2}^{K} e^{f\left(Z_i, Y\right)} \right)} \right) \right] \\
&= \mathbb{E}_{p\left(Z\right)p\left(X\right)} \left[\sum_Y p\left(Y|X\right) \left( f\left(X, Y\right) - f^{*} \right)\right] \\
&= \mathbb{E}_{p\left(Z\right)p\left(X\right)} \left[\sum_Y \frac{p\left(Y\right)e^{f\left(X, Y\right)}}{\mathbb{E}_{p\left(Y\right)}\left[e^{f\left(X, Y\right)}\right]} \left( f\left(X, Y\right) - f^{*} \right)\right] \\
&= \mathbb{E}_{p\left(Z\right)p\left(X\right)} \left[\frac{\sum_Y p\left(Y\right)e^{f\left(X, Y\right)}\left( f\left(X, Y\right) - f^{*} \right)}{\mathbb{E}_{p\left(Y\right)}\left[e^{f\left(X, Y\right)}\right]}\right] \\
\end{align}$$

What if we did have labeled data and did not have to marginalize over all possible outputs $$Y$$? Then the above equation becomes 

$$\begin{align}
I_{NCE} &= \mathbb{E}_{p\left(Z\right)p\left(X\right)} \left[\frac{p\left(Y\right)e^{f\left(X, Y\right)}\left( f\left(X, Y\right) - f^{*} \right)}{\mathbb{E}_{p\left(Y\right)}\left[e^{f\left(X, Y\right)}\right]}\right]
\end{align}$$

This is exactly the MMI objective scaled by the term $$\left( f\left(X, Y\right) - f^{*} \right)$$. We therefore see that under this objective function, a good $$f\left(X, Y\right)$$ is one that learns to discriminate between the correct *output* and competing outputs, as well ensuring that different *inputs* result in different outputs.

## Tractable Alternative Factorization
The above alternative factorization is unfortunately completely intractable. To solve this we can bound it one more time using Jensen's Inequality. Also we will assume that the $$K-1$$ draws from $$p\left(Z\right)$$ will be the $$K-1$$ other examples in a minibatch. We call the whole minibatch 

$$X = \{X, Z_2, \ldots, Z_k \} = \{X_1, X_2, \ldots, X_k\}$$ 

$$\begin{align}
  \sum_Y p\left(Y | X \right) \left(f\left(X, Y\right) - f^{*}\right) &= \sum_Y p\left(Y | X \right) f\left(X, Y\right) - \sum_Y p\left(Y | X\right) \log{\sum_{i=1}^K e^{f\left(X_i, Y\right)}} \\
  &\geq \sum_Y p\left(Y | X \right) f\left(X, Y\right) - \log{\sum_Y p\left(Y | X\right) \sum_{i=1}^K e^{f\left(X_i, Y\right)}} \\
  &= \sum_Y p\left(Y | X \right) f\left(X, Y\right) - \log{\sum_{i=1}^K \sum_Y p\left(Y | X\right) e^{f\left(X_i, Y\right)}} \\
\end{align}$$

So finally we have 

$$\begin{align}
I_{NCE} = \mathbb{E}_{p\left(Z\right)p\left(X\right)} \left[ \sum_Y p\left(Y | X \right) f\left(X, Y\right) - \log{\sum_{i=1}^K \sum_Y p\left(Y | X\right) e^{f\left(X_i, Y\right)}} \right] &= \mathbb{E}_{p\left(X\right)} \left[ \sum_Y p\left(Y | X \right) f\left(X, Y\right) \right] - \mathbb{E}_{p\left(Z\right)p\left(X\right)}\left[\log{\sum_{i=1}^K \sum_Y p\left(Y | X\right) e^{f\left(X_i, Y\right)}}\right] \\
\end{align}$$
<!--
&\approx \frac{1}{K} \sum_{i=1}^K \sum_Y p\left(Y | X_i \right) f\left(X_i, Y\right) - \frac{1}{K} \sum_{k=1}^K \log{\sum_{i=1}^K \sum_Y p\left(Y | X_k\right) e^{f\left(X_i, Y\right)}} \\
-->

## Gradient of the alternative factorization

The above objective leaves us with a catch-22. We are trying to estimate a posterior distribution, but doing so requires an estimate for it. One potential solution is to hold fixed the posterior distribution when updating the model with unlabeled data. Also assume that our expectations are over minibatches $$\mathcal{B} = \{X_1, \ldots, X_K\}$$ and $$X$$ is simply the first element in the minibatch $$X_1$$. We also will assume a neural network $$\phi$$ parameterized $$f\left(\cdot\right)$$ and inputs are of lenght $$T$$.

$$f\left(X, Y\right) = \sum_{t=1}^T \phi\left(X\right)_{Y_t}^t$$ 

and that marginalization over output sequences is handled by representing the space of sequences with a WFST $$G$$. We denote the forward and backward scores at state $$s$$ and time $$\tau$$ over this graph respectively as 

$$\alpha\left(s, \tau\right), \beta\left(x, \tau\right)$$

In this case the gradient becomes ...

$$\begin{align}
\frac{\partial I_{NCE}}{\partial y_s^{\tau}\left(j\right)} &= \mathbb{E}_{\mathcal{B}}\left[\sum_{Y} p\left(Y | X_1\right) \mathbb{1}\left(Y_{\tau}, s\right) \mathbb{1}\left(j, 1\right) - \frac{\sum_Y p\left(Y | X_1\right) e^{f\left(X_j, Y\right)} \mathbb{1}\left(Y_{\tau}, s\right)}{\sum_{i=1}^K \sum_Y p\left(Y|X_1\right)e^{f\left(X_i, Y\right)}}\right] \\
&= \mathbb{E}_{\mathcal{B}}\left[ \gamma_{X_1}\left(s, \tau\right)\mathbb{1}\left(j, 1\right) -  \frac{\alpha_{X_{1,j}}\left(s, \tau\right)\beta_{X_{1, j}}\left(s, \tau\right)}{\sum_{i=1}^K \sum_{\sigma} \alpha_{X_{1, i}}\left(\sigma, \tau\right)\beta_{X_{1,i}}\left(\sigma, \tau\right)} \right]\\
\end{align}$$

Since in a single minibatch we have $$K$$ examples of speech, and $$K-1$$ negative samples, we can use a single example to generate $$K$$ unique minibatches where instead of using $$X_1$$ as $$X$$ we use $$X_i$$.

The foward score through the lattice generated by inputs $$X_k, X_j$$ is 

$$E(k, j) = [\![\left(\phi\left(X_k\right) + \phi\left(X_j\right)\right) \circ G]\!]$$

The loss function for the minibatch of data then becomes.

$$\begin{align}
\frac{\partial I_{NCE}}{\partial y_s^{\tau}\left(j\right)} &= \frac{1}{K} \sum_{k=1}^K \gamma_{X_k}\left(s, \tau\right)\mathbb{1}\left(j, k\right) -  \frac{\alpha_{X_{k,j}}\left(s, \tau\right)\beta_{X_{k, j}}\left(s, \tau\right)}{\sum_{i=1}^K \sum_{\sigma} \alpha_{X_{k, i}}\left(\sigma, \tau\right)\beta_{X_{k,i}}\left(\sigma, \tau\right)}\\
&= \frac{1}{K} \left[\gamma_{X_j}\left(s, \tau\right) - \sum_{k=1}^K \frac{\alpha_{X_{k,j}}\left(s, \tau\right)\beta_{X_{k, j}}\left(s, \tau\right)}{\sum_{i=1}^K \sum_{\sigma} \alpha_{X_{k, i}}\left(\sigma, \tau\right)\beta_{X_{k,i}}\left(\sigma, \tau\right)} \right]\\
&= \frac{1}{K} \left[\gamma_{X_j}\left(s, \tau\right) - \sum_{k=1}^K \gamma_{X_{k, j}}\left(s, \tau\right) \frac{e^{E\left(k, j\right)}}{\sum_{i=1}^K e^{E\left(k, i\right)}} \right] \\
\end{align}$$


## Updating p(Y | X)

The problem with the above update is that as mentioned before, the optimal critic is

$$ f\left(X, Y\right) = \log{\frac{p\left(Y | X \right)}{p\left(Y\right)}}$$

When we evaluate over the wrong distribution
$$q\left(Y | X\right) \neq p\left(Y | X\right)$$
then we are only able to train the network to perform as well as the original posterior we supplied. To increase the mutual information, we have to also be able to update our model for the posterior distribution. Unfortunately, we run into some computation that as far as I can tell is intractable. Notably

$$ \sum_Y p\left(Y | X\right) f\left(X, Y\right) $$

is intractable because of the form of $$f\left(X, Y\right)$$. If this were a simple classification task, then we could probably evaluate this quantity, however, in sequence tasks, evaluating $$f\left(X, Y\right)$$ of the sequence $$Y$$, for all possible values is not feasible. Nonetheless we take the gradient of this term holding fixed 

$$f\left(X, Y\right)$$ 

this time. For the purpose of taking the gradient, I will actually use a specific functional form for 

$$p\left(Y | X\right) = \frac{p\left(Y\right)e^{f\left(X, Y\right)}}{\mathbb{E}_{p\left(Y\right)}\left[e^{f\left(X, Y\right)}\right]}$$ 

$$\begin{align}
\frac{\partial p\left(Y | X \right)}{\partial y_s^{\tau}\left(j\right)} &= \frac{\partial}{\partial y_s^{\tau}\left(j\right)} p\left(Y\right) e^{f\left(X, Y\right)}\left(\sum_Y p\left(Y\right)e^{f\left(X, Y\right)}\right)^{-1} \\
&= p\left(Y\right)e^{f\left(X, Y\right)} \frac{\partial}{\partial y_s^{\tau}\left(j\right)} f\left(X, Y\right)\left(\sum_Y p\left(y\right)e^{f\left(X, Y\right)}\right)^{-1} - p\left(Y\right)e^{f\left(X, Y\right)} \frac{\sum_Y p\left(y\right)e^{f\left(X, Y\right)}\frac{\partial  f\left(X, Y\right)}{\partial y_s^{\tau}\left(j\right)}}{\left(\sum_Y p\left(y\right)e^{f\left(X, Y\right)}\right)^2} \\
&= p\left(Y | X\right) \left(\mathbb{1}\left(Y_{\tau}, s\right) - \gamma_{X}\left(s, \tau \right)\right) \\
&\implies \sum_Y \frac{\partial p\left(Y | X \right)}{\partial y_s^{\tau}\left(j\right)} f\left(X, Y\right) = \mathbb{1}\left(j, 1\right)\sum_Y p\left(Y | X\right) f\left(X, Y\right)\left(\mathbb{1}\left(Y_{\tau}, s\right) - \gamma_{X}\left(s, \tau \right)\right) \\
\end{align}$$

Having worked out the gradient of the posterior we can easily get the gradient of the second term in the objective function.

$$\begin{align}
\frac{\partial}{\partial y_s^{\tau}\left(j\right)} \mathbb{E}_{\mathcal{B}}\left[\log{\sum_{i=1}^K \sum_Y p\left(Y | X_1\right) e^{f\left(X_i, Y\right)}} \right] &= \mathbb{E}_{\mathcal{B}}\left[\frac{1}{\sum_{i=1}^K \sum_Y p\left(Y | X_1\right) e^{f\left(X_i, Y\right)}} \sum_Y p\left(Y | X_1\right) e^{f\left(X_1, Y\right)}\left(\mathbb{1}\left(Y_{\tau}, s\right) - \gamma_{X_1}\left(s, \tau\right)\right)\right] \\
&= \mathbb{E}_{\mathcal{B}}\left[\frac{\alpha_{1, 1}\left(s, \tau\right)\beta_{1, 1}\left(s, \tau\right) - \gamma_{X_1}\left(s, \tau\right)\sum_{\sigma}\alpha_{1, 1}\left(s, \tau\right)\beta_{1,1}\left(s, \tau\right)}{\sum_{i=1}^K \sum_{\sigma} \alpha_{1, i}\left(\sigma, \tau\right)\beta_{1, i}\left(\sigma, \tau\right)}\right] \\
&= \mathbb{E}_{\mathcal{B}}\left[ \left(\gamma_{X_{1, 1}}\left(s, \tau\right) - \gamma_{X_1}\left(s, \tau\right)\right) \frac{e^{E\left(1, 1\right)}}{\sum_{i=1}^K e^{E\left(1, i\right)}}\right]
\end{align}$$

<!--
Putting the gradients together we get 
$$\begin{align}
\mathbb{1}\left(j, 1\right)\sum_Y p\left(Y | X\right) \sum_Y p\left(Y | X_1\right) f\left(X_1, Y\right)\left(\mathbb{1}\left(Y_{\tau}, s\right) - \gamma_{X}\left(s, \tau \right)\right) + \gamma_{X_1}\left(s, \tau\right)\left(1 + \frac{e^{E\left(1, j\right)}}{\sum_{i=1}^K e^{E\left(1, i\right)}}\right) -  \gamma_{X_{1, j}}\left(s, \tau\right) \frac{2e^{E\left(1, j\right)}}{\sum_{i=1}^K e^{E\left(1, i\right)}}
\end{align}$$
-->

You can interpret this as an acoustic confidence of an input weighted by how distinguishable it is on average (from a sample of $$K$$ other examples).
Now we have to deal with the intractable(?) first term.

$$\begin{align}
\sum_Y p\left(Y | X_1\right) f\left(X_1, Y\right) &= \frac{1}{Z\left(X_1\right)} \sum_Y p\left(Y\right) e^{\sum_t \phi\left(X_1\right)_{Y_t}^t} \sum_{t^{\prime}} \phi\left(X_1\right)_{Y_{t^{\prime}}}^{t^{\prime}} \\
&= \frac{1}{Z\left(X_1\right)} \sum_Y p\left(Y\right) \sum_{t^{\prime}} \phi\left(X_1\right)_{Y_{t^{\prime}}}^{t^{\prime}} e^{\sum_t \phi\left(X_1\right)_{Y_t}^t} \\
&= \frac{1}{Z\left(X_1\right)} \sum_{t^{\prime}} \sum_Y \phi\left(X_1\right)_{Y_{t^{\prime}}}^{t^{\prime}}  p\left(Y\right) e^{\sum_t \phi\left(X_1\right)_{Y_t}^t} \\
&\simeq \frac{T}{N Z\left(X_1\right)} \sum_{i=1}^N \sum_Y \phi\left(X_1\right)_{Y_{t_i}}^{t_i} p\left(Y\right) e^{\sum_t \phi\left(X_1\right)_{Y_t}^t} \\
&= T \frac{Z\left(X_1\right)}{NZ\left(X_1\right)} \sum_{i=1}^N \sum_{\sigma}\gamma_{X_1}\left(\sigma, t_i\right) \phi\left(X_1\right)^{t_i}\\
&=\frac{T}{N} \sum_{i=1}^N \sum_{\sigma}\gamma_{X_1}\left(\sigma, t_i\right) \phi\left(X_1\right)^{t_i} \\
&= T \hat{Z}\left(X_1 \right)\\
&\implies \sum_Y p\left(Y | X_1\right) f\left(X_1, Y\right) \left(\mathbb{1}\left(Y_{\tau}, s\right) - \gamma_{X_1}\left(s, \tau\right) \right) \simeq \sum_Y p\left(Y | X_1\right) f\left(X_1, Y\right) \mathbb{1}\left(Y_{\tau}, s\right) - \gamma_{X_1}\left(s, \tau \right) T\hat{Z}\left(X_1\right) \\
\end{align}$$


## Alternative Approximation
This is clearly not necessarily a lower bound any more, but it makes computation easy. I just used Jensen's Inequality on the numerator and denominator separately after splitting the log fraction up. The second term is lower than it should be, but the first term is greater than it should be.

$$\begin{align}
I_{NCE} &= \mathbb{E}_{p\left(X, Y\right)p\left(Z\right)}\left[\log{\frac{e^{f\left(X, Y\right)}}{\sum_{i=1}^{K} e^{f\left(X_i, Y\right)}}}\right] \\
&= \mathbb{E}_{p\left(X, Y\right)p\left(Z\right)}\left[\sum_Y p\left(Y | X\right)\log{\frac{e^{f\left(X, Y\right)}}{\sum_{i=1}^{K} e^{f\left(X_i, Y\right)}}}\right] \\
&\simeq \mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[\log{\sum_Y p\left(Y | X\right)e^{f\left(X, Y\right)}} - \log{\sum_{i=1}^{K} \sum_Y p\left(Y | X\right)e^{f\left(X_i, Y\right)}}\right] \\
&= \mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[\log{\frac{\sum_Y p\left(Y | X\right)e^{f\left(X, Y\right)}}{\sum_{i=1}^{K} \sum_Y p\left(Y | X\right)e^{f\left(X_i, Y\right)}}}\right] \\
&= \mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[\log{\frac{\sum_Y p\left(Y | X\right)e^{f\left(X, Y\right)}}{\sum_{i=1}^{K} \sum_Y p\left(Y | X\right)e^{f\left(X_i, Y\right)}}}\right]  \\
&= \mathbb{E}_{p\left(X\right)p\left(Z\right)}\left[[\![\left(\phi_X + \phi_X\right) \circ G]\!] - \mbox{logsumexp}_i\left([\![\left(\phi_X + \phi_{X_i}\right) \circ G]\!]\right )\right] \\
\end{align}$$



<!--
$$\begin{align}
&\implies \frac{\partial}{\partial y_s^{\tau}\left(j\right)} I_{MCE} \simeq \mathbb{E}_{\mathcal{B}}\left[ \mathbb{1}\left(1, j\right)\left(\left(\hat{\gamma}_{X_1}\left(s, \tau\right) - \gamma_{X_1}\left(s, \tau \right)\right) \frac{T\hat{Z}\left(X_1\right)}{Z\left(X_1\right)}\right) - \left(\gamma_{X_{1, 1}}\left(s, \tau\right) - \gamma_{X_1}\left(s, \tau\right)\right) \frac{e^{E\left(1, 1\right)}}{\sum_{i=1}^K e^{E\left(1, i\right)}} + \mathbb{1}\left(1, j\right)\gamma_{X_1}\left(s, \tau\right) - \gamma_{X_{1, j}}\left(s, \tau\right) \frac{e^{E\left(1, j\right)}}{\sum_{i=1}^K e^{E\left(1, i\right)}}  \right]\\
&\simeq \frac{1}{K} \left[ \hat{\gamma}_{X_j}\left(s, \tau\right) + \gamma_{X_j}\left(s, \tau \right) \left(1 - \frac{T\hat{Z}\left(X_j\right)}{Z\left(X_j\right)}\right) - \sum_{k=1}^K \left[\left(\gamma_{X_k}\left(s, \tau\right) - \gamma_{X_{k, k}}\left(s, \tau\right)\right) \frac{e^{E\left(k, k\right)}}{\sum_{i=1}^K e^{E\left(k, i\right)}} + \gamma_{X_{k, j}}\left(s, \tau\right) \frac{e^{E\left(k, j\right)}}{\sum_{i=1}^K e^{E\left(k, i\right)}}\right]\right] \\
\end{align}$$
-->
<!--
&\simeq \frac{1}{K}\left[ \left(\hat{\gamma}_{X_j}\left(s, \tau\right) - \gamma_{X_j}\left(s, \tau \right) \frac{\hat{Z}\left(X_j\right)}{Z\left(X_j\right)}\right) - \sum_{k=1}^{K}\left(\gamma_{X_{k, j}}\left(s, \tau\right) - \gamma_{X_k}\left(s, \tau\right)\right) \frac{e^{E\left(k, j\right)}}{\sum_{i=1}^K e^{E\left(k, i\right)}} + \mathbb{1}\left(1, j\right)\gamma_{X_1}\left(s, \tau\right) - \gamma_{X_{1, j}}\left(s, \tau\right) \frac{e^{E\left(1, j\right)}}{\sum_{i=1}^K e^{E\left(1, i\right)}}  \right]
\end{align}$$
-->

<!--
&\implies \frac{\partial}{\partial y_s^{\tau}\left(j\right)} I_{MCE} = \mathbb{E}_{\mathcal{B}}\left[ \hat{\gamma}_{X_1}\left(s, \tau\right) + \gamma_{X_1}\left(s, \tau\right)\left[\left(1-\frac{\hat{Z}\left(X_1\right)}{Z\left(X_1\right)}\right) + \frac{e^{E\left(1, j\right)}}{\sum_{i=1}^K e^{E\left(1, i\right)}}\right] -  \gamma_{X_{1, j}}\left(s, \tau\right) \frac{2e^{E\left(1, j\right)}}{\sum_{i=1}^K e^{E\left(1, i\right)}}\right]
### Semi-supervised Algorithm
$$\begin{align}
\mbox{Sample} B_{sup} &~ \mathcal{D}_{sup} \\
\mbox{Sample} B_{unsup} &~ \mathcal{D}_{unsup} \\
\mbox{Update } \Theta \mbox{ according to } &\mathcal{L}_{MMI}\left(B_{sup}, \Theta \right) \\
\mbox{compute} \phi\left(B_{unsup}\right) &\\
\mbox{\textbf{for}} &\mbox{all combinations } \left(i, j\right) \\
& \phi_{i,j} = \phi\left(B_{unsup}\right)_i + \phi\left(B_{unsup}\right)_j \\
& \mbox{Do forward-backward on} \phi_{i,j} \circ G \mbox{and store} \\
\mbox{compute gradients using the graident fomula from before}\\
\end{align}$$

-->
