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

There has been incredible success using InfoNCE as a constrastive unsupervised pretraining objective for ASR in the past year. The original paper that introduced this training objective,
which when coupled with a specific neural architecture was called [Contrastive Predictive Coding (CPC)](https://arxiv.org/pdf/1807.03748.pdf), is very useful, but a subsequent [paper](https://arxiv.org/pdf/1905.06922.pdf) I think more thoroughly explores the training objective InfoNCE, and provides some better theoretical underpinnings for why this objective works.

In general estimating the mutual information between two variables is difficult. In practice, the best we can really do is to estimate bounds on the mutual information. In this review
as well as in the paper we are describing, the mutual information is viewed in terms of the KL-divergence

$$I\left(X; Y\right) = I\left(p\left(X\right); p\left(Y\right)\right) = D_{KL}\left(p\left(X, Y\right) || p\left(X\right)p\left(Y\right)\right)$$

Recall that KL-divergence can be viewed as the expected log ratio of the joint distribution over two random variables to the product of their marginal distributions.
If the joint distribution is the same as the product of the marginal distributions, i.e., the random variables are independent, then the mutual information is 0, meaning that $$X$$ is completely independent of $$Y$$.

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

$$I\left(X; Y\right) \leq \mathbb{E}_{p\left(X\right)}\left[D_{KL}\left(p\left(Y|X\right) || q\left(Y\right)\right)\right]$$

This term can be thought of as a regularizer: in an overfit model $$p\left(Y|X\right)$$ will be too confident in its predictions, and we can smooth these predictions by making them look more like the prior distribution over $$p\left(Y\right)$$. 


## Lower Bound on Mutual Information
To lower bound the mutual information, we factor in the opposite direction $$p\left(X, Y\right) = p\left(X | Y\right)p\left(Y\right)$$, introduce a third distribution $$q\left(X | Y\right)$$ and use the non-negativity of the KL-divergence as well as the differential entropy of a random variable to arrive at our lower bound. 

$$\begin{align}
I\left(X; Y\right) &= \mathbb{E}_{p\left(X, Y\right)}\left[\frac{p\left(X|Y\right)q\left(X | Y\right)}{p\left(X\right)q\left(X|\Y\right)}\right] \\
&= \mathbb{E}_{p\left(X, Y\right)}\left[\frac{q\left(X | Y\right)}{p\left(X\right)}\right] + \mathbb{E}_{p\left(y\right)}\left[D_{KL}\left(p\left(X|Y\right) || q\left(X|Y\right)\right)\right] \\
&\geq \mathbb{E}_{p\left(X, Y\right)}\left[\frac{q\left(X | Y\right)}{p\left(X\right)}\right] \mbox{due to the non-negativity of KL-divergence} \\
&= \mathbb{E}_{p\left(X, Y\right)}\left[q\left(X | Y\right)\right] - \mathbb{E}_{p\left(X\right)}\left[\log{p\left(X\right)}\right] \\
&= \mathbb{E}_{p\left(X, Y\right)}\left[q\left(X | Y\right)\right] + h\left(X\right) \\
\end{align}$$

