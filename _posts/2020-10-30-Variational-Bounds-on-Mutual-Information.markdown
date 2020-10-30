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
which when coupled with a specific neural architecture was called [Contrastive Predictive Coding (CPC)](https://arxiv.org/pdf/1807.03748.pdf), is very useful, but a subsequent [paper
](https://arxiv.org/pdf/1905.06922.pdf) I think more thoroughly explores the training objecgive InfoNCE, and provides some better theoretical underpinnings for why this objective works.

In general estimating the mutual information between two variables is difficult. In practice, the best we can really do is to estimate bounds on the mutual information. In this review
as well as in the paper we are describing, the mutual information is viewed in terms of the KL-divergence

$$I\left(p\left(X\right); p\left(Y\right)\right) = D_{KL}\left(p\left(X, Y\right) || p\left(X\right)p\left(Y\right)\right)$$

Recall that KL-divergence can be viewed as the expected log ratio of the joint distribution over two random variables to the product of their marginal distributions.
If the joint distribution is the same as the product of the marginal distributions, i.e., the random variables are independent, then the mutual information is 0, meaning that $$X$$
is completely independent of $$Y$$. If $$p\left(X, Y\right).

## Lower Bound on Mutual Information
To lower

