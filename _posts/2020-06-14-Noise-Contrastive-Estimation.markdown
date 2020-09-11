---
title: "(Info) Noise Contrastive Estimation"
layout: post
date: 2020-06-14 13:35
image: /assets/images/markdown.jpg
headerImage: false
tag:
- Machine Learning
star: false
category: blog
author: MatthewWiesner
published: false
description: (Info) Noise Contrastive Estimation
---


I'm writing this to summarize what I know about Noise Contrastive Estimation (NCE), and the more recently proposed InfoNCE. I just want it written down in a way that makes sense to me. I'm mostly basing this off of [this paper](https://arxiv.org/pdf/1206.6426.pdf), by Andriy Mnih, and Yee Whye Teh though it was originally proposed by [Gutmann and Hyvärinen]( http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf). [InfoNCE](https://arxiv.org/pdf/1807.03748.pdf) was proposed by van den Oord

_____________________________

## Estimating $$p_\theta\left(x\right)$$
In a [previous post](https://m-wiesner.github.io/Energy-Based-Models) discussing energy-based models, we focused on the problem training a model for 

$$p_\theta\left(x\right) = \frac{e^{-E_\theta\left(x\right)}}{\int_x e^{-E_\theta\left(x\right)}}$$

We found that the gradient of the objective function $$\mathbb{E}_{p\left(x\right)}[-\log{p_\theta\left(x\right)}]$$ took the form of a difference of two expectations. 

$$\nabla_{\theta} \log{p_{\theta}\left(x\right)} = \mathbb{E}_{p_{\theta}\left(x\right)} \left[\nabla_{\theta}E\left(x\right)\right] - \nabla_{\theta} E\left(x\right) $$

The first term is approximated by averaging over samples that come from the model distribution $$p_\theta\left(x\right)$$. The second term is the gradient from an actual speech sample (drawn from $$p\left(x\right)$$. The starting point for this post is to think about the model-generated samples as being negative examples against which the true examples are contrasted. 

When the model is bad --- at the start of training for instance --- the model-generated samples are somewhat random. The training objective tries to alter model parameters such that the energy of the ground-truth samples is lower than that of the The negative samples. In doing so, the model is also more likely to produce samples that resemble those seen in training. We are iteratively refining the model samples and forcing the model to contrast against better and better approximations of the ground truth data. In an ideal world, these approximations would eventually exactly match the ground truth data, the difference in gradients would be 0, and this means that training has converged.

## Simplifying training
In the above scenario, learning every detail about the data $$x$$, may ultimately prove hard, especially when $$x$$ is high-dimensional data, and sampling the negative examples from the model is computationally expensive. NCE addresses both of these issues. Instead of sampling from the model distribution, we estimate from a fixed noise distribution that should somewhat approximate the empircal data distribution. Furthermore, instead of directly learning $$p\left(x\right)$$ (think all sequences of words) we instead focus on learning conditional probabilities $$p\left(x | \mbox{context}\left(x\right) \right)$$ (think single word given a context of other surrounding words). Even this task can be challenging. It turns out that we can actually train the model to perform a *different* taks that involves learning the *same* conditional density $$p\left(x | \mbox{context}\left(x\right) \right)$$. This different task involves introducing a latent variable representing the source of the data, and then predicting that source. For simplicity we will call $$ h\left(x\right) = \mbox{context}\left(x\right)$$. In other words we want to optimize the following objective function.

$$\min_{\theta} \mathbb{E}_{\left(x, h\left(x\right)\right)} \left[-\log{p_{\theta}\left(D | x, h\left(x\right)\right)}\right]$$












