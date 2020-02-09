---
title: "Energy Based Models"
layout: post
date: 2020-02-05 15:56
image: /assets/images/markdown.jpg
headerImage: false
tag:
- Machine Learning
star: false
category: blog
author: MatthewWiesner
description: Energy-Based Models and Stochastic Gradient Langevin Dynamics
---

I've wanted to learn about generative neural models for some time. I'm focusing on energy-based models for now. I had been trying to come up with
other ways to use untranscribed speech in domain adaptation when I came across [this paper](https://arxiv.org/pdf/1912.03263.pdf), which was
doing something very similar to what I had been thinking about. Most of this post is just me going through the background I needed to
understand this paper.


# Energy Based Models

The key idea in Energy Based Models (EBMs) is to generate a score for data points. Our data can be viewed as measurements of the underlying system
that we are attempting to model. We use the score as a goodness measure of a particular configuration. This score is termed the Energy. 

$$E_{\theta}\left(x_i\right) : \mathbb{R}^{d x 1} \to \mathbb{R}$$

The only restriction on this score is that it result in a finite integral over the entire domain of our data. We can generate a probability distribution
from the energy.

\begin{align}
p_{\theta}\left(x\right) &= \frac{e^{-E\left(x\right)}}{\int_{x \in \mathcal{X}} e^{-E\left(x\right)} dx} \\\
&= \frac{e^{-E\left(x\right)}}{Z\left(\theta\right)}
\end{align}

Here, $$Z\left(\theta\right)$$ is known as the partition function, and computing it is impossible because we can never integrate over all possible value
for our data. In spite of this, we will proceed to take gradients of this function as if we could perform gradient descent.

$$\nabla_{\theta} \log{p_{\theta}\left(x\right)} = -\nabla_{\theta} E\left(x\right) - \nabla_{\theta} \log{Z\left(\theta\right)} $$

$$= -\nabla_{\theta} E\left(x\right) - \frac{1}{Z\left(\theta\right)} \int_{x \in \mathcal{X}} e^{-E\left(x\right)} \left(- \nabla_{\theta}E\left(x\right)\right) dx$$

$$= -\nabla_{\theta} E\left(x\right) + \int_{x \in \mathcal{X}} \frac{e^{-E\left(x\right)}}{Z\left(\theta\right)} \nabla_{\theta} E\left(x\right) dx$$

$$= -\nabla_{\theta} E\left(x\right) + \int_{x \in \mathcal{X}} p_{\theta}\left(x\right) \nabla_{\theta} E\left(x\right) dx$$

$$= \mathbb{E}_{p_{\theta}\left(x\right)} \left[\nabla_{\theta}E\left(x\right)\right] - \nabla_{\theta} E\left(x\right) $$

So if we know how to compute the gradient with respect to $$E\left(x\right)$$, then we can approximate the expectation by sampling.
This sampling procedure therefore becomes crucial. One easy way of sampling is to use a technique known as Stochastic Gradient Langevin Dynamics (SGLD).

# SGLD

The main idea behind SGLD is to generate low-energy data points according to our current model. If we can do this, then we basically have a way of
sampling from from $$p_{\theta}\left(x\right)$$ since the low energy points should correspond to likely points. And this sampling technique is itself very
similar to stochastic gradient descent (SGD).

We initially start with points sampled uniformly from our domain. Then we find the direction of minimum energy and take a step in that direction.
If we did this for enough steps, we would eventually reach the points of minimum energy, which correspond to the modes of the $$p_{\theta}\left(x\right)$$.
But we obviously do not want to only sample the modes of the distribution. To ensure that we are at least sometimes returning samples that correspond to other points
we need to inject some noise. In this way we can sample points around the modes of the distribution. The amount of noise, 
or if we model the noise as Gaussian, the variance should be tuned appropriately to ensure the desired behavior.

Formally, the sampling procedure is:

```python
x = uniform_sample(-1, 1) * 3 * [sigma_1, sigma2, ..., sigma_D] # Sample uniformly from the inout domain (approximated by 3 standard devations per dimension)
for i in range(num_sgld_steps):
  x += step_size * grad(E(x), x) + normal(x, sgld_variance) # Gradient of the energy E with respect to x
```

In this way we can generate a full minibatch of samples which we use to approximate the gradient. In practice rather than sampling at random uniformly
for each minibatch, a buffer of past generated points is stored and points at a new iteration can be sampled from this buffer instead of uniformly.
In this way we can sample points from previous iterations that may not be quite as random, and we can take more steps using these points leading to 
easier convergence without requiring too many SGLD steps for any given minibatch. This is known as replay memory. Some points are still sampled completely
randomly. 
