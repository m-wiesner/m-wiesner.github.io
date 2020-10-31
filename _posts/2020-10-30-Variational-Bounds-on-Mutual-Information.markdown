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
\log{\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]} &\leq \frac{\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]}{a\left(y\right)} + \log{a\left(y\right)} + 1 \\
&\implies I_{UBA} \geq \mathbb{E}_{p\left(X, Y\right)}\left[f\left(X, Y\right)\right] - \mathbb{E}_{p\left(Y\right)}\left[\frac{\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)}\right]}{a\left(y\right)} + \log{a\left(y\right)} + 1\right] = I_{TUBA} \\
\end{align}$$ 

Since this relationship holds true for all values of $$a\left(y\right)$$, we can set $$a\left(y\right) = e \ \forall \ y$$ at the expense of having a slightly looser bound on the mutual information. This gives us the bound 

$$I\left(X, Y\right) \geq I_{TUBA} \geq \mathbb{E}_{p\left(X, Y\right)}\left[f\left(X, Y\right)\right] - \mathbb{E}_{p\left(Y\right)}\left[ e^{-1}\mathbb{E}_{p\left(X\right)}\left[e^{f\left(X, Y\right)\right]I_{NJW}$$
