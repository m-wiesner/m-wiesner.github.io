---
title: "Adjoint Method for ODEs Part II"
layout: post
date: 2025-04-30 16:50
image: /assets/images/markdown.jpg
headerImage: false
tag:
- Machine Learning
star: false
category: blog
author: Matthew Wiesner
published: true
description: Adjoint Method for ODEs
---

In the first part, some basic results were introduced needed to understand the adjoint method for ODEs. I wrote about the adjoint method when solutions followed linear and non-linear constraints, or alternatively, we viewed them as implicitly defined functions. Here I introduce some very basic ideas about ODEs in case no one has seen them before and show how these effectively replace the constraints on solutions. 

First, when we looked at the adjoint method with linear constraints we were looking for vector solutions to the equation

$$Ax = b$$

Let us reinterpret our vector solution, $$x \in \mathbb{R}^{d x 1}$$ so that we view it as having some structure:

$$x = \begin{bmatrix} x_0[0] \\ \vdots \\ x_{\frac{d}{2}}[0] \\ \\ \hline \\ x_{\frac{d}{2} + 1}[1] \\ \vdots \\ x_{d}[1] \end{bmatrix}$$

It is actually two stacked vectors representing two different, but related solutions. Correspondingly, the matrix, $$A$$, and vector, $$b$$, would also have to encode this structure since

$$ x = A^{-1} b $$

If the two solutions represent an evolution of a solution across time, then it is reasonable to imagine that the two solutions are similar or related in some relatively easy to describe way. So rather than encoding each solution explicitly by specifying a submatrix of $$A$$ for each time interval, it would be more compact to specify some initial point and then directly encode the change from that point. This initial point could be given or could itself be the solution of some linear or non-linear equation $$x(0) = A^{-1}b$$. Then only the way $$x(t)$$ change is encoded rather than than solving a whole new equation for each moment in time. Derivatives are natural for this and this is where we define a differential equation. 
 
$$\mbox{Given } x(0), \mbox{ solutions, } x(t) \mbox{, satisfy } \frac{dx}{dt} = g\left(x(t), \theta, t\right),$$

i.e., $$x\left(t+\epsilon\right) = x\left(t\right) + g\left(x(t), \theta, t\right) \left(t + \epsilon\right)$$.

At each moment in time, a scalar function, $$f\left(x\left(t\right)\right)$$ of the solution $$x(t)$$ can be evaluated, e.g., a loss function in machine learning. Treating the entire function $$x(t)$$ over some range of time, $$t \in [t_0, t_1]$$, as a solution, the natural extention of the scalar cost function over the entire function space would be to sum the instantaneous costs over the interval. This corresponds to ...

$$J\left(x\left(t\right)\right) = \int\limits_{t_0}^{t_1} f\left(x\left(t\right)\right) dt,$$

remembering that $$x\left(t\right)$$ implicitly depends on $$\theta$$.

We are once again interested in seeing how this scalar function over the whole solution, $$x\left(t\right)$$ changes w.r.t. the parameters defining $$x\left(t\right)$$. So we look at the derivatives.

$$dJ\left(x\left(t\right)\right) = J`(x\left(t\right))\ dx\left(t\right)$$

Rewording this again, we are looking at the change in the loss function, which is itself defined as a sum of differences of a function over the entire range $$[t_0, t_1]$$, the function over the same range $$[t_0, t_1]$$ changes slightly. We are not talking about a change across time here. We are talking about how a change in the shape of the function results in a change in the loss function, i.e., if a different trajectory was used, would the loss be better?

## Solving for $$x(t)$$

Just as before we have defined $$x(t)$$ implicitly, but this time in terms of an initial condition and how it changes over time. The solution is

$$ x(t) = \int g\left(x\left(t\right), \theta, t\right) dt + C_0$$

And then we take the gradient.

$$ dx(t) =  \int g\left(x\left(t\right), \theta, t\right) dt + C_0
= \int\limits_{t_0}^{t_1} \frac{\partial f\left(x\left(t\right), \theta\right)}{\partial x} dx dt + \int\limits_{t_0}^{t_1} \frac{\partial f\left(x\left(t\right), \theta\right)}{\partial \theta} d\theta dt = \int\limits_{t_0}^{t_1} \left[ \frac{\partial f\left(x\left(t\right), \theta\right)}{\partial x} dx + \frac{\partial f\left(x\left(t\right), \theta\right)}{\partial \theta} d\theta \right] dt $$
