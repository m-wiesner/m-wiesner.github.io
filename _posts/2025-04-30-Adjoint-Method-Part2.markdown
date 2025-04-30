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

It is actually two stacked vectors representing two different, but related solutions. Correspondingly, the matrix $$A$$, and vector $$b$$ would also have structure encoding valid solutions. If the two solutions represent and evolution across time, then it is reasonable to imagine that the two solutions are similar or related in some relatively easy to describe way. For instance, the  
