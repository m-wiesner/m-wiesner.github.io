---
title: "Adjoint Method for ODEs Part I"
layout: post
date: 2025-04-29 15:56
image: /assets/images/markdown.jpg
headerImage: false
tag:
- Machine Learning
star: false
category: blog
author: Matthew Wiesner
published: true
description: Some basic results needed to understand Neural ODEs
---

I took a break from energy based models for a while, and in the meantime diffusion methods really took off.
I want to learn about diffusion mehtods, and neural ODEs more generally and these are just some basic results that
I think are needed to understand neural ODEs. Since I am hoping this will be a mostly standalone reference, I am including
some basic review.

# Matrix/Vector Calculus

A quick refresher on derivatives: The derivative of a function, $$f\left(x\right)$$ is defined as 

$$ \frac{df}{dx} = \lim\limits_{h \to 0} \frac{f\left(x + h\right) - f\left(x\right)}{h}$$ 

However, when working with vectors or matrices, this notion of a derivative -- which relies on division by the perturbation
we are applying -- doesn't make sense. Instead of thinking about the derivate as the ''slope'', let us use the interpretation of it
as affine approximations of a function at each point and define the derivate (now called gradient) implicitly as ...

$$ f\left(x + h\right) \approx f\left(x\right) + f`\left(x\right) \left(h\right)$$

Rearranging, and taking $$h \to 0$$ we have 

$$ f\left(x + h\right) - f\left(x\right) = f`\left(x\right)\left(h\right) $$

We will introduce some new notation now. 

$$ \partial f := f\left(x + h\right) - f\left(x\right) $$
$$ \partial x := h \mbox{ where }  h \mbox{ is really small}.$$

This gives

$$ \partial f = f`\left(x\right) \partial x.$$

When these things are vectors we can write

$$ df = \left(\nabla f\right)^T dx.$$

We can use this to derive all sorts of matrix / vector derivatives. For instance when $$f\left(x\right) = x^T x$$ we have

$$ df = \left(\left(x + dx\right)^T \left(x + dx\right)\right)  - x^T x = x^T x + dx^Tdx + x^T dx + dx^T x - x^T x = 2x^T dx$$ 

$$ df = \left(\nabla f\right)^T dx = 2x^T dx \implies \nabla f = 2x.$$

Similary we can define the matrix perturbations and associated "gradients". The product rule is quite helpful for this and still works for vector and matrix calculus.

$$ df(u, v) = f(u + du)g(v + dv) - f(u)g(v) = (f(u) + f`(u)\ du)(g(u) + g`(v)dv) - f(u)g(v) $$
$$ = f`(u)\ du\ g`(v) dv + f`(u)\ du\ g(u) + f(u)g`(v)\ dv + f(u)g(v) - f(u)g(v)$$  

$$d\left(AA^{-1}\right) = d\left(I\right) = \mathbb{0}_{n \times x} = Ad(A^{-1}) + dAA^{-1} \implies dA^{-1} = - A^{-1} dA A^{-1}$$

This result will be important.

$$ \boxed{dA^{-1} = - A^{-1} dA A^{-1}} $$

# Adjoint Method for Linear Functions

Imagine we want to take a gradient of some scalar function $$f\left(x\right)$$ where $$A(p)x=b$$ and depends on some parameters $$p$$.
The goal is to find the best matrix $$A(p)$$, i.e., the one that optimizes $$f\left(x\right)$$. Then we need the gradient of $$x$$ w.r.t. $$p$$. But this comes from the gradient w.r.t. $$A$$.

$$ x = A^{-1}b \implies dx = d(A^{-1}b) = dA^{-1}b = -A^{-1} dA A^{-1}b = -A^{-1} dA x$$

Alternatively, we can take the gradient of the implicit function ...

$$ d(Ax) = 0 = dAx + Adx \implies dx = A^{-1}dA x$$

This will be useful later.

In other words, the above relationship tells us how much $$x$$ would change when a small change in $$A$$ is applied. We are rewriting $$dx$$ in terms of $$dA$$, i.e., a corresponding change in the parameters. Substituting in for $$dx$$ in $$df = f`\left(x\right) dx$$ we have 

$$ df = f`\left(x\right) A^{-1} dA x = -\left(f`\left(x\right) A^{-1}\right) dA x $$  

So while $$f`\left(x\right)$$ may be easy to compute, we want to find the harder to compute set of derivatives $$\frac{\partial f}{\partial p_i}$$ for some parameter $$p_i$$. Now here is the trick, which we have previewed by adding parentheses in the righthand side of the equation above.

We will call $$v^T = \left(f`\left(x\right) A^{-1}\right)$$

$$\implies v^T A = f`\left(x\right) \implies A^T v = f`\left(x\right)^T$$ 

So compute
  - $$f`\left(x\right)$$, i.e., the row vector with the derivative w.r.t. $$x$$
  - Then solve $$v = (A^T)^{-1} f`\left(x\right)^T$$
  - Then solve $$df = -v^T\ dA\ x$$

# Adjoint Method for Non-Linear Functions

Let us repeat the above exercise using a general non-linear definition of $$x$$ in terms of parameters $$p$$ described by $$g(x, \theta) = 0$$.

As before we can take the gradient of the implicit function, and solve for $$dx$$, which by the multivariate chain rule is ...

$$dg = \frac{\partial g}{\partial \theta} dp + \frac{\partial g}{\partial x} dx = 0$$
$$ \implies dx = -(\frac{\partial g}{\partial x})^{-1} \frac{\partial g}{\partial \theta} d\theta$$

the terms $$\frac{\partial g}{\partial \theta}$$ and $$\frac{\partial g}{\partial x}$$ are the Jacobian matrices w.r.t $$x$$ and $$\theta$$ respectively.

Just as before we will plug this into $$df = f`\left(x\right) dx$$, obtaining

$$df = -(f`(x) (\frac{\partial g}{\partial x})^{-1}) \frac{\partial g}{\partial \theta} d\theta$$

Here, $$v^T = f`(x) (\frac{\partial g}{\partial x})^{-1} \implies \frac{\partial g}{\partial x}^T v = f`(x)^T$$ and $$v = (\frac{\partial g}{\partial x}^T)^{-1} f`(x)^T$$

# Summary
In summary, we can compute the gradient with respect to any parameter $$p_i$$, by first computing the gradient with respect to $$x$$, i.e, $$f`\left(x\right)$$ and then solving a linear system of equations $$\frac{\partial g}{\partial x}^T v = f`\left(x\right)^T$$ to compute $$v^T = \left(f`\left(x\right) \left(\frac{\partial g}{\partial x}\right)^{-1}\right)$$, and hence $$df = -v^T \frac{\partial g}{\partial \theta} d\theta$$.

In the next post we'll consider what happens when x is the solution of an ordinary differential equation.
