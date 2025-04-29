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
author: MatthewWiesner
description: Some basic results needed to understand Neural ODEs
---

I took a break from energy based models for a while, and in the meantime diffusion methods really took off.
I want to learn about diffusion mehtods, and neural ODES more generally and these are just some basic results that
I think are needed to understand neural ODES. Since I am hoping this will be a mostly standalone reference, I am including
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

We can use this to derive all sorts of matrix / vector derivatives. For instance when $f\left(x\right) = x^T x$$ we have

$$ df = \left(\left(x + dx\right)^T \left(x + dx\right)\right)  - x^T x = x^T x + dx^Tdx + x^T dx + dx^T x - x^T x = 2x^T dx$$ 

$$ df = \left(\nabla f\right)^T dx = 2x^T dx \implies \nabla f = 2x.$$

Similary we can define the matrix perturbations and associated "gradients". The product rule is quite helpful for this and still works for vector and matrix calculus.

$$ df(u, v) = f(u + du)g(v + dv) - f(u)g(v) = (f(u) + f\`(u)\ du)(g(u) + g\`(v)dv) - f(u)g(v) $$
$$ = f\`(u)\ du\ g\`(v) dv + f\`(u)\ du\ g(u) + f(u)g\`(v)\ dv + f(u)g(v) - f(u)g(v)$$  

$$d\left(AA^{-1}\right) = d\left(I\right) = \mathbb{0}_{n \times x} = Ad(A^{-1}) + dAA^{-1} \implies dA^{-1} = - A^{-1} dA A^{-1}$$

This result will be important.

>$$ dA^{-1} = - A^{-1} dA A^{-1} $$
