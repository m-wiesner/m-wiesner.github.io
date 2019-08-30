---
title: "Bootstrap Sampling"
layout: post
date: 2019-08-30 17:00
headerImage: false
tag:
- markdown
- elements
star: true
category: blog
author: Matthew Wiesner
description: Bootstrap Sampling
---
# Bootstrap Sampling

I have always wondered why we don't report confidence intervals in Automatic Speech Recognition. I have just assumed that we are so familiar with our data sets that we have a good feel for what consititutes significant improvement. It seems like something we should worry about more, especically because it is easy to compute. I'm writing this post to remind myself how/why bootstrap sampling works, document some tools that are easy to use that compute these things for us, and to maybe motivate others to start incorporating this in their papers.

\(E_D[X]\)

