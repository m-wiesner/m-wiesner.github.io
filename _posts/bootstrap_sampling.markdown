<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# Bootstrap Sampling

I have always wondered why we don't report confidence intervals in Automatic Speech Recognition. I have just assumed that we are so familiar with our data sets that we have a good feel for what consititutes significant improvement. It seems like something we should worry about more, especically because it is easy to compute. I'm writing this post to remind myself how/why bootstrap sampling works, document some tools that are easy to use that compute these things for us, and to maybe motivate others to start incorporating this in their papers.

\(E_D[X]\)

