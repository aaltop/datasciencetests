# What

Repository for various small data science projects that I work on
to study different data science concepts. The focus is specifically
to study the theoretical ideas through implementing them in practice.
As a result, the code may not be absolutely high quality,
as long as it roughly implements the data science task it's supposed to.
I already do general programming and software development elsewhere,
so I don't want to get bogged down in typing and testing every function
when the aim is to understand a data science concept.

# Why

Like many others, data science is a vast field and is currently a field
of focus as well, with many new ideas floating around. From working
with basic numerical data in tabular format to computer vision and
natural language processing, there's a lot of concepts to cover. Of course,
it may be better to try and focus on a specific area, but in turn,
many concepts may be usable across the different sub-disciplines:
convolution as a neural network concept originated from computer vision,
yet is also useful in processing non-image data. Similarly,
one approach to combatting the vanishing gradient problem was introduced
in relation to [an image classification task](https://arxiv.org/abs/1512.03385),
yet these residual mappings could be of use elsewhere too. Besides
being able to potentially implement these ideas yourself, it can
also be useful just to be aware of them when working with
ready-made solutions. Whether using a Model-as-a-Service or running
a pre-trained model yourself, understanding why the output is as it is
and what the limitations of the model are is, if not crucial, at least useful;
understanding that a large language model will generally have a cut-off
point, may not have access to the internet, may not be trained on a specific
set of data, and generally just chains the most statistically relevant words
and sentences together based on its training data 
helps in understanding what a large language model
can and cannot do, even if you've not constructed and trained one yourself
from scratch.

Besides understanding the theoretical better, I'll
get more and more comfortable with the tools of the trade. With a language
like Python, it is especially important to know how to vectorise operations
that could be computationally expensive. Knowing those specific methods
in PyTorch that allow you to manipulate tensors to forms more suitable
for operations that would by default for a programmer happen more naturally
in loops is helpful. However, it's often the case that to find these solutions and
become familiar with them, you need to come across a relevant problem first.
That is just one example, and knowing the ins and outs of any library
from memory takes some time and work with the library. Likewise,
having come across a problem and a corresponding solution in the past
will make it easier in the future to solve similar problems, even
when not working with the same toolset. Knowing of the concept
of tiling a tensor in PyTorch, you might think of the concept
when working with Numpy as well and as a result, have an easier
time looking up how to implement the necessary functionality
in Numpy.

Yes, I'm apparently describing why learning things is good.
