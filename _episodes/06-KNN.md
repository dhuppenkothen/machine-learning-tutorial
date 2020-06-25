---
title: "K-Nearest Neighbours"
teaching: 0
exercises: 0
questions:
- "How can we get a computer to classify new objects using the training data we recorded?"
objectives:
- "Learners understand how the K-Nearest Neighbours algorithm works conceptually"
- "More advanced learners can also write their own version of the algorithm in code, and use the data they generated to classify new data points"
keypoints:
- ""
---

So we've learned about decision boundaries, and we've drawn some by hand. Let's now figure out how a computer makes a decision to put a 
sample into one class or another. One algorithm to do that is called *K-Nearest Neighbours*. It is called that way because it takes a sample 
whose class you don't know (say, a candy you've just picked out of your bowl and placed on your paper), then looks at a number of samples 
that are close to that unknown one around it, and decides on the class of the sample based on what we know about those neighbours. 

Imagine you've taken a piece of candy from the bowl, and put it down on your paper. You then draw lines to all the other candies already on the 
paper to mark their distances, and then pick the five sweets with the smallest distances to your new, unknown example. Since those below 
to your training data, you know what they are! So say there are two peanut M&Ms, and three skittles. In the **K-Nearest Neighbour** algorithm, you 
would now decide that the piece of candy you've put down on the graph is probably a skittle, too, since the majority of samples closest to it 
are skittles.

Why did we pick 5? In fact, this was just an example, the "N" in the algorithm's name is a place-holder for the number of neighbours you 
choose to compare to. You could choose 1, or 100, but they'll change the behaviour of the algorithm. How to choose a good value is something 
we're going to explore a little later in this tutorial.
 
> ## Challenge
>
> [Note: I think this challenge needs work. It introduces an extra concept (imbalanced classes) that I don't think should be
> introduced here.]
>
>
> Let's use our new knowledge of the K-Nearest Neighbours algorithm on our sweets. Pick a new piece of paper, 
> and distribute two types of your sweets on a new graph (which can have the same axes as before, or different ones). 
> Leave about a couple so behind, we'll use those in a minute. 
> Instead of drawing decision boundaries like we did in the last episode, now pick one of the sweets you didn't 
> put down, and locate them on the graph. 
> Now estimate the five closest sweets to the one you've just put on your graph. Are they the same type of sweet?
> If you choose just 1, 10, or 20 neighbours, how does your result change?  
>
> > ## Solution
> >
> > Let's take a look at my version of this challenge:
> > 
> > <a href="{{ page.root }}/fig/sweets_knn.jpg"><img src="{{ page.root }}/fig/sweets_knn.jpg" alt="Graph of skittles, two types of M&Ms and jellybeans as a function of length (x-axis) and height (y-axis)." /></a>
> >
> > Here I've made a graph of the length (the widest extent) and the height (the smallest extent) of our sweets and then only skittles and plain M&Ms 
> > on the plot, because they're quite similar in overall shape, but skittles tend to be thicker. I pretended that I only have two plain M&Ms in my 
> > training set and many more skittles then placed another sweet (the blue one on the figure) on the graph. Looking at it it seems to be much closer 
> > to the two M&Ms than it is to any of the skittles, so we might assume that it's an M&M, too. In order to use our K-Nearest Neighbour algorithm 
> > with $k = 1$, we look for the closest sweet already on the plot: one of the orange ones, which seem to be about equidistant from our blue one.
> > Both of these are M&Ms, so with k=1 (and, indeed k=2) our conclusion stays the same.
> > 
> > Now what happens if I increase k? Let's try for k=5.   
> > The five sweets closest to my blue one still contain the two M&Ms, but they now also contain three skittles (the green, yellow and red ones on 
> > the figure above)! We would  now conclude that actually, our sweet is a skittle not an M&M!
> >
> > What happened here? Well we fell into a common pitfall of machine learning: if our *training data* (the sweets we placed on the graph for which 
> > we know what they are) is very _imbalanced_, that is if I have many more of one type (skittles) than the other (M&Ms), a machine learning 
> > algorithm might ignore the existance of examples of the smaller class (here the M&Ms) entirely. Notice how no matter where you'd place 
> > a sweet on the graph, if k = 5 or above, the algorithm will *always* conclude that the sweet we placed is a skittle, because for any sweet the 
> > number of M&Ms that are closest is at most two, and the number of nearby skittles must always be at least three or above.   
> > 
> > So when you're doing machine learning, beware of imbalanced training data sets! We will explore this more later in the lesson.
> >
> > {: .output}
> {: .solution}
{: .challenge}

## The Mathematics of K-Nearest Neighbour

Let's look a little more at how we can calculate our k-nearest neighbour results in practice.

For this, we're going to formalize our previous intuitions a little bit more. 
Assume you have training examples $$\{X_1, X_2, X_3, ... X_n\}$$, where each $$X_i$$ is a vector of 
_features_ (like length, height, colour etc.) and associated labels $$\{y_1, y_2, y_3 ..., y_n\}$$ (e.g. 
plain M&M, peanut M&M, skittle, ...). 

For a given new unknown example $$<X_u, y_u>$$, we compute a distance function $$d(X_u, X_i)$$ between our new 
example and all training examples. There are a number of different distance functions one can use. A commonly 
used one is the _Euclidean distance_, 

$$ d(X_u, X_i) = \sqrt{\sum_{j=1}^{m}{\left( x_j^{[u]} - x_j^{[i]}\right)^{2}}}$$

Assuming we have computed the distance between our unknown example and all training examples we can select the 
set $$D_k  = \{ <X_1, y_1>, <X_2, y_2>, ...,  <X_k, y_k> \} $$ of $$k$$ nearest neighbours (examples with the smallest 
distance). We can now compute the _conditional probability_ for each class (i.e. the fraction of points in $$D_k$$ with 
a given class label):

$$ P(y=j | X=x) = \frac{1}{k} \sum_{i=1}^{k}{I(y_i = j)} $$

where $$I$$ is the _indicator function_ ($$I = 1$$ if $$i=j$$, else $$I = 0$$).

> ## Advanced Challenge
>
> In this challenge, we are going to write our own implementation of the k-nearest neighbours algorithm!
> 
> 1) First, download [the data](sweets_data) from the GitHub repository. 
> 2) Load this data in whatever programming environment you use for data analysis (I use Python, so I would use 
> the Pandas library to load it). 
> 3) Separate out the first two thirds and the last third of the table into separate tables. The first table with 
> most of the data will be our designated **training data**. For the second table, we'll assume that we don't know 
> the labels. This'll be our **target data**, which we'd like to classify.
> 4) Then, for each of our training and target data, store the *features* in a separate array from the *labels*.
> 5) Write a function that takes two examples and calculates the Euclidean distance defined above between them.
> 6) Calculate the distances between the first example in your target data, and all examples in your training data.
> 7) Copy your array of class labels for your training data, and then order this array by the distances you 
> just derived (so that the label associated to the example with the smallest distance is first, the one with the second-smallest
> distance next, etc.)
> 8) Out of this array, you can now pick the first $$k$$ examples for a varying number of $$k$$. Try with $$k = 1, 2, 5, 10, 20$$, and 
> for each count the occurrence rates of the different class labels.
> 
> Assuming that we pick the class for our example that corresponds to the highest number of training examples out of our $$k$$ nearest 
> neighbours, how does the prediction for your chosen target example change as you increase $$k$$? Does the algorithm predict your example
> accurately?
>
> > ## Solution
> > 
> > Need to write solution in Python.
> >
> > {: .output}
> {: .solution}
{: .challenge}
  

[sweets_data]: https://github.com/dhuppenkothen/machine-learning-tutorial/tree/gh-pages/data

{% include links.md %}
