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

Now that we have recorded some features for our training data set, how do we go from features to classifying new 
data points? We're ready to learn our first algorithm!

In *supervised machine learning*, which is what we are going to do here, one uses the training data to adjust the 
model (or algorithm) such that it ideally correctly classifies any new samples it sees as accurately as possible.
There are many new concepts and complications hidden in that statement, but we'll break them down one by one.

### Decision Boundaries

First, let's see how good our features actually are at separating the different types of sweets. For that, we're 
going to draw a figure of our features on a large sheet of paper and see if some types of sweets are clustered 
together, whether others are not.

> ## Challenge: Making Plots of Features
>
> For this exercise, we are going to make plots of our features using the sweets. 
> 1. First, pick two of the features you defined and gathered data for in the previous lesson. 
> 2. Now take a large sheet of paper, and draw two perpendicular axes. Label them according to the features you selected.
> 3. Look at your data and find the minimum and maximum for each feature across all of our training data set, then label the axes such that it spans the full range of your data set.
> 4. Place your different sweets in different places on your graph, depending on their values of the features you selected. For this exercise, you don't need to identify each sweet with the appropriate row: just place them approximately in the right place. 
>
> What do you see? Do your sweets all end up in different corners of the graph? Are some of them squashed together? 
> Compare your results with your neighbour's. Did you pick similar features? Why not? How are your graphs different?
>
> > ## Solution
> >
> > For this solution, I'm going to look at the length of a sweet compared to its height. That is, if you let it lie
> > on an even surface, I'm going to measure the longest horizontal axis, and the vertical axis away from the surface.
> > The data gathered during the last lesson is available in [this file](). 
> > Now I'm going to draw two axes on a sheet of paper, and label them "length" and "height". My measurements are 
> > in centi-meters (cm), so that's the unit we will use. Looking at the data, I can see that the smallest value 
> > in the "length" column is 1.1cm, the largest is 2.3cm. So it makes sense to let the x-axis, where length is 
> > recorded, go from 1.0cm to 2.5cm. Let's add some ticks to the axis in 0.1cm intervals, to help us place our 
> > sweets on the paper later. For the purposes of this exercise, the ticks don't need to be *exactly* spaced 
> > equally apart.
> > On the y-axis, we will present the height, i.e. the extent of the sweet away from the surface it is lying on. 
> > The data in our height column spans the range between 0.5cm and 1.1 cm, so let's make appropriate markings 
> > on the y-axis, too.
> > 
> > Now it's time to plot your actual data. If you were doing this for a real research project, you would 
> > now go through your table and mark a little "x" at the width and height for each sample. To make this 
> > exercise a little more descriptive, though, and less tedious, let's distribute our actual sweets on the paper.
> > For what we're demonstrating, you don't need to exactly identify each sweet with its row in the table.
> > For each type of sweet we've collected data for, take a look and record the range of values we've recorded for 
> > both the height and width, then place your sweets into the appropriate parts of the graph you drew.
> > Here's an example:
> > 
> > <a href="{{ page.root }}/fig/sweets_length_height.jpg"><img src="{{ page.root }}/fig/mixedsweets.jpg" alt="Graph of skittles, two types of M&Ms and jellybeans as a function of length (x-axis) and height (y-axis)." /></a>
> > 
> > You might notice that the sweets fall in different parts of the graph, and separate out quite well into their own 
> > individual clusters. This bodes quite well for using these features and an algorithm to try and do this automatically.
> > One particular thing you might also notice is that the peanut M&Ms are much more separated from the rest of the sweets
> > than the other types of sweets are amongst each other. This is because peanut M&Ms are on the whole larger than any of 
> > the other sweets, both in length and height. On the other hand, the plain M&Ms and the skittles have pretty similar 
> > shapes, but the skittles are significantly thicker (corresponding to a larger height), than the plain M&Ms.
> >
{: .challenge}

Hopefully you've found and drawn some features on a graph that separate at least some of the types of sweets out all right. 
If they only separate out one type of sweet from the rest, but not the others from each other, that's fine for now. 

When we look at the graph, we can fairly immediately see whether different types of sweets end up in the same corner 
of the graph, or if they end up in different corners. Our brain can easily parse the blank spaces between clusters of 
sweets, and because we also know which sweets are which, we can easily evaluate whether we've found features that 
separate out the different types of sweets well. But how does a computer do this?

There are different ways a computer can tell whether two clusters of samples overlap well or not, but one of the most 
common ones is to use an algorithm to draw a _decision boundary_.

> ## Decision Boundaries
>
> A decision boundary is a line (in the case of two features), where all (or most) samples of one class are on one side of that 
> line, and all samples of the other class are on the opposite side of the line. The line _separates_ one class from the other. 
> If you have more than two features, the decision boundary is not a line, but a (hyper)-plane in the dimension of your feature
> space.
>
{: .callout}

Let's try to draw decision boundaries for our sweets on the graph you've made!

> ## Challenge: Drawing Decision Boundaries 
> 
> Take the plot you've made of the two features, with the sweets on it, and try to draw lines that separate one or 
> more types of sweets from one another. Can you draw lines between all types of sweets so that e.g. all skittles 
> are on one side, and all M&Ms on the other? Are there examples of one type of sweet that end up on the wrong side 
> of the line? 
> Can you draw _straight_ lines that separate out types of sweets, or do they need to curve?
> 
> > ## Solution
> > 
> > Let's look at the image from the last exercise again. If you look closely, you'll see that I've already drawn in 
> > the decision boundaries!
> >
> > <a href="{{ page.root }}/fig/sweets_length_height.jpg"><img src="{{ page.root }}/fig/mixedsweets.jpg" alt="Graph of skittles, two types of M&Ms and jellybeans as a function of length (x-axis) and height (y-axis)." /></a>
> >
> > In this case, the different types of sweets end up in quite distinctly different corners of the graph, so it's fairly 
> > easy to draw straight lines between them. 

So we can draw decision boundaries by hand, but that still doesn't tell us how we can get a computer to do this.
In the next lesson, we're going to meet our first machine learning algorithm: K-nearest neighbour!




> ## Features
>
> Many machine learning algorithms cannot use whatever objects--be they sweets, images or musical songs--directly for the 
> purpose of classification. We therefore need to define and extract *features*, i.e. descriptive summaries of these objects 
> in order to apply these algorithms in practice.  
>
> (Note: in the last part of this lesson, we will learn that there are, in fact, algorithms that _can_ extract features 
> themselves from the raw data, e.g. images. Let's worry about that later, though.)
>  
{: .callout}

> ## Challenge: Defining Features
>
> Let's take a look at the sweets you've received at the start of the class (and have hopefully not eaten yet!).
> If you have not received any sweets, you can also do this exercise with the image from [Lesson 1](). 
>
> Which types of sweets appear very different to you, and which are similar? 
> What are descriptive features that help your eyes distinguish the two types of M&Ms, the skittles and the jellybeans?
> Could you measure them and encode each in a single number per feature and sweet? Are all of the features you write 
> down able to equally distinguish all types of sweets, or are some maybe good at distinguishing skittles from jellybeans, 
> but not skittles from plain M&Ms? 
>
> Note down your features and also describe how you would measure them.
>
> > ## Solution
> > 
> > If you are describing your sweets visually, then you might notice that the different sweets vary in shape and 
> > colour. For example, peanut M&Ms and jellybeans look very different. However, skittles and plain M&Ms actually look 
> > quite similar! We can define some features in that context: length, width, height, for example, or perhaps the ratio of 
> > length to width, and the circumference. You might also notice that M&Ms have an "m" printed on them, while skittles 
> > have an "s" printed on them. That might help distinguish skittles from M&Ms, but not the two types of M&Ms from each 
> > other. Finally, the colours might help distinguish skittles and M&Ms from jellybeans, but note that there are some 
> > overlaps in colours (for example, there are red and blue jellybeans).   
> >
{: .challenge}

Once we've defined some features that we think are helpful in telling the computer how to distinguish the different 
types of sweets, we're ready to record some _training data_. In a training data set, we include measurements for 
all of our relevant features for each sweet, but we also include the type of sweet we measured, i.e. whether it was a 
skittle or one of our two types of M&Ms or a jellybean. 

In machine learning, each object recorded in a training data set is called a *sample*. Each sample belongs to one of 
several *classes* (hence the name classification; in our case the different types of sweets), and the class we record 
for each is called a *label*.   

> ## Challenge: Recording training data
> 
> Let's now record some data in practice! If you don't have the sweets or tools to record data, you can also 
> use [this pre-recorded data set]().
>
> Pick four features you think will do well at describing your sweets, and record the value for ten sweets of 
> each class. 
>
> Hint: you can use any spreadsheet programme to record your data, or you can use [this template](). 
> The template is a `.tsv` file: this is a text file where each row corresponds to a sample, and each column 
> to a feature. Columns are separated by tabs, hence the name (`tsv` stands for "tab-separated values").
>
{: .challenge}




[sweets_data]: 
[sweets_template]: https://github.com/AstroHackWeek/AstroHackWeek2018/tree/master/day3_machine_learning


{% include links.md %}
