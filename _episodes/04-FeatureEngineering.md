---
title: "Feature Engineering"
teaching: 0
exercises: 0
questions:
- "Can a computer classify sweets as easily as a computer does?"
- "How can we derive meaningful summaries of information from our sweets to use in classification?"
- "Which features are better suited for classification, which are not?"
objectives:
- "Learners understand that in order to classify objects, many approaches require well-structured information."
- "Learners can generate features from a set of objects."
keypoints:
- "In order to use a computer for classification, we need to summarize the information our eyes see into a few meaningful numbers that the computer can parse."
- "For the current problem of classifying sweets, there are a number of features related to the appearance that may be useful."
---

So how do we approach the problem of sorting sweets in practice? If you look at the picture from [Lesson 1](), you probably notice some difference between the different types of sweets. Humans tend to be very good at classification. Many of you can probably take one look at the picture and immediately recognize that some sweets are different than others. You can make categories of different sweets based on their appearance, and crucially, you probably don't even have to articulate *what* exactly is different about one sweet compared to another: you can just tell. This, of course, extends past the visual domain: I could give you different types of sweets to touch, and you could tell the difference by touch, too.

So what do we do? Can we throw the sweets into our robot, and the robot will just know where to sort the sweets? 
Well, no. Our ability to classify objects is trained from early childhood, and rests upon many hours of learning and experience. 
A computer, being a machine, generally operates accordingt to *rules* and *models*: we need to tell it what to do!
In the case of machine learning, we will tell it to optimize a *model*: this is a mathematical equation or set of 
mathematical equations that describe how to put our sweets into different categories. These mathematical equations 
have *free parameters*: we don't know from first principles how exactly to write them down. What exactly those mathematical 
equations are, we'll talk about in a later lesson, and see some examples from the numerous algorithms that exist. Along with the 
mathematical description of the algorithm, we will give the 
computer a second ingredient: sweets for whom we know whether they're skittles or M&Ms or jellybeans, because you and 
your friends like all of those so much, you've always got a couple of small bags at home. But how do we combine our sweets, 
which are real, physical objects, with the mathematical equations of our model?

We need to make explicit what our eyes and brains do automatically: find descriptors of our objects that might separate them 
into different classes. In machine learning, we call these descriptors *features*: they are generally _summaries_ of our information 
in some way. 

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
> >
> >
> >
{: .challenge}


This tutorial owes a lot to previous machine learning tutorials at [Astro Hack Week][ahw],
especially Iain Murray's and Gilles Louppe's [tutorials](ml2018) and Adrian Price-Whelan and David W Hogg's [tutorial](ml2017). It is also a follow-up on Gwen Eadie's [paper](mmpaper) on teaching Bayesian statistics with 
M&Ms. Because what better way to learn statistics and machine learning through chocolate?


[survey]: 
[ml2018]: https://github.com/AstroHackWeek/AstroHackWeek2018/tree/master/day3_machine_learning
[ml2017]: https://github.com/AstroHackWeek/AstroHackWeek2017/tree/master/day1
[mmpaper]: https://www.tandfonline.com/doi/full/10.1080/10691898.2019.1604106


{% include links.md %}
