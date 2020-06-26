---
title: "Feature Engineering"
teaching: 10
exercises: 20
questions:
- "Can a computer classify candy as easily as a computer does?"
- "How can we derive meaningful summaries of information from our candy to use in classification?"
- "Which features are better suited for classification, which are not?"
objectives:
- "Learners understand that in order to classify objects, many approaches require well-structured information."
- "Learners should be able to identify features from a set of objects and quantify these features."
keypoints:
- "In order to use a computer for classification, we need to summarize the information our eyes see into a few meaningful numbers that the computer can parse."
- "For the current problem of classifying candy, there are a number of features related to the appearance that may be useful."
---

### How do we approach the problem of sorting candy in practice? 

If you look at the picture from [Episode 2](ep2), you probably notice some difference between the different types of candy. Humans tend to be very good at classification. Many of you can probably take one look at the picture and immediately recognize that some candy are different than others. You can make categories of different candy based on their appearance, and crucially, you probably don't even have to articulate *what* exactly is different about one sweet compared to another: you can just tell. This, of course, extends past the visual domain: I could give you different types of candy to touch, and you could tell the difference by touch, too.

Our ability to classify objects is trained from early childhood, and rests upon many hours of learning and experience. 
A computer, on the other hand, generally operates according to *rules* and *models*. We need to tell it what to do!

In the case of machine learning, we will tell the computer to optimize a *model*, a mathematical equation or set of mathematical equations that describe how to put our candy into different categories. We'll talk more about some exact mathematical equations in a later lesson. For now, it's important to understand that these mathematical equations have *free parameters*, and that we don't know from first principles how exactly to write them down. 

Along with the mathematical description of the algorithm, we will give the 
computer a second ingredient: the _training set_. The training set are data points (in our case, individual pieces of candy) for which we already know whether they're skittles, M&M's, or jellybeans.

### How do we combine our candy, which are real, physical objects, with the mathematical equations of our model?

We need to make explicit what our eyes and brains do automatically: find descriptors of our objects that might separate them 
into different classes. In machine learning, we call these descriptors *features*: they are generally _summaries_ of our information 
in some way. 

> ## Features
>
> Many machine learning algorithms cannot use objects directly --be they candy, images or musical songs-- for the 
> purpose of classification. We therefore need to define and extract *features*, i.e. descriptive summaries of these objects 
> in order to apply these algorithms in practice.  
>
> (Note: in the last part of this lesson, we will learn that there are, in fact, algorithms that _can_ extract features 
> themselves from the raw data, e.g. images. Let's worry about that later, though.)
>  
{: .callout}

> ## Challenge: Defining Features
>
> Let's take a look at the candy you've received at the start of the class (and have hopefully not eaten yet!).
> If you have not received any candy, you can also do this exercise with the image from [Episode 2](ep2). 
>
> Identify features of your candy and describe how you would measure them. Here are some helpful questions to get you started:
>
> * Which types of candy appear very different to you, and which are similar? 
> * What are descriptive features that help your eyes distinguish the two types of M&Ms, the skittles, and the jellybeans?
> * Could you measure these features and encode each in a single number per feature and sweet? 
> * Are all of the features you write 
> down able to equally distinguish all types of candy, or are some maybe good at distinguishing skittles from jellybeans, 
> but not skittles from plain M&Ms? 
>
> > ## Solution
> > 
> > If you are describing your candy visually, then you might notice that the different candy vary in shape and 
> > colour. For example, peanut M&Ms and jellybeans look very different. However, skittles and plain M&Ms actually look 
> > quite similar! We can define some features in that context: length, width, height, for example, or perhaps the ratio of 
> > length to width, and the circumference. You might also notice that M&Ms have an "m" printed on them, while skittles 
> > have an "s" printed on them. That might help distinguish skittles from M&Ms, but not the two types of M&Ms from each 
> > other. Finally, the colours might help distinguish skittles and M&Ms from jellybeans, but note that there are some 
> > overlaps in colours (for example, there are red and blue jellybeans).   
> > 
> > {: .output}
> {: .solution}
{: .challenge}

Once we've defined some features that we think are helpful in telling the computer how to distinguish the different 
types of candy, we're ready to record some _training data_. In a training data set, we include measurements for 
all of our relevant features for each sweet, but we also include the type of sweet we measured, i.e. whether it was a 
skittle or one of our two types of M&Ms or a jellybean. 

In machine learning, each object recorded in a training data set is called a *sample*. Each sample belongs to one of 
several *classes* (hence the name classification; in our case the different types of candy), and the class we record 
for each is called a *label*.   

> ## Challenge: Recording training data
> 
> Let's now record some data in practice! If you don't have the candy or tools to record data, you can also 
> use [this pre-recorded data set](candy_data) (when you get to that page, right-click on the file `candy_data_200611.tsv`, 
> and choose "download linked file" or the equivalent).
>
> Pick four features you think will do well at describing your candy, and record the value for ten candy of 
> each class. 
>
> Hint: you can use any spreadsheet programme to record your data, or you can use [this template](candy_template). 
> The template is a `.tsv` file: this is a text file where each row corresponds to a sample, and each column 
> to a feature. Columns are separated by tabs, hence the name (`tsv` stands for "tab-separated values").
>
{: .challenge}



[ep2]: https://huppenkothen.org/machine-learning-tutorial/02-ProblemSetUp/index.html
[candy_data]: https://github.com/dhuppenkothen/machine-learning-tutorial/tree/gh-pages/data 
[candy_template]: https://github.com/AstroHackWeek/AstroHackWeek2018/tree/master/day3_machine_learning


{% include links.md %}
