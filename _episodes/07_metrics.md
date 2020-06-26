---
title: "Model Evaluation"
teaching: 20
exercises: 10
questions:
- "How can we evaluate whether our model does a good job?"
- "How can we define what a 'good job' means to us when setting up a machine learning model to solve a problem?"
objectives:
- "Learners know how to split training data into training and test sets."
- "Learners can state the differences between true positives, true negatives, false positives, false negatives"
- "Learners can use these definitions to define precision, recall, specificity, and F1 score, and understand when they are relevant."
keypoints:
- "Splitting the data into a training set and a test set we hold back until the end of the training phase helps us evaluate the performance of our machine learning model."
- "The choice of evaluation metrics depends on the problem to be solved using machine learning."
- "Different evaluation metrics optimize for different outcomes, and should be used in different circumstances."
- "Evaluation metrics can be used jointly or be combined to give a fuller picture of performance."
---

**How do we decide whether our model does a good job?**

This is one of the key questions when training a machine learning algorithm. So far, we've looked at some individual examples on 
paper. But how do we know if our algorithm does a good job overall? We *could* try classifying some new, unknown sample (for example, 
a handful of candy fished out of the giant box we mixed together). However, even though the algorithm might predict a class, we won't know whether it predicted the _correct_ class. 

### Train-Test Splits

In order to test whether the model does a good job, we can split our training data set--the data for which we know the correct 
outcomes--into two subsets. We will use one of these subsets to train the algorithm (this will be our new training data set), and the 
other to test the performance of the algorithm (the test set). For the test set, we will pretend that we 
don't know what the correct classes are, and let our newly trained system predict the classes. We can then compare the predicted
classes to the known true classes, which will give us a sense of how often our model predicts these classes correctly.  


> ## Training + Test Data
>
> Split the your data set with known labels into two separate subsets: one you will use for training the model, the other you will
> hold back and use to measure how well the model predicts unknown cases. 
> 
> In order to get the most accurate and unbiased performance estimate, it is advisable to keep the test data set aside until the very end
> of the process, when you have tuned your parameters, extracted all the relevant features, and are ready to use the model in practice.
>
{: .callout}

Keeping your test data set hidden until the end might seem counterintuitive at this stage: how will you know whether to use a different 
algorithm, or different parameters in your algorithm, if you can't test its performance throughout the process? Fear not, you'll learn a 
way to do that a little later in this lesson! 

> ## Challenge
>
> There are many ways to split up a data set into a training and a test set. For example, you could split it clean in half, and use 
> the first half for training, the second half for classification. Or you could _randomly_ select half of the data points for training, 
> and the other half for testing. You could also use more (or less) than half to train. 
> * Can you think of situations where you would do 
> a random split, rather than simply splitting the data set in the middle? 
> * Can you think of situations where using more or less than 50% 
> for training/testing might be useful? 
>
> Discuss with a partner.
>
> > ## Solution
> > 
> > Data sets are often not recorded randomly: perhaps when you built your training data, you measured all the peanut M&Ms first, 
> > then the skittles, and so forth. Or perhaps your friend sorted all the candies by size, and then measured them. 
> > If you have a training data set like that, and you split it in the middle, you might end up in a situation where your training 
> > data set contains all of the peanut M&Ms but none of the plain M&Ms. Because the training data set doesn't contain any plain M&Ms, 
> > it won't know how to classify these in the test data set. Conversely, you won't know whether your model did well on the peanut M&Ms, 
> > because there are none in the test data. Therefore, it is good practice to _randomize_ your data (or subsample randomly when generating training/test data).
> > 
> > There is also a balance to be found between the amount of training and the amount of test data. You want to show the algorithm 
> > as much training data as possible, so that the resulting model has seen many different variations within each class.
> > On the other hand, if your test data set is too small, it might not contain examples from all of the classes, or have too few examples 
> > to reliably establish performance. Usually, people try to use somewhat more data for training than testing: 70% training/30% test data is 
> > a common split, but the details often depend on the problem at hand (and the overall amount of training data that's available).
> > {: .output} 
> {: .solution}
{: .challenge}

### Evaluation Metrics

Above, we've stated that we want to figure out whether our model does a good job. One key element of building a machine learning model 
is defining what "good job" means to us. This is usually done via _evaluation metrics_, a numerical summary of the results of the 
classification on the test data set, often defined such that a large number means the algorithm does a good job, whereas a small number 
indicates a bad model. 

You might think that there is a universally accepted standard way to estimate performance, but 
the actual measure you want to use is deeply problem-dependent. Let's illustrate with an example. 

Imagine you have a friend who is 
very allergic to peanuts, who is coming to your party. Your housemate also has a couple of friends coming who belong to the Skittles 
Appreciation Society and absolutely abhor M&Ms. While you don't want to offend those friends, more importantly you don't want to accidentally 
give your friend with the peanut allergy a candy containing peanuts. As a result, you decide that you need to prioritize correctly identifying peanut M&Ms over all other classes, which means that you need an evaluation 
metric that reflects that priority.


> ## Evaluation Metrics Need to Reflect Priorities
> 
> Think carefully about the metrics you use for evaluating performance, and ensure that 
> these metrics reflect the goals you have for the classification outcomes. 
>
{: .callout}

> ## Challenge
>
> Can you think of ways you would measure performance of a machine learning model? 
> Can you imagine what kind of measure one might want to define for a problem where we want 
> to minimize the chance of inadvertently classifying a peanut M&M as something else?
>
{: .challenge}

Let's talk about some common evaluation metrics. The most common (and most straightforward) metric is called _accuracy_. 
For this metric, you look at the predicted classes on your test data set, and the known correct classes for the same set of 
examples. Count all the samples that were correctly identified by the model, also called the _true positives_, and 
divide them by the total number of samples in the test data set. The resulting fraction is the accuracy score: 

$$
\mathrm{accuracy} = \frac{\mathrm{True\, Positives}}{\mathrm{all\, samples}}
$$

> ## Challenge
>
> Can you think of situations where the accuracy score is a good measure of successful classification? 
> Are there situations where perhaps the accuracy score might not be the best option?
>
> > ## Solution
> >
> > The accuracy score gives you a reasonably good measure of accuracy if all your classes are _balanced_, that 
> > is, if you have roughly the same number of training examples in all classes. If you don't, then 
> > the accuracy will preferentially measure the performance on the most common class(es), at the detriment of performance
> > for the classes with fewer examples.
> >
> >
> > To see how this works, let's take a look at an example: in our introduction, we mentioned 
> > that we bought the same amount of different candies by _weight_. However, because skittles, plain M&Ms, and jellybeans
> > are much lighter than peanut M&Ms, we might have much fewer peanut M&Ms in our data set by *number*.
> > Let's assume an extreme case: We have only 5% peanut M&Ms, everything else is other types of candy. If we set up a 
> > binary classification (peanut M&Ms versus others), then our examples are severely _imbalanced_: out of a 100 candies, 
> > only five are peanut M&Ms. What would happen if we set up an algorithm that just classified *everything* as *not* a 
> > peanut M&M? Out of a hundred examples, it would only get five wrong, i.e. the accuracy is 95/100 = 0.95. 
> > Only five! you might say, and conclude that it's a pretty good classifier. However, it's not a good classifier if the peanut M&Ms are 
> > what you really care about. In this case, accuracy gives you a very misleading measure of performance.
> > {: .output}
> {: .solution}
{: .challenge} 

> ## Use Accuracy for Balanced Data Sets
>
> Accuracy works best as an evaluation metric if the number of examples for each class in the training data set is 
> balanced. For very imbalanced data sets, accuracy might give a misleading high score, even though a large fraction 
> of examples in the less common classes are misclassified.
>
{: .callout}

Accuracy is probably not the best metric to use if we want to make sure that our friend doesn't accidentally eat peanut M&Ms. So, what other options are there? 

In the example problem, we're much more interested in the number of _false negatives_ ---  These are examples that are actually peanut M&Ms, but classified as other candy. To complete the number of possible categories a classified example could fall into, we also need to define 
_true negatives_, the total number of other candy that was correctly identified as such, and the _false positives_, the number 
of other candies falsely identified as peanut M&Ms. Take together, this yields the following table:

|             | peanut M&M     | other candy    |
|-------------|----------------|----------------|
| peanut M&M  | True positive  | False positive |
| other candy | False negative | True negative  |

Here, the columns denote the true class of a sample, and the rows denote the class returned by a machine learning model.  

> ## Challenge
>
> Which of the quantities defined above is the most relevant to avoiding accidentally giving our friend
> peanut M&Ms? Discuss with a partner
> 
> > ## Solution
> > 
> > We're mainly concerned with accidentally misclassifying peanut M&Ms as something else, so the number 
> > we're most interested in is the _false negative_ rate. 
> > 
> > {: .output}}
> {: .solution}
{: .challenge}

*Special Note:* that it is often impossible to optimize for all possible cases. That is, if we optimize a machine learning model of minimizing false negatives, we might get more false positives as a result, and the other way around.

We can now define a number of other metrics based on these quantities, each of which are useful for different circumstances. 

The _precision_ is a measure that tells us how many of the candies we identified as peanut M&Ms are actually peanut M&Ms. That is, 
it compares the number of true positives--the number of candies correctly identified as M&Ms--and compares it to the total number 
of candies identified as M&Ms:

$$
\mathrm{precision} = \frac{\mathrm{true\, positives}}{\mathrm{true\, positives} + \mathrm{false\, positives}}
$$

This measure is especially useful if you're interested in making sure that our sample is not _contaminated_. 
Perhaps you have a friend who only eats peanut M&Ms, and gets really annoyed if they're given anything else. In this case, you 
might want to consider precision as a score.

Conversely, the _recall_ is a measure of the fraction of actual peanut M&Ms identified as such. It measures the number of true positives 
and compares it to the number of true positives and false negatives:

$$
\mathrm{recall} = \frac{\mathrm{true\, positives}}{\mathrm{true\, positives} + \mathrm{false\, negatives}}
$$

This measure is particularly useful for minimizing false negatives, that is, for making sure that all of the peanut M&Ms are correctly 
identified as such, and not mistakenly identified as other types of candy. This is the best measure to use if you want to make sure 
your friend doesn't end up in hospital with an allergic reaction!

If we're particularly interested in the other types of candy, we could calculate the _specificity_, i.e. the inverse of the recall. This 
measures the proportion of other candy identified as such, by looking at the number of true negatives, and comparing it to the combination 
of true negatives and false positives:

$$
\mathrm{specificity} = \frac{\mathrm{true\, negatives}}{\mathrm{true\, negatives} + \mathrm{false\, positives}}
$$

Of course, in reality, we might be interested in multiple (perhaps competing) different priorities. You can always evaluate your 
algorithm using more than one single score (and perhaps weight those scores by importance: you care more about your friend not ending 
up in hospital than you care about your friends from the Skittles Appreciation Society only eating skittles). It is also possible 
to _combine_ scores into a new one. A popular version is the _F1 score_, which combines precision and recall into one metric by taking 
the _harmonic mean_ between them:

$$
F1 = \frac{2\, \mathrm{precision}\, \mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}
$$

This score is useful when you are looking for a metric that balances a little more effectively between making sure there are no peanut 
M&Ms in your other candy, but it's also important to not get too many false positives.



{% include links.md %}
