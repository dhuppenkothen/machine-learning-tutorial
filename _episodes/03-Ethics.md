---
title: "Incorporating Ethics into your Machine Learning Project"
teaching: 20
exercises: 20
questions:
- "Is the project I am planning to do an ethical application of machine learning?"
- "Should I do this project in the first place?"
- "If I do this project, who might be harmed by it?"
- "How could the methods/tools/software I'm developing be misused for purposes I consider unethical?"
objectives:
- "Learner's should be able to distinguish between algorithms with explicit instructions and machine learning algorithms"
- "Learners understand that ethical considerations are integral to all machine learning applications"
- "Learners are able to critically question their own motives and approach to solving their problem."
- "Learners can lay out worst-case scenarios in order to design effective counter-measures to mis-use."
keypoints:
- "All machine learning projects should take ethical considerations into account during the planning stages and throughout the completion of the project"
- "Not all machine learning projects are ethical: sometimes, the right choice is to abandon a project."
- "Projects might be used for data sets or applications that the developers never intended it for. It is important to envision potential uses that might turn out to be harmful."
---

Let's start by taking a look at a more formal definition of machine learning:

> ## Machine Learning Definition
>
> Machine learning (ML) is the scientific study of algorithms and statistical models 
> that computer systems use to perform a specific task without using explicit instructions, 
> relying on patterns and inference instead. It is seen as a subset of artificial intelligence. 
>
{: .callout}

It's worth taking a moment to unpack this statement in more detail. In machine learning, a 
computer performs a specific task without using explicit instructions, using algorithms and 
statistical models. Often, that task is 
_prediction_: given some information, the machine learning algorithm attempts to predict an outcome. For example, Facebook may 
know about your age, where you live and what you usually post about, and use this information to predict whether you're likely to engage (e.g., click on a link) with a specific advertisement. As another example, your bank might know 
how much money is in your bank account and how reliably you pay your bills, and may use that to predict whether they should give you a loan. 

Generally, computers operate on specific and rigid mathematical principles: they require explicit 
instructions to operate, and the user (i.e., you) must tell the computer these instructions or rules. For example, you might write a rule that someone who has been posting about 
children at least once a day should be shown an advertisement for toys. However, this simple rule may miss a lot of context. What if the person has in fact been posting about how their neighbour's children are always 
loud? Perhaps they should be shown advertisements for noise-cancelling headphones instead?
If you write down rule-based logic, you have to make sure your logic includes any and all possible
outcomes and contexts, even when you _don't know what they are_. This is obviously difficult!

Machine learning goes a very different route. It aims to solve that problem by _inferring_, _estimating_, or _approximating_ the rules 
and patterns from a set of data points. For example, you might take a whole lot of social media posts
about children from different people, and show each person different advertisements. If you do that many, many times, 
you might be able to infer from the data that posters who use words like "son", "daughter", and "happy 
birthday", should be shown toys, and posters who use the words "children" and "noise" should be 
shown advertisements for noise-cancelling headphones. You might also learn that there's a third group 
of posters, that you hadn't even thought of before, who post a lot about children together with 
words like "school" and "classroom". Those are teachers, and perhaps you'd want to show them 
advertisements of school supplies like pens and erasers. In this way, the exact rules or instructions for classifying have not been supplied, but patterns in the data have been found.

How exactly we can make a computer find and describe patterns and rules in our data (i.e. the algorithms 
and statistical models part of the definition above), we'll learn about a little bit later in this 
lesson. But here's the important part to keep in mind for now: *machine learning algorithms learn patterns 
from data, and use these patterns to predict future outcomes*.


> ## Challenge
>
> Earlier in this tutorial, you thought about a place where your life has been affected by 
> computers making decisions for you. First on your own, then with your neighbour, think about and 
> discuss whether that decision-making was a positive or negative experience for you. Did the computer
> make a decision in a way that you felt was fair and transparent? 
{: .challenge}

Our lives are affected every day by predictions that are made from algorithms. What news you see on social media, what advertisements 
you are shown, what search results come up in a search engine, and major life decisions (e.g. whether you may apply for a mortgage to buy 
a house, whether you get a job offer) is now decided upon either wholly or in part by computers, and particularly by machine learning 
algorithms. 

As machine learning is used in more and more circumstances, we, as a society, are also becoming aware of the ways technology and algorithms can be *misused* or in ways that lead to unethical and harmful outcomes. 
Even if you are a student in a field that doesn't directly involve data from humans (like my own field, astronomy), it is important 
for you to know how your own life is affected by these decision making processes. You might also one day end up in a field where 
you may programme algorithms that do make decisions affecting human lives, so it is important to know how to tread with care.

> ## Challenge
>
> Can you think of situations where an algorithm should *not* be used to 
> support human decision making? 
>
>
{: .challenge} 


You might think that there's a clear distinction: in some fields, algorithms make decisions that directly affect human lives, for 
example in healthcare, or during college admissions. In other fields, that connection may be less direct. Perhaps you're providing 
a service that recognizes faces in images. This is a service that many people find useful. For example, modern photo apps can recognize 
faces in photos, and can also recognize _different_ faces, grouping them together by person. This makes it easy to look for, for example, 
photos of my sister. These apps often employ machine learning: taking a photo, an algorithm predicts whether there's a face in the 
photo, and it also compares it to groups of similar faces it has already seen in your library. These algorithms are trained using large 
sets of images that someone collected, and then annotated.

You might think that this sounds great --- if you're training your algorithm on lots of images, then it should be pretty good at finding faces, shouldn't it? What ethical questions are there to think about? How could anyone 
possibly mess this up?

This could be messed up, as it turns out, in very many different ways, and ethical questions are indeed important to address.

All machine learning algorithms need *training data*; example data where you know exactly what they are and how they are, e.g., categorized. You then show to the algorithm these data, so that when it sees new, unknown examples, it knows what to predict and classify.

Where do we get that training data from? By far the most popular, standard data set used to train image classifiers is called [ImageNet](). The creators of this data set (academics who needed data to help test computer vision algorithms) collected a large number of photos, and then asked 
people on the internet, through crowdsourcing, to look at each image in turn and attach labels to each category. Originally, this was meant 
to classify objects, not people, but there are people in the data set, and they are labelled. Those labels, however, are not objective:
any categorization requires a definition of those categories, and when it comes to people, those categories will be influenced by the 
creators' social and political knowledge and views (and often, also, their ignorance). And so the ImageNet categorization contains a number of 
categories that one would consider neutral, like "child" or "doctor", but it also contains offensive descriptions and racial slurs (for a great 
investigation of the ImageNet data set, read [Excavating AI: The Politics of Images in Machine Learning Training Sets](excavatingai). 

These categories are given to a large set of people, who then apply these categories to the images they see. They, too, will bring along their 
biases and prejudices, which in turn leads to images being labelled in ways that are biased. And this is the information we feed into the 
algorithm, which cannot do anything but learn from the only information source it has. So when you pass in an image to a machine learning algorithm to predict and categorize, you might end up with a 
prediction that says "doctor" if you gave it the photo of a white man, and "criminal" if you gave it the image of a black man. 
The machine learning model doesn't predict anything that approximates a fundamental truth, instead it contains biases in the labelling, where the people who labelled the data set encoded their prejudices and biases.

The categorization, and the way it is applied to images, is only one problem in this context. ImageNet has many photos of white people, and 
very few photos of black people. Because algorithms learn from the examples they are presented with, and learn better with more examples, 
the algorithm will be worse at predicting the presence of a black person in an image, compared to a white person. So, what you put into 
your training data set, and in which numbers, directly matters to the outcomes.   

The creators of the ImageNet data set have since deactivated their classification scheme for people, but other data sets that have sprung up
are similarly troubled. Creating a training data set involves neatly stuffing all of the complex, messy reality we live in into a 
countable set of simple (often one-word) descriptive labels. The "Japanese Female Facial Expression" database classifies all of human emotion 
into seven categories. Can we neatly put all of human emotion into these categories? I don't know. This dataset specifically uses Japanese 
models. But different cultures might express emotions or reactions differently. For example, nodding means agreement in some cultures, and 
disagreement in others. The authors have made an implicit assumption about the complexity and diversity of human emotions, and anyone using 
it for any purpose other than predicting emotions in images of Japanese women assumes that those expressions are universally applicable, when 
they might not be _representative_ of the wider world.

So far, we've just talked about the training data set. You might say that we're fine, as long as we have an unbiased training data set (which, I would argue, is impossible). But even if you could manage a training set that is truly representative and unbiased, you're still not guaranteed to not mess it up. Machine learning encompasses a whole range of 
different algorithms, and they make a lot of different assumptions about what the underlying data is like. For example, some might assume 
that your data follows a particular statistical distribution, or it might assume that you have an equal number of examples for all categories.

You also have to define a _metric_ to tell you how well your machine learning algorithm is doing. This is usually a number that's higher if it's doing a good 
job, and lower if it's doing a bad job. 
While that may sound simple, it requires a *lot* of thought about what a "good job" actually means. The simplest metric is called _accuracy_, 
and it essentially counts the number of examples that were predicted correctly (given a training data set where you know what the outcome 
ought to be), and divides that number by the total number of examples it has seen. So if you classified 1000 images of faces, and got 900 right, 
then your accuracy is 900/1000 = 0.9. This may seem straightforward and fair, but it's not. 
Imagine that in your data set of 1000 photos are 290 photos of white people, 10 photos of black people, and 700 photos of things other than people.
Let's look at these categories in turn. Imagine the algorithm didn't learn anything about people at all, and never predicted the label "person" 
for any of them, it would mis-label 300 examples. That would give you an accuracy of 700/1000 = 0.7. Most people working on machine learning algorithms would argue that that's pretty bad. Mis-labelling one in three images is pretty terrible performance, so you might go back to the drawing board and 
try something else. That situation doesn't change very much if, for example, we just look at the white people category. 290/1000 is still a lot, so 
the accuracy would still tell us we need to change our approach. However, imagine that because it sees so few images of black people, it now 
classifies none of the images of black people as "person". Because you only have ten examples of that single category, your accuracy will be 
990/1000 = 0.99. That's pretty high. A lot of machine learning people would be excited to see that, and argue that their system is working really 
well. _But you've just created a system that is incapable of identifying an entire group of people in images!_ That's bad. 


There are other types of metrics you can use that provide a more nuanced view, and we'll talk about those in more detail in a later episode.
The key point here is: the choice of algorithm, and the choice of evaluation metric, _also_ encode assumptions that the creator makes about the 
world, and consequently their biases and prejudices. In this case, we assumed that optimizing for the highest number of correct classifications 
is good enough.

So here's the lesson you should take away from the story above:

> ## Algorithms & Data
>
> Algorithms are designed by _people_, and people embed biases in algorithms,
> data collection and the interpretation of results.
>
{: .callout}

This is important, because there is a strong narrative around the use of data and algorithms as more objective and as unbiased. While it is true that 
when done really carefully, these can occasionally be *less biased*, there are now many examples where researchers and technologists have claimed 
that their machine learning system would make a process unbiased, only to be shown horribly wrong.
This view of algorithms and data sets as objective representations of the world, goes hand in hand with a 
concept that [Meredith Broussard]() calls *technochauvinism* in her book, [_Artificial Unintelligence_](). This term refers to a mindset that says 
that algorithms are superior to human judgment, and argues that technology is always the best strategy. Before you even start building, it's worth 
thinking about whether technology is really the right solution to the problem you're trying to solve, especially if that problem is a social one.
Especially when in many of these examples, the predictions the system produces have shown to _not_ be unbiased, but rather exacerbate existing 
inequalities and discrimination.

That, in itself, should be enough to make you pause, because hopefully none of us set out to build biased or sexist or racist or other -ist systems. 
That means we have to carefully evaluate what the _effect_ of our algorithm or machine learning system is when it is used in the real world. 
For example, let's take our face detection example above. If your company implements and sells that system to a photo service, the photo service 
might get lots of angry calls, because while I might be able to search through my library by face (e.g. to find all photos with my sister in it), a 
black customer can't. That's infuriating and racist. 
What if your company now sells that same technology to a company that builds self-driving cars, which wants to use it to identify pedestrians automatically.
Suddenly, the biases in your algorithm become life-threatening, and they become life-threatening to an already oppressed group of people, because if they 
won't be recognized as people by the car, the car might decide not to stop when it sees a black pedestrian. 
There are many different examples of companies and academics producing machine learning systems that are harmful, either through negligence or purposeful. 

Finally, when creating a system, it's worth considering whether it might be misused by someone setting out to purposefully harm or manipulate others. 
When researchers first created the large-scale natural language model GTP-2, they initially did not release the full model, because they were afraid that 
it might be used to generate realistic-sounding text that could be used in malicious ways. A debate ensued whether that was a reasonable approach. 
Facial recognition as a whole has become the subject of an important debate around privacy and the ability for companies and governments to 
track citizens. 

> ## Ethical Concerns in Machine Learning
>
> When considering to build or implement a machine learning system for a 
> particular problem, consider a wide range of possible ethical concerns 
> that could affect the usage and outcomes you put into the world 
> with that system.
>
> This involves concerns around your own limited perspective and the biases and prejudices
> you bring into the design and development process, the data sets you use for training, their 
> representativeness and categorization, the associated privacy and data rights of 
> the people who will be impacted by the predictions the system generates, and the abiity 
> to use the system for purposeful harm and manipulation. 
>
>
{: .callout}
 

> ## Challenge
>
> Think about our candy-sorting task. Let's think down the ethical implication of letting a computer/robot help 
> us sort through the different types of candy. 
> * Is this something a computer should assist with? Why? Why not?
> * Where can you see the potential for the process to go wrong? 
> * Could anyone be harmed by the process? If so, how? 
> 
> Take a look at our mix of different candies. 
> * Is our training data set balanced (i.e. are there similar quantities of all kinds of candy)? Why would this matter? 
> * In the last lesson, you thought about _features_. Could we mess up those features in a way that might lead to harm?
>
{: .challenge}

[excavatingai]: https://www.excavating.ai

{% include links.md %}
