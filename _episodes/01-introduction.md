---
title: "Introduction"
teaching: 0
exercises: 0
questions:
- "What is machine learning?"
- "Where is machine learning useful?"
objectives:
- "Learners should have a basic working understanding of machine learning."
- "Learners should have a basic understanding of the problems where machine learning may be applied."
keypoints:
- "Machine learning is the study of algorithms to learn patterns from data."
- "Machine learning is ubiquitous in the modern world."
---

**How do we make sense of large data sets? How do we distinguish different groups of objects?**

In our modern world, there are many problems where one might wish to distinguish different types of 
objects. For example, an e-mail provider might wish to distinguish spam e-mails from e-mails that are 
not spam, so that its customers aren't inundated with messages they don't want to receive. 
Different people have different handwriting, but the post needs to make sure that an "a" is always 
recognized as an "a", so that addresses are parsed correctly and letters get sent to the right addresses.
Maybe you're an amateur photographer like me and have photographed a lot of cats and dogs. You now want a 
way to distinguish your cat photos from your dog photos, perhaps because you'd like to print out some cat 
photos for your cat-loving friend. 

All these are problems one *could* do by hand: you could sit down and sift through all of your photos 
and decide based on each one whether there's a cat or a dog in the photo. Humans are extremely good at 
_classification_: many of you will be able to look at a photo of a four-legged animal and tell whether it's a 
cat or a dog, provided that you've seen both at least once before. You'll probably also be able to note some 
specific differences between both (maybe the shape of the head, or the ears, or the presence of whiskers) that 
help you make that distinction. Once you've seen a picture of a dog or a cat, you'll probably also be able to 
abstract from that picture to a new one. If on that new picture, the cat or dog is shown from a different angle, 
chances are you'll still be able to extrapolate and tell me what kind of animal it is.

However, you might not _want_ to sit down and sift through all of your photos by hand. If you've got many hundreds 
or thousands of photos (perhpas you're a photographer who specializes in pets), that might become very tedious, 
time-consuming and boring. It might also not _scale_ very well: each day, the US Postal Service processes 
something like 187 million pieces of mail! Imagine if each of those addresses had to be read and typed in by a 
postal worker by hand. Our hypothetical pet photographer might be able to get through all of their photos, 
but that means that every other pet photographer needs to do the same. Wouldn't it be cool if we could 
make a computer help us classify those images of cats and dogs, or help process the mail?

This is where **machine learning** comes in. Machine learning is a subfield of computer science concerned with 
methods and ways that we can teach computers to learn patterns from data. Often, the goal is **classification**: 
distinguishing photos of cats from those of dogs, looking for similar songs to the one you've just been 
listening to, or even providing medical diagnoses. In these types of problems, we are trying to predict 
whether a given example belongs to one of several discrete _classes_. In this tutorial, we will focus on 
problems with this kind of structure. It's worth nothing, however, that machine learning can also solve problems 
where the output variable is _continuous_, where we are trying to perhaps predict the temperature in Seattle 
tomorrow from past weather data.  

> ## Challenge
>
> Take a moment to think about where you might have encountered classification problems in your 
> daily life. Can you think of places where your life is affected by computers making decisions on 
> your behalf?
>
{: .challenge}


This tutorial owes a lot to previous machine learning tutorials at [Astro Hack Week][ahw],
especially Iain Murray's and Gilles Louppe's [tutorials](ml2018) and Adrian Price-Whelan and David W Hogg's [tutorial](ml2017). It is also a follow-up on Gwen Eadie's [paper](mmpaper) on teaching Bayesian statistics with 
M&Ms. Because what better way to learn statistics and machine learning through chocolate?


[ml2018]: https://github.com/AstroHackWeek/AstroHackWeek2018/tree/master/day3_machine_learning
[ml2017]: https://github.com/AstroHackWeek/AstroHackWeek2017/tree/master/day1
[mmpaper]: https://www.tandfonline.com/doi/full/10.1080/10691898.2019.1604106


{% include links.md %}
