# 365VDataScience

PROBABILITY FOR STATISTICS AND DATA SCIENCE

Introduction to Probability: Cheat Sheet

Probability Formula | Sample Space | Expected Values | Complements

![## Image Analysis: 77951dd7bc738e15393d2bab69422a6f1cbf03550306f96ca5efebdf4b0a7b34.jpg

**Conceptual Understanding:**
This image conceptually illustrates the core process of deriving machine learning insights from a dataset. The main purpose is to highlight that while a 'Dataset' is the starting point and 'ML Insight' is the desired outcome, the journey between the two is characterized by a significant, often complex, and possibly opaque transformation or processing stage, symbolized by the question mark. It conveys the idea that achieving valuable insights from data is not a trivial task and involves navigating through uncertainties or intricate procedures.

**Content Interpretation:**
The image depicts a conceptual pipeline for generating Machine Learning (ML) insights. It shows three main stages: the initial 'Dataset' as the input, an intermediate '?' stage representing an unknown, complex, or challenging transformation process, and the final 'ML Insight' as the output. The central diamond shape filled with question marks signifies the 'black box' nature or the inherent complexities, decisions, and challenges involved in turning raw data into meaningful ML insights. This could involve steps like data cleaning, feature engineering, model selection, training, or evaluation, which are not explicitly detailed but are collectively acknowledged as non-trivial. The sequential flow indicates a progression from data acquisition to actionable knowledge.

**Key Insights:**
The primary takeaway from this image is that the generation of 'ML Insight' from a raw 'Dataset' is not a simple, direct conversion. Instead, there is a significant, often complex, and potentially uncertain intermediate process, represented by the prominent '?' diamond. This highlights the reality of data science and machine learning workflows, where the steps between data and insight (such as data preprocessing, feature engineering, model training, and validation) are critical, can be challenging to define or execute, and often involve iterative experimentation and decision-making. The diagram suggests that this intermediate phase is a 'black box' or a point of critical inquiry and effort in the overall process.

**Document Context:**
Given the document context 'Section: 365VDataScience', this image likely serves as an introductory diagram to emphasize the journey from raw data to valuable machine learning insights, particularly highlighting that the intermediate steps are not always straightforward or fully defined. It sets the stage for discussions about the challenges, methodologies, and intricacies involved in data science and machine learning pipelines, such as data preprocessing, model development, or the interpretability of ML models. The image underscores the idea that gaining ML insights from a dataset is not a simple direct step, but involves an often complex or uncertain transformation phase.

**Summary:**
The image illustrates a simplified linear process flow, beginning with a 'Dataset' (represented by a green circle), progressing through an intermediate stage characterized by uncertainty or complexity (depicted as a pink diamond with a large white question mark and several smaller question marks on its facets), and culminating in an 'ML Insight' (represented by an orange circle). The blue arrows clearly indicate the sequential flow from the dataset, through the ambiguous processing step, to the final machine learning insight. The diagram suggests that while raw data is the input and machine learning insight is the desired output, the actual transformation process is not always transparent or straightforward, highlighting an area of potential complexity or unknown factors.](images/77951dd7bc738e15393d2bab69422a6f1cbf03550306f96ca5efebdf4b0a7b34.jpg)

You are here because you want to comprehend the basics of probability before you can dive into the world of statistics and machine learning. Understanding the driving forces behind key statistical features is crucial to reaching your goal of mastering data science. This way you will be able to extract important insight when analysing data through supervised machine learning methods like regressions, but also fathom the outputs unsupervised or assisted ML give you.

Bayesian Inference is a key component heavily used in many fields of mathematics to succinctly express complicated statements. Through Bayesian Notation we can convey the relationships between elements, sets and events. Understanding these new concepts will aid you in interpreting the mathematical intuition behind sophisticated data analytics methods.

Distributions are the main way we lie to classify sets of data. If a dataset complies with certain characteristics, we can usually attribute the likelihood of its values to a specific distribution. Since many of these distributions have elegant relationships between certain outcomes and their probabilities of occurring, knowing key features of our data is extremely convenient and useful.

Probability is the likelihood of an event occurring. This event can be pretty much anything – getting heads, rolling a 4 or even bench pressing 225lbs. We measure probability with numeric values between 0 and 1, because we like to compare the relative likelihood of events. Observe the general probability formula.

$$
\mathsf { P } ( \mathsf { X } ) { = } \frac { P r e f e r r e d o u t c o m e s } { S a m p l e S p a c e }
$$

# Probability Formula:

• The Probability of event X occurring equals the number of preferred outcomes over the number of outcomes in the sample space. Preferred outcomes are the outcomes we want to occur or the outcomes we are interested in. We also call refer to such outcomes as “Favorable”.   
• Sample space refers to all possible outcomes that can occur. Its “size” indicates the amount of elements in it.

# If two events are independent:

The probability of them occurring simultaneously equals the product of them occurring on their own.

$$
\mathsf { P } ( \mathsf { A } \odot ) = \mathsf { P } ( \mathsf { A } ) \mathrm { ~ . ~ } \mathsf { P } ( \odot )
$$

Trial – Observing an event occur and recording the outcome.

Experiment – A collection of one or multiple trials.

Experimental Probability – The probability we assign an event, based on an experiment we conduct.

Expected value – the specific outcome we expect to occur when we run an experiment.

# Example: Trial

Example: Experiment

Flipping a coin and recording the outcome.

Flipping a coin 20 times and recording the 20 individual outcomes.

In this instance, the experimental probability for getting heads would equal the number of heads we record over the course of the 20 outcomes, over 20 (the total number of trials).

The expected value can be numerical, Boolean, categorical or other, depending on the type of the event we are interested in. For instance, the expected value of the trial would be the more likely of the two outcomes, whereas the expected value of the experiment will be the number of time we expect to get either heads or tails after the 20 trials.

Expected value for categorical variables.

Expected value for numeric variables.

$$
E ( X ) = n \times p
$$

$$
E ( X ) = \sum _ { i = 1 } ^ { n } x _ { i } \times p _ { i }
$$

# What is a probability frequency distribution?:

A collection of the probabilities for each possible outcome of an event.

# Why do we need frequency distributions?:

We need the probability frequency distribution to try and predict future events when the expected value is unattainable.

# What is a frequency?:

Frequency is the number of times a given value or outcome appears in the sample space.

# What is a frequency distribution table?:

The frequency distribution table is a table matching each distinct outcome in the sample space to its associated frequency.

# How do we obtain the probability frequency distribution from the frequency distribution table?:

By dividing every frequency by the size of the sample space. (Think about the “favoured over all” formula.)

<table><tr><td>Sum</td><td>Frequency</td><td>Probability</td></tr><tr><td>2</td><td>1</td><td>1/36</td></tr><tr><td>了</td><td>2</td><td>1/18</td></tr><tr><td>4</td><td>3</td><td>1/12</td></tr><tr><td>5</td><td>4</td><td>1/9</td></tr><tr><td>6</td><td>5</td><td>5/36</td></tr><tr><td>7</td><td>6</td><td>1/6</td></tr><tr><td>8</td><td>5</td><td>5/36</td></tr><tr><td>9</td><td>4</td><td>1/9</td></tr><tr><td>10</td><td>3</td><td>1/12</td></tr><tr><td>11</td><td>2</td><td>1/18</td></tr><tr><td>12</td><td>1</td><td>1/36</td></tr></table>

The complement of an event is everything an event is not. We denote the complement of an event with an apostrophe.

![## Image Analysis: ba46daddafbef960714d3b797a84d5f93b2aab715f099bbf2c6bdcbaa3801d3d.jpg

**Conceptual Understanding:**
This image represents the conceptual definition and notation of a complementary event in probability or set theory. Its main purpose is to visually explain that 'A'' is the mathematical symbol used to denote 'Not A', which signifies the complement or opposite of an original event 'A'. It communicates the idea that for any given event, there is an inverse event encompassing all outcomes where the original event does not occur.

**Content Interpretation:**
The image displays a fundamental concept in probability or set theory: the definition of a complementary event. It establishes the notation 'A'' as equivalent to 'Not A'. The left arrow pointing to 'A'' is labeled 'complement', indicating that 'A'' represents the complementary event. The middle arrow pointing to 'Not A' is labeled 'opposite', further emphasizing the meaning of 'Not A' as the event that is contrary to 'A'. The right arrow pointing to 'A' is labeled 'original event', clearly identifying 'A' as the initial event under consideration. These textual elements collectively explain how to denote and interpret the complement of an event.

**Key Insights:**
The main takeaway from this image is the clear definition and notation for a complementary event. Specifically, 'A'' is the standard notation for the complement of an event 'A', which means 'Not A'. This concept, explicitly labeled as 'complement' for 'A'' and 'opposite' for 'Not A', is fundamental for calculating probabilities where the non-occurrence of an event is relevant. The 'original event' is designated as 'A'. This understanding is crucial for correctly interpreting and calculating probabilities in statistical contexts.

**Document Context:**
This image is highly relevant to a section discussing 'How do we obtain the probability frequency distribution from the frequency distribution table?' because understanding complementary events (A' = Not A) is a prerequisite for many probability calculations, especially when dealing with the probability of an event *not* occurring. It lays the groundwork for later discussions on probability rules, such as P(A) + P(A') = 1, which are essential for constructing probability frequency distributions from raw frequencies.

**Summary:**
The image clearly defines the notation for a complementary event in probability or set theory. It shows the expression A' = Not A, where A' is identified as the 'complement' of event A, 'Not A' is identified as the 'opposite' of event A, and 'A' itself is identified as the 'original event'. This visual representation provides a foundational understanding of how a complementary event is denoted and what it signifies in relation to the original event, which is crucial for comprehending probability frequency distributions.](images/ba46daddafbef960714d3b797a84d5f93b2aab715f099bbf2c6bdcbaa3801d3d.jpg)

# Characteristics of complements:

• Can never occur simultaneously. • Add up to the sample space. $( \mathsf { A } + \mathsf { A } ^ { \prime } =$ Sample space) Their probabilities add up to 1. $( \mathsf { P } ( \mathsf { A } ) + \mathsf { P } ( \mathsf { A } ^ { \prime } ) = 1 )$ • The complement of a complement is the original event. $( ( \mathsf { A } ^ { \prime } ) ^ { \prime } = \mathsf { A } )$

# Example:

• Assume event A represents drawing a spade, so $\mathsf { P } ( \mathsf { A } ) = 0 . 2 5$ .   
• Then, A’ represents not drawing a spade, so drawing a club, a diamond or a heart. $\mathsf { P } ( \mathsf { A } ^ { \prime } ) = 1 - \mathsf { P } ( \mathsf { A } ) ,$ , so $\mathsf { P } ( \mathsf { A } ^ { \prime } ) = 0 . 7 5 .$ .