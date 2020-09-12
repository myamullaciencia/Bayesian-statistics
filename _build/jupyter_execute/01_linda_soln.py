# Bite Size Bayes

Copyright 2020 Allen B. Downey

License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# Load utils.py

import os

if not os.path.exists('utils.py'):
    !wget https://github.com/AllenDowney/BiteSizeBayes/raw/master/utils.py

# Load the data file

if not os.path.exists('gss_bayes.csv'):
    !wget https://github.com/AllenDowney/BiteSizeBayes/raw/master/gss_bayes.csv

import pandas as pd
import numpy as np

from utils import values

## Introduction

This notebook takes a computational approach to understanding probability.  We'll use data from the General Social Survey to compute the probability of propositions like:

* If I choose a random survey respondent, what is the probability they are female?

* If I choose a random survey respondent, what is the probability they work in banking?

From there, we will explore two related concepts:

* Conjunction, which is the probability that two propositions are both true; for example, what is the probability of choosing a female banker?

* Conditional probability, which is the probability that one proposition is true, given that another is true; for example, given than a respondent is female, what is the probability that she is a banker?

I chose these examples because they are related to a famous experiment by Tversky and Kahneman, who posed the following question:

> Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations.  Which is more probable?
1. Linda is a bank teller.
2. Linda is a bank teller and is active in the feminist movement.

Many people choose the second answer, presumably because it seems more consistent with the description.  It seems unlikely that Linda would be *just* a bank teller; if she is a bank teller, it seems likely that she would also be a feminist.

But the second answer cannot be "more probable", as the question asks.  Suppose we find 1000 people who fit Linda's description and 10 of them work as bank tellers.  How many of them are also feminists?  At most, all 10 of them are; in that case, the two options are *equally* likely.  More likely, only some of them are; in that case the second option is *less* likely.  But there can't be more than 10 out of 10, so the second option cannot be more likely.

The error people make if they choose the second option is called the [conjunction fallacy](https://en.wikipedia.org/wiki/Conjunction_fallacy).   It's called a [fallacy](https://en.wikipedia.org/wiki/Fallacy) because it's a logical error and "conjunction" because "bank teller AND feminist" is a [logical conjunction](https://en.wikipedia.org/wiki/Logical_conjunction).

If this example makes you uncomfortable, you are in good company.  The biologist [Stephen J. Gould wrote](https://sci-hub.tw/https://doi.org/10.1080/09332480.1989.10554932) :

> I am particularly fond of this example because I know that the [second] statement is least probable, yet a little [homunculus](https://en.wikipedia.org/wiki/Homunculus_argument) in my head continues to jump up and down, shouting at me, "but she can't just be a bank teller; read the description."

If the little person in your head is still unhappy, maybe this notebook will help.

## Probability

At this point I should define probability, but that [turns out to be surprisingly difficult](https://en.wikipedia.org/wiki/Probability_interpretations).  To avoid getting bogged down before we get started, I'll start with a simple definition: a **probability** is a **fraction** of a dataset.

For example, if we survey 1000 people, and 20 of them are bank tellers, the fraction that work as bank tellers is 0.02 or 2\%.  If we choose a person from this population at random, the probability that they are a bank teller is 2\%.  

(By "at random" I mean that every person in the dataset has the same chance of being chosen, and by "they" I mean the [singular, gender-neutral pronoun](https://en.wikipedia.org/wiki/Singular_they), which is a correct and useful feature of English.)

With this definition and an appropriate dataset, we can compute probabilities by counting.

To demonstrate, I'll use a data set from the [General Social Survey](http://gss.norc.org/) or GSS.  The following cell reads the data.

gss = pd.read_csv('gss_bayes.csv', index_col=0)

The results is a Pandas DataFrame with one row for each person surveyed and one column for each variable I selected.

Here are the number of rows and columns:

gss.shape

And here are the first few rows:

gss.head()

The columns are

* `caseid`: Respondent id (which is the index of the table).

* `year`: Year when the respondent was surveyed.

* `age`: Respondent's age when surveyed.

* `sex`: Male or female.

* `polviews`: Political views on a range from liberal to conservative.

* `partyid`: Political party affiliation, Democrat, Independent, or Republican.

* `indus10`: [Code](https://www.census.gov/cgi-bin/sssd/naics/naicsrch?chart=2007) for the industry the respondent works in.

Let's look at these variables in more detail, starting with `indus10`.

## Banking

The code for "Banking and related activities" is 6870, so we can select bankers like this:

banker = (gss['indus10'] == 6870)

The result is a Boolean series, which is a Pandas Series that contains the values `True` and `False`.  Here are the first few entries:

banker.head()

We can use `values` to see how many times each value appears.

values(banker)

In this dataset, there are 728 bankers.

If we use the `sum` function on this Series, it treats `True` as 1 and `False` as 0, so the total is the number of bankers.

banker.sum()

To compute the *fraction* of bankers, we can divide by the number of people in the dataset:

banker.sum() / banker.size

But we can also use the `mean` function, which computes the fraction of `True` values in the Series:

banker.mean()

About 1.5% of the respondents work in banking.

That means if we choose a random person from the dataset, the probability they are a banker is about 1.5%.

**Exercise**: The values of the column `sex` are encoded like this:

```
1    Male
2    Female
```

The following cell creates a Boolean series that is `True` for female respondents and `False` otherwise.

female = (gss['sex'] == 2)

* Use `values` to display the number of `True` and `False` values in `female`.

* Use `sum` to count the number of female respondents.

* Use `mean` to compute the fraction of female respondents.

# Solution

values(gss['sex'])

# Solution

female.sum()

# Solution

female.mean()

The fraction of women in this dataset is higher than in the adult U.S. population because [the GSS does not include people living in institutions](https://gss.norc.org/faq), including prisons and military housing, and those populations are more likely to be male.

**Exercise:** The designers of the General Social Survey chose to represent sex as a binary variable.  What alternatives might they have considered?  What are the advantages and disadvantages of their choice?

For more on this topic, you might be interested in this article: Westbrook and Saperstein, [New categories are not enough: rethinking the measurement of sex and gender in social surveys](https://sci-hub.tw/10.1177/0891243215584758)

## Political views

The values of `polviews` are on a seven-point scale:

```
1	Extremely liberal
2	Liberal
3	Slightly liberal
4	Moderate
5	Slightly conservative
6	Conservative
7	Extremely conservative
```

Here are the number of people who gave each response:

values(gss['polviews'])

I'll define `liberal` to be `True` for anyone whose response is "Extremely liberal", "Liberal", or "Slightly liberal".

liberal = (gss['polviews'] < 4)

Here are the number of `True` and `False` values:

values(liberal)

And the fraction of respondents who are "liberal".

liberal.mean()

If we choose a random person in this dataset, the probability they are liberal is about 27%.

## The probability function

To summarize what we have done so far:

* To represent a logical proposition like "this respondent is liberal", we are using a Boolean series, which contains the values `True` and `False`.

* To compute the probability that a proposition is true, we are using the `mean` function, which computes the fraction of `True` values in a series.

To make this computation more explicit, I'll define a function that takes a Boolean series and returns a probability:

def prob(A):
    """Computes the probability of a proposition, A.
    
    A: Boolean series
    
    returns: probability
    """
    assert isinstance(A, pd.Series)
    assert A.dtype == 'bool'
    
    return A.mean()

The `assert` statements check whether `A` is a Boolean series.  If not, they display an error message.

Using this function to compute probabilities makes the code more readable.  Here are the probabilities for the propositions we have computed so far.

prob(banker)

prob(female)

prob(liberal)

**Exercise**: The values of `partyid` are encoded like this:

```
0	Strong democrat
1	Not str democrat
2	Ind,near dem
3	Independent
4	Ind,near rep
5	Not str republican
6	Strong republican
7	Other party
```

I'll define `democrat` to include respondents who chose "Strong democrat" or "Not str democrat":

democrat = (gss['partyid'] <= 1)

* Use `mean` to compute the fraction of Democrats in this dataset.

* Use `prob` to compute the same fraction, which we will think of as a probability.

# Solution

democrat.mean()

# Solution

prob(democrat)

## Conjunction

Now that we have a definition of probability and a function that computes it, let's move on to conjunction.

"Conjunction" is another name for the logical `and` operation.  If you have two propositions, `A` and `B`, the conjunction `A and B` is `True` if both `A` and `B` are `True`, and `False` otherwise.

I'll demonstrate using two Boolean series constructed to enumerate every combination of `True` and `False`:

A = pd.Series((True, True, False, False))
A

B = pd.Series((True, False, True, False))
B

To compute the conjunction of `A` and `B`, we can use the `&` operator, like this:

A & B

The result is `True` only when `A` and `B` are `True`.

To show this operation more clearly, I'll put the operands and the result in a DataFrame:

table = pd.DataFrame()
table['A'] = A
table['B'] = B
table['A & B'] = A & B
table

This way of representing a logical operation is called a [truth table](https://en.wikipedia.org/wiki/Truth_table).

In a previous section, we computed the probability that a random respondent is a banker:

prob(banker)

And the probability that a respondent is a Democrat:

prob(democrat)

Now we can compute the probability that a random respondent is a banker *and* a Democrat:

prob(banker & democrat)

As we should expect, `prob(banker & democrat)` is less than `prob(banker)`, because not all bankers are Democrats.

**Exercise:** Use `prob` and the `&` operator to compute the following probabilities.

* What is the probability that a random respondent is a banker and liberal?

* What is the probability that a random respondent is female, a banker, and liberal?

* What is the probability that a random respondent is female, a banker, and a liberal Democrat?

Notice that as we add more conjunctions, the probabilities get smaller.

# Solution

prob(banker & liberal)

# Solution

prob(female & banker & liberal)

# Solution

prob(female & banker & liberal & democrat)

**Exercise:** We expect conjunction to be commutative; that is, `A & B` should be the same as `B & A`.

To check, compute these two probabilies:

* What is the probability that a random respondent is a banker and liberal?
* What is the probability that a random respondent is liberal and a banker?

prob(banker & liberal)

prob(liberal & banker)

If they are not the same, something has gone very wrong!

## Conditional probability

Conditional probability is a probability that depends on a condition, but that might not be the most helpful definition.  Here are some examples:

* What is the probability that a respondent is a Democrat, given that they are liberal?

* What is the probability that a respondent is female, given that they are a banker?

* What is the probability that a respondent is liberal, given that they are female?

Let's start with the first one, which we can interpret like this: "Of all the respondents who are liberal, what fraction are Democrats?"

We can compute this probability in two steps:

1. Select all respondents who are liberal.

2. Compute the fraction of the selected respondents who are Democrats.

To select liberal respondents, we can use the bracket operator, `[]`, like this:

selected = democrat[liberal]

The result is a Boolean series that contains a subset of the values in `democrat`.  Specifically, it contains only the values where `liberal` is `True`.

To confirm that, let's check the length of the result:

len(selected)

If things have gone according to plan, that should be the same as the number of `True` values in `liberal`:

liberal.sum()

Good.  

`selected` contains the value of `democrat` for liberal respondents, so the mean of `selected` is the fraction of liberals who are Democrats:

selected.mean()

A little more than half of liberals are Democrats.  If the result is lower than you expected, keep in mind:

1. We used a somewhat strict definition of "Democrat", excluding Independents who "lean" democratic.

2. The dataset includes respondents as far back as 1974; in the early part of this interval, there was less alignment between political views and party affiliation, compared to the present.

Let's try the second example, "What is the probability that a respondent is female, given that they are a banker?"

We can interpret that to mean, "Of all respondents who are bankers, what fraction are female?"

Again, we'll use the bracket operator to select only the bankers:

selected = female[banker]
len(selected)

As we've seen, there are 728 bankers in the dataset.

Now we can use `mean` to compute the conditional probability that a respondent is female, given that they are a banker:

selected.mean()

About 77% of the bankers in this dataset are female.

We can get the same result using `prob`:

prob(selected)

Remember that we defined `prob` to make the code easier to read.  We can do the same thing with conditional probability.

I'll define `conditional` to take two Boolean series, `A` and `B`, and compute the conditional probability of `A` given `B`:

def conditional(A, B):
    """Conditional probability of A given B.
    
    A: Boolean series
    B: Boolean series
    
    returns: probability
    """
    return prob(A[B])

Now we can use it to compute the probability that a liberal is a Democrat:

conditional(democrat, liberal)

And the probability that a banker is female:

conditional(female, banker)

The results are the same as what we computed above.

**Exercise:** Use `conditional` to compute the probability that a respondent is liberal given that they are female.

Hint: The answer should be less than 30%.  If your answer is about 54%, you have made a mistake (see the next exercise).


# Solution

conditional(liberal, female)

**Exercise:**  In a previous exercise, we saw that conjunction is commutative; that is, `prob(A & B)` is always equal to `prob(B & A)`.

But conditional probability is NOT commutative; that is, `conditional(A, B)` is not the same as `conditional(B, A)`.

That should be clear if we look at an example.  Previously, we computed the probability a respondent is female, given that they are banker.

conditional(female, banker)

The result shows that the majority of bankers are female.  That is not the same as the probability that a respondent is a banker, given that they are female:

conditional(banker, female)

Only about 2% of female respondents are bankers.

**Exercise:** Use `conditional` to compute the following probabilities:

* What is the probability that a respondent is liberal, given that they are a Democrat?

* What is the probability that a respondent is a Democrat, given that they are liberal?

Think carefully about the order of the series you pass to `conditional`.

conditional(liberal, democrat)

conditional(democrat, liberal)

## Conditions and conjunctions

We can combine conditional probability and conjunction.  For example, here's the probability a respondent is female, given that they are a liberal Democrat.

conditional(female, liberal & democrat)

Almost 57% of liberal Democrats are female.

And here's the probability they are a liberal female, given that they are a banker:

conditional(liberal & female, banker)

About 17% of bankers are liberal women.

**Exercise:** What fraction of female bankers are liberal Democrats?

Hint: If your answer is less than 1%, you have it backwards.  Remember that conditional probability is not commutative.

# Solution

conditional(liberal & democrat, female & banker)

## Summary

At this point, you should understand the definition of probability, at least in the simple case where we have a finite dataset.  Later we will consider cases where the definition of probability is more controversial.

And you should understand conjunction and conditional probability.  [In the next notebook](https://colab.research.google.com/github/AllenDowney/BiteSizeBayes/blob/master/02_bayes.ipynb), we will explore the relationship between conjunction and conditional probability, and use it to derive Bayes's Theorem, which is the foundation of Bayesian statistics.

