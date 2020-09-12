# Chapter 2

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

## Review

[In the previous notebook](https://colab.research.google.com/github/AllenDowney/BiteSizeBayes/blob/master/01_linda.ipynb) I defined probability, conjunction, and conditional probability, and used data from the General Social Survey (GSS) to compute the probability of various logical propositions.

To review, here's how we loaded the dataset:

gss = pd.read_csv('gss_bayes.csv', index_col=0)

And here are the logical propositions we defined, represented using Boolean series.

banker = (gss['indus10'] == 6870)

female = (gss['sex'] == 2)

liberal = (gss['polviews'] < 4)

democrat = (gss['partyid'] <= 1)

I defined the following function, which uses `mean` to compute the fraction of `True` values in a Boolean series.

def prob(A):
    """Computes the probability of a proposition, A.
    
    A: Boolean series
    
    returns: probability
    """
    assert isinstance(A, pd.Series)
    assert A.dtype == 'bool'
    
    return A.mean()

So we can compute the probability of a proposition like this:

prob(female)

Then we used the `&` operator to compute the probability of a conjunction, like this:

prob(female & banker)

Next I defined the following function, which uses the bracket operator to compute conditional probability:

def conditional(A, B):
    """Conditional probability of A given B.
    
    A: Boolean series
    B: Boolean series
    
    returns: probability
    """
    return prob(A[B])

We showed that conjunction is commutative, so `prob(A & B)` equals `prob(B & A)`, for any logical propositions `A` and `B`.

For example:

prob(liberal & democrat)

prob(democrat & liberal)

But conditional probability is NOT commutative, so `conditional(A, B)` is generally not the same as `conditional(B, A)`.

For example, here's the probability that a respondent is female, given that they are a banker.

conditional(female, banker)

And here's the probability that a respondent is a banker, given that they are female.

conditional(banker, female)

Not even close.

## More propositions

For the sake of variety in our examples, let's define some new propositions.

Here's the probability that a random respondent is male.

male = (gss['sex']==1)
prob(male)

The industry code for "Construction" is `770`.  Let's call someone in this field a "builder".

builder = (gss['indus10'] == 770)
prob(builder)

And let's define propositions for conservatives and Republicans.

conservative = (gss['polviews'] > 4)
prob(conservative)

republican = (gss['partyid'].isin([5,6]))
prob(republican)

The `isin` function checks whether values are in a given sequence.  In this example, the values `5` and `6` represent the responses "Strong Republican" and "Not Strong Republican".

Finally, I'll use `age` to define propositions for `young` and `old`.

young = (gss['age'] < 30)
prob(young)

old = (gss['age'] >= 65)
prob(old)

For these thresholds, I chose round numbers near the 20th and 80th percentiles.  Depending on your age, you may or may not agree with these definitions of "young" and "old".

**Exercise:** There's a [famous quote](https://quoteinvestigator.com/2014/02/24/heart-head/) about young people, old people, liberals, and conservatives that goes something like:

> If you are not a liberal at 25, you have no heart. If you are not a conservative at 35, you have no brain.

Whether you agree with this proposition or not, it suggests some probabilities we can compute as a review exercise.  
Use `prob` and `conditional` to compute these probabilities.

* What is the probability that a randomly chosen respondent is a young liberal?

* What is the probability that a young person is liberal?

* What fraction of respondents are old conservatives?

* What fraction of conservatives are old?

For each statement, think about whether it is expressing a conjunction, or a conditional probability, or both.

And for the conditional probabilities, be careful about the order!

# Solution goes here

# Solution goes here

# Solution goes here

# Solution goes here

If your last answer is greater than 30%, you have it backwards!

## Onward!

In this notebook, we'll derive three relationships between conjunction and conditional probability:

* Theorem 1: Using conjunction to compute a conditional probability.

* Theorem 2: Using a conditional probability to compute a conjunction.

* Theorem 3: Using `conditional(A, B)` to compute `conditional(B, A)`.

Theorem 3 is also known as Bayes's Theorem, which is the foundation of Bayesian statistics.

For parts of this notebook it will be useful to use mathematical notation for probability, so I'll introduce that now.

* $P(A)$ is the probability of proposition $A$.

* $P(A~\mathrm{and}~B)$ is the probability of the conjunction of $A$ and $B$, that is, the probability that both are true.

* $P(A | B)$ is the conditional probability of $A$ given that $B$ is true.  The vertical line between $A$ and $B$ is pronounced "given". 

With that, we are ready for Theorem 1.

## Theorem 1

What fraction of builders are male?  We have already seen one way to compute the answer:

1. Use the bracket operator to select the builders, then

2. Use `mean` to compute the fraction of builders who are male.

We can write these steps like this:

male[builder].mean()

Or we can use the `conditional` function, which does the same thing:

conditional(male, builder)

But there is another way: to compute the fraction of builders who are male, we can compute the ratio of two probabilities:

1. The fraction of respondents who are male builders, and

2. The fraction of respondents who are builders.

Here's what that looks like.

prob(male & builder) / prob(builder)

The result is the same.

This example demonstrates a general rule that relates conditional probability and conjunction.  Here's what it looks like in math notation:

$P(A|B) = \frac{P(A~\mathrm{and}~B)}{P(B)}$

And that's Theorem 1.

In this example:

`conditional(male, builder) = prob(male & builder) / prob(builder)`

**Exercise:**  What fraction of conservatives are Republican?  Compute the answer two ways:

* Use `conditional` (which uses the bracket operator), and

* Use Theorem 1.

Confirm that you get the same answer.

Note: Due to floating-point arithmetic, the results might not be exactly the same, but almost all of the digits should be the same.

# Solution goes here

# Solution goes here

## Proof?

I didn't really prove Theorem 1; mostly, it is a statement of what conditional probability means.

For example, consider this Venn diagram:

<img width="200" src="https://github.com/AllenDowney/BiteSizeBayes/raw/master/theorem1_venn_diagram.png">


The blue circle represents male respondents.  The red circle represents builders.  The intersection represents male builders.

To compute the fraction of builders who are male, we can compute the ratio of the intersection, which is `prob(male & builder)` to the red circle, which is `prob(builder)`.

**Exercise:** For practice, compute fraction of bankers who are old both ways: using `conditional` and using Theorem 1.

# Solution goes here

# Solution goes here

## Theorem 2

Here's Theorem 1 again:

$P(A|B) = \frac{P(A~\mathrm{and}~B)}{P(B)}$ 

If we multiply both sides by $P(B)$, we get Theorem 2.

$P(A~\mathrm{and}~B) = P(B) P(A|B)$

This formula suggests a second way to compute a conjunction: instead of using the `&` operator, we can compute the product of two probabilities.

Let's see if it works for `conservative` and `republican`.  Here's the result using `&`:

prob(conservative & republican)

And here's the result using Theorem 2:

prob(republican) * conditional(conservative, republican)

Because of floating-point errors, they might not be identical, but almost all of the digits are the same.

**Exercise:** Check Theorem 2 one more time by computing the fraction of respondents who are old liberals both ways:

* Using the `&` operator, and

* Using Theorem 2.

The results should be the same, or at least very close.

# Solution goes here

# Solution goes here

## Conjunction is commutative

We have already established that conjunction is commutative.  In math notation, that means:

$P(A~\mathrm{and}~B) = P(B~\mathrm{and}~A)$

If we apply Theorem 2 to both sides, we have

$P(B) P(A|B) = P(A) P(B|A)$

Here's one way to interpret that: if you want to check $A$ and $B$, you can do it in either order:

1. You can check $B$ first, then $A$ conditioned on $B$, or

2. You can check $A$ first, then $B$ conditioned on $A$.

To try it out, I'll compute the fraction of young builders both ways:

prob(young) * conditional(builder, young)

prob(builder) * conditional(young, builder)

Same thing!

**Exercise:** Compute the probability of being a male banker both ways and see if you get the same thing.

# Solution goes here

# Solution goes here

## Theorem 3

In the previous section we established that 

$P(B) P(A|B) = P(A) P(B|A)$

If we divide through by $P(B)$, we get Theorem 3:

$P(A|B) = \frac{P(A) P(B|A)}{P(B)}$

And that, my friends, is Bayes's Theorem.

To see how it works, let's try one more combination of our propositions.  Let's compute the fraction of builders who are liberal, first using `conditional`:

conditional(liberal, builder)

Now using Bayes's Theorem:

prob(liberal) * conditional(builder, liberal) / prob(builder)

Same thing!

**Exercise:** Try it yourself!  Compute the fraction of young people who are Republican both ways: using `conditional` and using Bayes's Theorem.  See if you get the same thing.

conditional(republican, young)

prob(republican) * conditional(young, republican) / prob(young)

## Summary

Here's what we have so far:

**Theorem 1** gives us a new way to compute a conditional probability using a conjunction:

$P(A|B) = \frac{P(A~\mathrm{and}~B)}{P(B)}$ 

**Theorem 2** gives us a new way to compute a conjunction using a conditional probability:

$P(A~\mathrm{and}~B) = P(B) P(A|B)$

**Theorem 3**, also known as Bayes's Theorem, gives us a way to get from $P(A|B)$ to $P(B|A)$, or the other way around:

$P(A|B) = \frac{P(A) P(B|A)}{P(B)}$

But at this point you might ask, "So what?"  If we have all of the data, we can compute any probability we want, any conjunction, or any conditional probability, just by counting.  Why do we need these formulas?

And you are right, *if* we have all of the data.  But often we don't, and in that case, these formulas can be pretty useful -- especially Bayes's Theorem.

In the next notebook, we'll see how.

