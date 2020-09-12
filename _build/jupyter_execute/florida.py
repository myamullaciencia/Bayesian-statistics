# Bite Size Bayes

Copyright 2020 Allen B. Downey

License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## The "Girl Named Florida" problem

In [The Drunkard's Walk](https://www.goodreads.com/book/show/2272880.The_Drunkard_s_Walk), Leonard Mlodinow presents "The Girl Named Florida Problem":

>"In a family with two children, what are the chances, if [at least] one of the children is a girl named Florida, that both children are girls?"

I added "at least" to Mlodinow's statement of the problem to avoid a subtle ambiguity (which I'll explain at the end).

To avoid some real-world complications, let's assume that this question takes place in an imaginary city called Statesville where:

* Every family has two children.

* 50% of children are male and 50% are female.

* All children are named after U.S. states, and all state names are chosen with equal probability.

* Genders and names within each family are chosen independently.

To answer Mlodinow's question, I'll create a DataFrame with one row for each family in Statesville and a column for the gender and name of each child.

Here's a list of genders and a [dictionary of state names](https://gist.github.com/tlancon/9794920a0c3a9990279de704f936050c):

gender = ['B', 'G']

us_states = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
#    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

To enumerate all possible combinations of genders and names, I'll use `from_product`, which makes a Pandas MultiIndex.

names = ['gender1', 'name1', 'gender2', 'name2']

index = pd.MultiIndex.from_product([gender, us_states]*2, 
                                   names=names)

Now I'll create a DataFrame with that index:

df = pd.DataFrame(index=index)
df.head()

It will be easier to work with if I reindex it so the levels in the MultiIndex become columns.

df = df.reset_index()
df.head()

This DataFrame contains one row for each family in Statesville; for example, the first row represents a family with two boys, both named Alabama.

As it turns out, there are 10,000 families in Statesville:

len(df)

## Probabilities

To compute probabilities, we'll use Boolean Series.  For example, the following Series is `True` for each family where the first child is a girl:

girl1 = (df['gender1']=='G')

The following function takes a Boolean Series and counts the number of `True` values, which is the probability that the condition is true.

def prob(A):
    """Computes the probability of a proposition, A.
    
    A: Boolean series
    
    returns: probability
    """
    assert isinstance(A, pd.Series)
    assert A.dtype == 'bool'
    
    return A.mean()

Not surprisingly, the probability is 50% that the first child is a girl.

prob(girl1)

And so is the probability that the second child is a girl.

girl2 = (df['gender2']=='G')
prob(girl2)

Mlodinow's question is a conditional probability: given that one of the children is a girl named Florida, what is the probability that both children are girls?

To compute conditional probabilities, I'll use this function, which takes two Boolean Series, `A` and `B`, and computes the conditional probability $P(A~\mathrm{given}~B)$.

def conditional(A, B):
    """Conditional probability of A given B.
    
    A: Boolean series
    B: Boolean series
    
    returns: probability
    """
    return prob(A[B])

For example, here's the probability that the second child is a girl, given that the first child is a girl.

conditional(girl2, girl1)

The result is 50%, which is the same as the unconditioned probability that the second child is a girl:

prob(girl2)

So that confirms that the genders of the two children are independent, which is one of my assumptions.

Now, Mlodinow's question asks about the probability that both children are girls, so let's compute that.

gg = (girl1 & girl2)
prob(gg)

In 25% of families, both children are girls.  And that should be no surprise: because they are independent, the probability of the conjunction is the product of the probabilities:

prob(girl1) * prob(girl2)

While we're at it, we can also compute the conditional probability of two girls, given that the first child is a girl.

conditional(gg, girl1)

That's what we should expect.  If we know the first child is a girl, and the probability is 50% that the second child is a girl, the probability of two girls is 50%.

## At least one girl

Before I answer Mlodinow's question, I'll warm up with a simpler version: given that at least one of the children is a girl, what is the probability that both are?

To compute the probability of "at least one girl" I will use the `|` operator, which computes the logical `OR` of the two Series:

at_least_one_girl = (girl1 | girl2)
prob(at_least_one_girl)

75% of the families in Statesville have at least one girl.

Now we can compute the conditional probability of two girls, given that the family has at least one girl.

conditional(gg, at_least_one_girl)

Of the families that have at least one girl, `1/3` have two girls.

If you have not thought about questions like this before, that result might surprise you.  The following figure might help:

<img width="200" src="https://github.com/AllenDowney/BiteSizeBayes/raw/master/GirlNamedFlorida1.png">

In the top left, the gray square represents a family with two boys; in the lower right, the dark blue square represents a family with two girls.

The other two quadrants represent families with one girl, but note that there are two ways that can happen: the first child can be a girl or the second child can be a girl.

There are an equal number of families in each quadrant.

If we select families with at least one girl, we eliminate the gray square in the upper left.  Of the remaining three squares, one of them has two girls.

So if we know a family has at least one girl, the probability they have two girls is 33%.

## What's in a name?

So far, we have computed two conditional probabilities:

* Given that the first child is a girl, the probability is 50% that both children are girls.

* Given that at least one child is a girl, the probability is 33% that both children are girls.

Now we're ready to answer Mlodinow's question:

* Given that at least one child is a girl *named Florida*, what is the probability that both children are girls?

If your intuition is telling you that the name of the child can't possibly matter, brace yourself.

Here's the probability that the first child is a girl named Florida.

gf1 = girl1 & (df['name1']=='Florida')
prob(gf1)

And the probability that the second child is a girl named Florida.

gf2 = girl2 & (df['name2']=='Florida')
prob(gf2)

To compute the probability that at least one of the children is a girl named Florida, we can use the `|` operator again.  

at_least_one_girl_named_florida = (gf1 | gf2)
prob(at_least_one_girl_named_florida)

We can double-check it by using the disjunction rule:

prob(gf1) + prob(gf2) - prob(gf1 & gf2)

So, the percentage of families with at least one girl named Florida is a little less than 2%.

Now, finally, here is the answer to Mlodinow's question:

conditional(gg, at_least_one_girl_named_florida)

That's right, the answer is about 49.7%.  To summarize:

* Given that the first child is a girl, the probability is 50% that both children are girls.

* Given that at least one child is a girl, the probability is 33% that both children are girls.

* Given that at least one child is a girl *named Florida*, the probability is 49.7% that both children are girls.

If your brain just exploded, I'm sorry.

Here's my best attempt to put your brain back together.

For each child, there are three possibilities: boy (B), girl not named Florida (G), and girl named Florida (GF), with these probabilities:

$P(B) = 1/2 $

$P(G) = 1/2 - x $

$P(GF) = x $

where $x$ is the percentage of people who are girls named Florida. 

In families with two children, here are the possible combinations and their probabilities:

$P(B, B) = (1/2)(1/2)$

$P(B, G) = (1/2)(1/2-x)$

$P(B, GF) = (1/2)(x)$

$P(G, B) = (1/2-x)(1/2)$

$P(G, G) = (1/2-x)(1/2-x)$

$P(G, GF) = (1/2-x)(x)$

$P(GF, B) = (x)(1/2)$

$P(GF, G) = (x)(1/2-x)$

$P(GF, GF) = (x)(x)$

If we select only the families that have at least one girl named Florida, here are their probabilities:

$P(B, GF) = (1/2)(x)$

$P(G, GF) = (1/2-x)(x)$

$P(GF, B) = (x)(1/2)$

$P(GF, G) = (x)(1/2-x)$

$P(GF, GF) = (x)(x)$

Of those, if we select the families with two girls, here are their probabilities:

$P(G, GF) = (1/2-x)(x)$

$P(GF, G) = (x)(1/2-x)$

$P(GF, GF) = (x)(x)$

To get the conditional probability of two girls, given at least one girl named Florida, we can add up the last 3 probabilities and divide by the sum of the previous 5 probabilities.

With a little algebra, we get:

$P(\mathrm{two~girls} ~|~ \mathrm{at~least~one~girl~named~Florida}) = (1 - x) / (2 - x)$

As $x$ approaches $0$ the answer approaches $1/2$.

As $x$ approaches $1/2$, the answer approaches $1/3$.

Here's what all of that looks like graphically:

<img width="200" src="https://github.com/AllenDowney/BiteSizeBayes/raw/master/GirlNamedFlorida2.png">

Here `B` a boy, `Gx` is a girl with some property `X`, and `G` is a girl who doesn't have that property.  If we select all families with at least one `Gx`, we get the five blue squares (light and dark).  Of those, the families with two girls are the three dark blue squares.

If property `X` is common, the ratio of dark blue to all blue approaches `1/3`.  If `X` is rare, the same ratio approaches `1/2`.

In the "Girl Named Florida" problem, `x` is 1/100, and we can compute the result: 

x = 1/100
(1-x) / (2-x)

Which is what we got by counting all of the families in Statesville.

## Controversy

[I wrote about this problem in my blog in 2011](http://allendowney.blogspot.com/2011/11/girl-named-florida-solutions.html).  As you can see in the comments, my explanation was not met with universal acclaim.

One of the issues that came up is the challenge of stating the question unambiguously.  In this article, I rephrased Mlodinow's statement to clarify it.

But since we have come all this way, let me also answer a different version of the problem.

>Suppose you choose a house in Statesville at random and ring the doorbell.  A girl (who lives there) opens the door and you learn that her name is Florida.  What is the probability that the other child in this house is a girl?

In this version of the problem, the selection process is different.  Instead of selecting houses with at least one girl named Florida, you selected a house, then selected a child, and learned that her name is Florida.

Since the selection of the child was arbitrary, we can say without loss of generality that the child you met is the first child in the table.

In that case, the conditional probability of two girls is:

conditional(gg, gf1)

Which is the same as the conditional probability, given that the first child is a girl:

conditional(gg, girl1)

So in this version of the problem, the girl's name is irrelevant.

