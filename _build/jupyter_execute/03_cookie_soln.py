# Bite Size Bayes

Copyright 2020 Allen B. Downey

License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Review

[In the previous notebook](https://colab.research.google.com/github/AllenDowney/BiteSizeBayes/blob/master/02_bayes.ipynb) I presented and proved (sort of) three theorems of probability:

**Theorem 1** gives us a way to compute a conditional probability using a conjunction:

$P(A|B) = \frac{P(A~\mathrm{and}~B)}{P(B)}$ 

**Theorem 2** gives us a way to compute a conjunction using a conditional probability:

$P(A~\mathrm{and}~B) = P(B) P(A|B)$

**Theorem 3** gives us a way to get from $P(A|B)$ to $P(B|A)$, or the other way around:

$P(A|B) = \frac{P(A) P(B|A)}{P(B)}$

In the examples we've seen so far, we didn't really need these theorems, because when you have all of the data, you can compute any probability you want, any conjunction, or any conditional probability, just by counting. 

Starting with this notebook, we will look at examples where we don't have all of the data, and we'll see that these theorems are useful, expecially Theorem 3, which is also known as Bayes's Theorem.

## Bayes's Theorem

There are two ways to think about Bayes's Theorem:

* It is a divide-and conquer strategy for computing conditional probabilities.  If it's hard to compute $P(A|B)$ directly, sometimes it is easier to compute the terms on the other side of the equation: $P(A)$, $P(B|A)$, and $P(B)$.

* It is also a recipe for updating beliefs in the light of new data.

When we are working with the second interpretation, we often write Bayes's Theorem with different variables.  Instead of $A$ and $B$, we use $H$ and $D$, where

* $H$ stands for "hypothesis", and

* $D$ stands for "data".

So we write Bayes's Theorem like this:

$P(H|D) = P(H) ~ P(D|H) ~/~ P(D)$

In this context, each term has a name:

* $P(H)$ is the "prior probability" of the hypothesis, which represents how confident you are that $H$ is true prior to seeing the data,

* $P(D|H)$ is the "likelihood" of the data, which is the probability of seeing $D$ if the hypothesis is true,

* $P(D)$ is the "total probability of the data", that is, the chance of seeing $D$ regardless of whether $H$ is true or not.

* $P(H|D)$ is the "posterior probability" of the hypothesis, which indicates how confident you should be that $H$ is true after taking the data into account.

An example will make all of this clearer.

## The cookie problem

Here's a problem I got from Wikipedia a long time ago, but now it's been edited away.

> Suppose you have two bowls of cookies.  Bowl 1 contains 30 vanilla and 10 chocolate cookies.  Bowl 2 contains 20 vanilla of each.
>
> You choose one of the bowls at random and, without looking into the bowl, choose one of the cookies at random.  It turns out to be a vanilla cookie.
>
> What is the chance that you chose Bowl 1?

We'll assume that there was an equal chance of choosing either bowl and an equal chance of choosing any cookie in the bowl.

We can solve this problem using Bayes's Theorem.  First, I'll define $H$ and $D$:

* $H$ is the hypothesis that the bowl you chose is Bowl 1.

* $D$ is the datum that the cookie is vanilla ("datum" is the rarely-used singular form of "data").

What we want is the posterior probability of $H$, which is $P(H|D)$.  It is not obvious how to compute it directly, but if we can figure out the terms on the right-hand side of Bayes's Theorem, we can get to it indirectly.

1. $P(H)$ is the prior probability of $H$, which is the probability of choosing Bowl 1 before we see the data.  If there was an equal chance of choosing either bowl, $P(H)$ is $1/2$.

2. $P(D|H)$ is the likelihood of the data, which is the chance of getting a vanilla cookie if $H$ is true, in other words, the chance of getting a vanilla cookie from Bowl 1, which is $30/40$ or $3/4$.

3. $P(D)$ is the total probability of the data, which is the chance of getting a vanilla cookie whether $H$ is true or not.  In this example, we can figure out $P(D)$ directly: because the bowls are equally likely, and they contain the same number of cookies, you were equally likely to choose any cookie.  Combining the two bowls, there are 50 vanilla and 30 chocolate cookies, so the probability of choosing a vanilla cookie is $50/80$ or $5/8$.

Now that we have the terms on the right-hand side, we can use Bayes's Theorem to combine them.

prior = 1/2
prior

likelihood = 3/4
likelihood

prob_data = 5/8
prob_data

posterior = prior * likelihood / prob_data
posterior

The posterior probability is $0.6$, a little higher than the prior, which was $0.5$.  

So the vanilla cookie makes us a little more certain that we chose Bowl 1.

**Exercise:** What if we had chosen a chocolate cookie instead; what would be the posterior probability of Bowl 1?

# Solution

prior = 1/2
likelihood = 1/4
prob_data = 3/8

posterior = prior * likelihood / prob_data
posterior

## Evidence

In the previous example and exercise, notice a pattern:

* A vanilla cookie is more likely if we chose Bowl 1, so getting a vanilla cookie makes Bowl 1 more likely.

* A chocolate cookie is less likely if we chose Bowl 1, so getting a chocolate cookie makes Bowl 1 less likely.

If data makes the probability of a hypothesis go up, we say that it is "evidence in favor" of the hypothesis.

If data makes the probability of a hypothesis go down, it is "evidence against" the hypothesis.

Let's do another example:

> Suppose you have two coins in a box.  One is a normal coin with heads on one side and tails on the other, and one is a trick coin with heads on both sides.
>
> You choose a coin at random and see that one of the sides is heads.  Is this data evidence in favor of, or against, the hypothesis that you chose the trick coin?

See if you can figure out the answer before you read my solution.  I suggest these steps:

1. First, state clearly what is the hypothesis and what is the data.

2. Then think about the prior, the likelihood of the data, and the total probability of the data.

3. Apply Bayes's Theorem to compute the posterior probability of the hypothesis.

4. Use the result to answer the question as posed.

In this example:

* $H$ is the hypothesis that you chose the trick coin with two heads.

* $D$ is the observation that one side of the coin is heads.

Now let's think about the right-hand terms:

* The prior is 1/2 because we were equally likely to choose either coin.

* The likelihood is 1 because if we chose the the trick coin, we would necessarily see heads.

* The total probability of the data is 3/4 because 3 of the 4 sides are heads, and we were equally likely to see any of them.

Here's what we get when we apply Bayes's theorem:

prior = 1/2
likelihood = 1
prob_data = 3/4

posterior = prior * likelihood / prob_data
posterior

The posterior is greater than the prior, so this data is evidence *in favor of* the hypothesis that you chose the trick coin.

And that makes sense, because getting heads is more likely if you choose the trick coin rather than the normal coin.

## The Bayes table

In the cookie problem and the coin problem we were able to compute the probability of the data directly, but that's not always the case.  In fact, computing the total probability of the data is often the hardest part of the problem.

Fortunately, there is another way to solve problems like this that makes it easier: the Bayes table.

You can write a Bayes table on paper or use a spreadsheet, but in this notebook I'll use a Pandas DataFrame.

I'll do the cookie problem first.  Here's an empty DataFrame with one row for each hypothesis:

import pandas as pd

table = pd.DataFrame(index=['Bowl 1', 'Bowl 2'])

Now I'll add a column to represent the priors:

table['prior'] = 1/2, 1/2
table

And a column for the likelihoods:

table['likelihood'] = 3/4, 1/2
table

Here we see a difference from the previous method: we compute likelihoods for both hypotheses, not just Bowl 1:

* The chance of getting a vanilla cookie from Bowl 1 is 3/4.

* The chance of getting a vanilla cookie from Bowl 2 is 1/2.

The next step is similar to what we did with Bayes's Theorem; we multiply the priors by the likelihoods:

table['unnorm'] = table['prior'] * table['likelihood']
table

I called the result `unnorm` because it is an "unnormalized posterior".  To see what that means, let's compare the right-hand side of Bayes's Theorem:

$P(H) P(D|H)~/~P(D)$

To what we have computed so far:

$P(H) P(D|H)$

The difference is that we have not divided through by $P(D)$, the total probability of the data.  So let's do that.

There are two ways to compute $P(D)$:

1. Sometimes we can figure it out directly.

2. Otherwise, we can compute it by adding up the unnormalized posteriors.

I'll show the second method computationally, then explain how it works.

Here's the total of the unnormalized posteriors:

prob_data = table['unnorm'].sum()
prob_data

Notice that we get 5/8, which is what we got by computing $P(D)$ directly.

Now we divide by $P(D)$ to get the posteriors:

table['posterior'] = table['unnorm'] / prob_data
table

The posterior probability for Bowl 1 is 0.6, which is what we got using Bayes's Theorem explicitly.

As a bonus, we also get the posterior probability of Bowl 2, which is 0.4.

The posterior probabilities add up to 1, which they should, because the hypotheses are "complementary"; that is, either one of them is true or the other, but not both.  So their probabilities have to add up to 1.

When we add up the unnormalized posteriors and divide through, we force the posteriors to add up to 1.  This process is called "normalization", which is why the total probability of the data is also called the "[normalizing constant](https://en.wikipedia.org/wiki/Normalizing_constant#Bayes'_theorem)"

It might not be clear yet why the unnormalized posteriors add up to $P(D)$.  I'll come back to that in the next notebook.

**Exercise:** Solve the trick coin problem using a Bayes table:

> Suppose you have two coins in a box.  One is a normal coin with heads on one side and tails on the other, and one is a trick coin with heads on both sides.
>
> You choose a coin at random and see the one of the sides is heads.  What is the posterior probability that you chose the trick coin?

Hint: The answer should still be 2/3.

# Solution

table = pd.DataFrame(index=['Trick', 'Normal'])
table['prior'] = 1/2, 1/2
table['likelihood'] = 1, 1/2
table['unnorm'] = table['prior'] * table['likelihood']
prob_data = table['unnorm'].sum()
table['posterior'] = table['unnorm'] / prob_data
table

## Summary



In this notebook I introduced two example problems: the cookie problem and the trick coin problem.  

We solved both problem using Bayes's Theorem; then I presented the Bayes table, a method for solving problems where it is hard to compute the total probability of the data directly.

[In the next notebook](https://colab.research.google.com/github/AllenDowney/BiteSizeBayes/blob/master/04_dice.ipynb), we'll see examples with more than two hypotheses, and I'll explain more carefully how the Bayes table works.

