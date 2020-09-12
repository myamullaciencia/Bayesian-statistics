# Bite Size Bayes

Copyright 2020 Allen B. Downey

License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Review

[In the previous notebook](https://colab.research.google.com/github/AllenDowney/BiteSizeBayes/blob/master/05_test.ipynb) we used a Bayes table to interpret medical tests.

In this notebook we'll solve an expanded version of the cookie problem with 101 Bowls.  It might seem like a silly problem, but it's not: the solution demonstrates a Bayesian way to estimate a proportion, and it applies to lots of real problems that don't involve cookies.

Then I'll introduce an alternative to the Bayes table, a probability mass function (PMF), which is a useful way to represent and do computations with distributions.

Here's the function, from the previous notebook, we'll use to make Bayes tables:

import pandas as pd

def make_bayes_table(hypos, prior, likelihood):
    """Make a Bayes table.
    
    hypos: sequence of hypotheses
    prior: prior probabilities
    likelihood: sequence of likelihoods
    
    returns: DataFrame
    """
    table = pd.DataFrame(index=hypos)
    table['prior'] = prior
    table['likelihood'] = likelihood
    table['unnorm'] = table['prior'] * table['likelihood']
    prob_data = table['unnorm'].sum()
    table['posterior'] = table['unnorm'] / prob_data
    return table

## 101 Bowls

In [Notebook 4](https://colab.research.google.com/github/AllenDowney/BiteSizeBayes/blob/master/05_dice.ipynb), we saw that the Bayes table works with more than two hypotheses.  As an example, we solved a cookie problem with five bowls.

Now we'll take it even farther and solve a cookie problem with 101 bowls:

* Bowl 0 contains no vanilla cookies,

* Bowl 1 contains 1% vanilla cookies,

* Bowl 2 contains 2% vanilla cookies,

and so on, up to

* Bowl 99 contains 99% vanilla cookies, and

* Bowl 100 contains all vanilla cookies.

As in the previous problems, there are only two kinds of cookies, vanilla and chocolate.  So Bowl 0 is all chocolate cookies, Bowl 1 is 99% chocolate, and so on.

Suppose we choose a bowl at random, choose a cookie at random, and it turns out to be vanilla.  What is the probability that the cookie came from Bowl $x$, for each value of $x$?

To solve this problem, I'll use `np.arange` to represent 101 hypotheses, numbered from 0 to 100.

import numpy as np

xs = np.arange(101)

The prior probability for each bowl is $1/101$.  I could create a sequence with 101 identical values, but if all of the priors are equal, we only have to probide one value:

prior = 1/101

Because of the way I numbered the bowls, the probability of a vanilla cookie from Bowl $x$ is $x/100$.  So we can compute the likelihoods like this:

likelihood = xs/100

And that's all we need; the Bayes table does the rest:

table = make_bayes_table(xs, prior, likelihood)

Here's a feature we have not seen before: we can give the index of the Bayes table a name, which will appear when we display the table.

table.index.name = 'Bowl'

Here are the first few rows:

table.head()

Because Bowl 0 contains no vanilla cookies, its likelihood is 0, so its posterior probability is 0.  That is, the cookie cannot have come from Bowl 0.

Here are the last few rows of the table.

table.tail()

The posterior probabilities are substantially higher for the high-numbered bowls.

There is a pattern here that will be clearer if we plot the results.

import matplotlib.pyplot as plt

def plot_table(table):
    """Plot results from the 101 Bowls problem.
    
    table: DataFrame representing a Bayes table
    """
    table['prior'].plot()
    table['posterior'].plot()

    plt.xlabel('Bowl #')
    plt.ylabel('Probability')
    plt.legend()

plot_table(table)
plt.title('One cookie');

The prior probabilities are uniform; that is, they are the same for every bowl.

The posterior probabilities increase linearly; Bowl 0 is the least likely (actually impossible), and Bowl 100 is the most likely.

## Two cookies

Suppose we put the first cookie back, stir the bowl thoroughly, and draw another cookie from the same bowl.  and suppose it turns out to be another vanilla cookie.

Now what is the probability that we are drawing from Bowl $x$?

To answer this question, we can use the posterior probabilities from the previous problem as prior probabilities for a new Bayes table, and then update with the new data.

prior2 = table['posterior']
likelihood2 = likelihood

table2 = make_bayes_table(xs, prior2, likelihood2)
plot_table(table2)
plt.title('Two cookies');

The blue line shows the posterior after one cookie, which is the prior before the second cookie.

The orange line shows the posterior after two cookies, which curves upward.  Having see two vanilla cookies, the high-numbered bowls are more likely; the low-numbered bowls are less likely.

I bet you can guess what's coming next.

## Three cookies

Suppose we put the cookie back, stir, draw another cookie from the same bowl, and get a chocolate cookie.

What do you think the posterior distribution looks like after these three cookies?

Hint: what's the probability that the chocolate cookie came from Bowl 100?

We'll use the posterior after two cookies as the prior for the third cookie:

prior3 = table2['posterior']

Now, what about the likelihoods?  Remember that the probability of a vanilla cookie from Bowl $x$ is $x/100$.  So the probability of a chocolate cookie is $(1 - x/100)$, which we can compute like this.

likelihood3 = 1 - xs/100

That's it.  Everything else is the same.

table3 = make_bayes_table(xs, prior3, likelihood3)

And here are the results

plot_table(table3)
plt.title('Three cookies');

The blue line is the posterior after two cookies; the orange line is the posterior after three cookies.

Because Bowl 100 contains no chocolate cookies, the posterior probability for Bowl 100 is 0.

The posterior distribution has a peak near 60%.  We can use `idxmax` to find it: 

table3['posterior'].idxmax()

The peak in the posterior distribution is at 67%.

This value has a name; it is the **MAP**, which stands for "Maximum Aposteori Probability" ("aposteori" is Latin for posterior).

In this example, the MAP is close to the proportion of vanilla cookies in the dataset: 2/3.

**Exercise:** Let's do a version of the dice problem where we roll the die more than once.  Here's the statement of the problem again:

> Suppose you have a 4-sided, 6-sided, 8-sided, and 12-sided die.  You choose one at random, roll it and get a 1. What is the probability that the die you rolled is 4-sided?  What are the posterior probabilities for the other dice?

And here's a solution using a Bayes table:

hypos = ['H4', 'H6', 'H8', 'H12']
prior = 1/4
likelihood = 1/4, 1/6, 1/8, 1/12

table = make_bayes_table(hypos, prior, likelihood)
table

Now suppose you roll the same die again and get a 6.  What are the posterior probabilities after the second roll?

Use `idxmax` to find the MAP.

# Solution

prior2 = table['posterior']
likelihood2 = 0, 1/6, 1/8, 1/12

table2 = make_bayes_table(hypos, prior2, likelihood2)
table2

# Solution

table2['posterior'].idxmax()

## Probability Mass Functions

When we do more than one update, we don't always want to keep the whole Bayes table.  In this section we'll replace the Bayes table with a more compact representation, a probability mass function, or PMF.

A PMF is a set of possible outcomes and their corresponding probabilities.  There are many ways to represent a PMF; in this notebook I'll use a Pandas Series.

Here's a function that takes a sequence of outcomes, `xs`, and a sequence of probabilities, `ps`, and returns a Pandas Series that represents a PMF.

def make_pmf(xs, ps, **options):
    """Make a Series that represents a PMF.
    
    xs: sequence of values
    ps: sequence of probabilities
    options: keyword arguments passed to Series constructor
    
    returns: Pandas Series
    """
    pmf = pd.Series(ps, index=xs, **options)
    return pmf

And here's a PMF that represents the prior from the 101 Bowls problem.

xs = np.arange(101)
prior = 1/101

pmf = make_pmf(xs, prior)
pmf.head()

Now that we have a priod, we need to compute likelihoods.

Here are the likelihoods for a vanilla cookie:

likelihood_vanilla = xs / 100

And for a chocolate cookie.

likelihood_chocolate = 1 - xs / 100

To compute posterior probabilities, I'll use the following function, which takes a PMF and a sequence of likelihoods, and updates the PMF:

def bayes_update(pmf, likelihood):
    """Do a Bayesian update.
    
    pmf: Series that represents the prior
    likelihood: sequence of likelihoods
    """
    pmf *= likelihood
    pmf /= pmf.sum()

The steps here are the same as in the Bayes table:

1. Multiply the prior by the likelihoods.

2. Add up the products to get the total probability of the data.

3. Divide through to normalize the posteriors.

Now we can do the update for a vanilla cookie.

bayes_update(pmf, likelihood_vanilla)

Here's what the PMF looks like after the update.

pmf.plot()

plt.xlabel('Bowl #')
plt.ylabel('Probability')
plt.title('One cookie');

That's consistent with what we got with the Bayes table.

The advantage of using a PMF is that it is easier to do multiple updates.  The following cell starts again with the uniform prior and does updates with two vanilla cookies and one chocolate cookie:

data = 'VVC'

pmf = make_pmf(xs, prior)

for cookie in data:
    if cookie == 'V':
        bayes_update(pmf, likelihood_vanilla)
    else:
        bayes_update(pmf, likelihood_chocolate)

Here's what the results look like:

pmf.plot()

plt.xlabel('Bowl #')
plt.ylabel('Probability')
plt.title('Three cookies');

Again, that's consistent with what we got with the Bayes table.

In the next section, I'll use a PMF and `bayes_update` to solve a dice problem.

## The dice problem

As an exercise, let's do one more version of the dice problem:

> Suppose you have a 4-sided, 6-sided, 8-sided, 12-sided, and a **20-sided die**.  You choose one at random, roll it and **get a 7**. What is the probability that the die you rolled is 4-sided?  What are the posterior probabilities for the other dice?

Notice that in this version I've added a 20-sided die and the outcome is 7, not 1.

Here's a PMF that represents the prior:

sides = np.array([4, 6, 8, 12, 20])
prior = 1/5

pmf = make_pmf(sides, prior)
pmf

In this version, the hypotheses are integers rather than strings, so we can compute the likelihoods like this:

likelihood = 1 / sides

But the outcome is 7, so any die with fewer than 7 sides has likelihood 0.

We can adjust `likelihood` by making a Boolean Series:

too_low = (sides < 7)

And using it to set the corresponding elements of `likelihood` to 0.

likelihood[too_low] = 0
likelihood

Now we can do the update and display the results.

bayes_update(pmf, likelihood)
pmf

The 4-sided and 6-sided dice have been eliminated.  Of the remaining dice, the 8-sided die is the most likely.

**Exercise:** Suppose you have the same set of 5 die.  You choose a die, roll it six times, and get 6, 7, 2, 5, 1, and 2 again.  Use `idxmax` to find the MAP.  What is the posterior probability of the MAP?

# Solution

sides = np.array([4, 6, 8, 12, 20])
prior = 1/5

pmf = make_pmf(sides, prior)

# Solution

outcomes = [6, 7, 2, 5, 1, 2]

for outcome in outcomes:
    likelihood = 1 / sides
    too_low = (sides < outcome)
    likelihood[too_low] = 0
    
    bayes_update(pmf, likelihood)

pmf

# Solution

MAP = pmf.idxmax()
MAP

# Solution

pmf[MAP]

## Summary

In this notebook, we extended the cookie problem with more bowls and the dice problem with more dice.

I defined the MAP, which is the quantity in a posterior distribution with the highest probability.

Although the cookie problem is not particularly realistic or useful, the method we used to solve it applies to many problems in the real world where we want to estimate a proportion.

[In the next notebook](https://colab.research.google.com/github/AllenDowney/BiteSizeBayes/blob/master/07_euro.ipynb) we'll use the same method to take another step toward doing Bayesian statistics.

