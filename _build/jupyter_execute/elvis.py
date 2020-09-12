# Bite Size Bayes

Copyright 2020 Allen B. Downey

License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## The Elvis problem

Here's a problem from [*Bayesian Data Analysis*](http://www.stat.columbia.edu/~gelman/book/):

> Elvis Presley had a twin brother (who died at birth).  What is the probability that Elvis was an identical twin?

I will answer this question in three steps:

1. First, we need some background information about the relative frequencies of identical and fraternal twins.

2. Then we will use Bayes's Theorem to take into account one piece of data, which is that Elvis's twin was male.

3. Finally, living up to the name of this blog, I will overthink the problem by taking into account a second piece of data, which is that Elvis's twin died at birth.

For background information, I'll use data from 1935, the year Elvis was born, from the
U.S. Census Bureau, [Birth, Stillbirth, and Infant Mortality Statistics for the Continental United States, the Territory of Hawaii, the Virgin Islands 1935](https://www.cdc.gov/nchs/data/vsushistorical/birthstat_1935.pdf).

It includes this table:

<img width="300" src="https://github.com/AllenDowney/BiteSizeBayes/raw/master/birth_data_1935.png">

The table doesn't report which twins are identical or fraternal, but we can use the data to compute

* $x$: The fraction of twins that are opposite sex.

opposite = 8397
same = 8678 + 8122

x = opposite / (opposite + same)
x

The quantity we want is

* $f$: The fraction of twins who are fraternal.

So let's see how we can get from $x$ to $f$.

Because identical twins have the same genes, they are almost always the same sex.  Fraternal twins do not have the same genes; like other siblings, they are about equally likely to be the same or opposite sex.

So we can write this relationship between $x$ and $f$

$x = f/2 + 0$

which says that the opposite sex twins include half of the fraternal twins and none of the identical twins.

And that implies

$f = 2x$

f = 2*x
f

In 1935, about 2/3 of twins were fraternal, and 1/3 were identical.

Getting back to the Elvis problem, we can use $1-f$ and $f$ as prior probabilities for the two hypotheses, `identical` and `fraternal`:

index = ['identical', 'fraternal']
prior = 1-f, f

Now we can take into account the data:

* $D$: Elvis's twin was male.

The probability of $D$ is nearly 100% if they were identical twins and about 50% if they were fraternal.

likelihood = 1, 0.5

Here's a function that takes the hypotheses, the priors, and the likelihoods and puts them in a Bayes table:

import pandas as pd

def make_bayes_table(index, prior, likelihood):
    """Make a Bayes table.
    
    index: sequence of hypotheses
    prior: prior probabilities
    likelihood: sequence of likelihoods
    
    returns: DataFrame
    """
    table = pd.DataFrame(index=index)
    table['prior'] = prior
    table['likelihood'] = likelihood
    table['unnorm'] = table['prior'] * table['likelihood']
    prob_data = table['unnorm'].sum()
    table['posterior'] = table['unnorm'] / prob_data
    return table

And here are the results

table = make_bayes_table(index, prior, likelihood)
table

From the Bayes table I'll extract $p_i$, which is the probability that same sex twins are identical:

p_i = table['posterior']['identical']
p_i

With priors based on data from 1935, the posterior probability that Elvis was a twin is close to 50%.

But there is one more piece of data to take into account; the fact that Elvis's twin died at birth.

Let's assume that there are different risks for fraternal and identical twins.  The quantities we want are

* $r_f$: The probability that one twin is stillborn, given that they are fraternal.

* $r_i$: The probability that one twin is stillborn, given that they are identical.

We can't get those quantities directly from the table, but we can compute:

* $y$: the probability of "1 living", given that the twins are opposite sex

* $z$: the probability of "1 living", given that the twins are the same sex

y = (258 + 299) / opposite
y

z = (655 + 564) / same
z

Assuming that all opposite sex twins are fraternal, we can infer that the risk for fraternal twins is $y$, the risk for opposite sex twins:

$r_f = y$

r_f = y
r_f

And because we know the fraction of same sex twins who are identical, $p_i$, we can write the following relation

$z = p_i r_i + (1-p_i) r_f$

which says that the risk for same sex twins is the weighted sum of the risks for identical and fraternal twins, with the weight $p_i$.

Solving for $r_i$, we get

$r_i = 2z - r_f$

And we have already computed $z$ and $r_f$

r_i = 2*z - r_f
r_i

In this dataset, it looks like the probability of "1 alive" is a little higher for identical twins.

So we can do a second update to take into account the data that Elvis's twin died at birth.   The posterior probabilities from the first update become the priors for the second.

prior2 = table['posterior']

Here are the likelihoods:

likelihood2 = r_i, r_f

And here are the results.

table2 = make_bayes_table(index, prior2, likelihood2)
table2

With the new data, the posterior probability that Elvis was an identical twin is about 54%.

**Credit:** Thanks to Jonah Spicher, who took my Bayesian Stats class at Olin and came up with the idea to use data from 1935 and take into account the fact that Elvis's twin died at birth.

