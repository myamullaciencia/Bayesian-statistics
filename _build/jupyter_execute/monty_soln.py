# Bite Size Bayes

Copyright 2020 Allen B. Downey

License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def prob(A):
    """Computes the probability of a proposition, A.
    
    A: Boolean series
    
    returns: probability
    """
    return A.mean()

def conditional(A, B):
    """Conditional probability of A given B.
    
    A: Boolean series
    B: Boolean series
    
    returns: probability
    """
    return prob(A[B])

## Two coins

> Suppose I flip two fair coins and tell you (honestly) that at least one of the coins is heads.  What is the probability that both coins are heads?

The answer is 1/3, and here's an argument that explains it.

1. If you toss two coins, there are 4 equally likely outcomes: HH, HT, TH, TT

2. If I tell you that at least one is heads, that eliminates TT.  

3. The remaining 3 outcomes are still equally likely, so their probability is now 1/3 each.

4. Therefore, the probability of HH is now 1/3.

However, you might still have some doubts.  For me, Step 3 feels like an unsupported assertion: How do we know the 3 remaining outcomes are still equally likely?

The following simulation might help convince you.

First I'll generate two sets of coin flips.

size = 10000
first = np.random.choice(['H', 'T'], size)
second = np.random.choice(['H', 'T'], size)

We can confirm that each coin has a 50% chance of landing heads:

prob(first == 'H')

prob(second == 'H')

Now we can compute a Boolean Series that is `True` if either coin landed heads, or both.

at_least_one = (first == 'H') | (second == 'H')

And we can confirm that happens 75% of the time:

prob(at_least_one)

We can compute a Boolean Series that is `True` if both coins landed heads.

both = (first == 'H') & (second == 'H')

And confirm that it happens 25% of the time.

prob(both)

Finally, we can compute the conditional probability of `both` given `at_least_one`:

conditional(both, at_least_one)

## The Monty Hall problem

From [Wikipedia](https://en.wikipedia.org/wiki/Monty_Hall_problem):

> Suppose you're on a game show, and you're given the choice of three doors: Behind one door is a car; behind the others, goats. You pick a door, say No. 1, and the host, who knows what's behind the doors, opens another door, say No. 3, which has a goat. He then says to you, "Do you want to pick door No. 2?" Is it to your advantage to switch your choice?

To avoid ambiguities, we have to make some assumptions about the behavior of the host:

1. The host never opens the door you picked.

2. The host never opens the door with the car.

3. If you choose the door with the car, the host chooses one of the other doors at random.

4. The host always offers you the option to switch.

Under these assumptions, are you better off sticking or switching?

The correct answer is that you are better off switching.  If you stick, you win 1/3 of the time.  If you switch, you win 2/3 of the time.

Here's one of many arguments that might persuade you.

> If you always stick, you win if you initially choose the door with the car, so the probability is 1/3.
>
> If you always switch, you win if you did _not_ choose the door with the car, so the probability is 2/3.

However, many people do not find any verbal arguments persuasive.  So, maybe a simulation will help.

**Exercise:** Write a simulation that confirms that you are better off switching if the host opens Door 3.

# Solution

df = pd.DataFrame()

size = 10000
df['winner'] = np.random.choice([1,2,3], size)

# Solution

win1 = (df['winner'] == 1)
win2 = (df['winner'] == 2)
win3 = (df['winner'] == 3)

np.mean(win1), np.mean(win2), np.mean(win3)

# Solution

df.loc[win1, 'open'] = np.random.choice([2,3], np.sum(win1))
df.loc[win2, 'open'] = 3
df.loc[win3, 'open'] = 2

df.head()

# Solution

open3 = (df['open'] == 3)

# Solution

conditional(win1, open3)

# Solution

conditional(win2, open3)

# Solution

conditional(win3, open3)

## Bayes's Theorem

In the previous two examples, you might have noticed a seeming contradiction:

* In the coin example, we start with four hypotheses with equal probability; one of them is eliminated by the data, and I argue that the other three still have equal probability.

* In the Monty Hall example, we start with three hypotheses with equal probability; one of them is eliminated by the data, but it turns out that the other two do _not_ have equal probability.

When one hypothesis is eliminated, its probability is redistributed to the remaining hypotheses, but it seems like there is no general rule for _how_ it is redistibuted.

Fortunately, Bayes's Theorem resolves this contradiction; if we apply it carefully, it tells us exactly how the probability should be redistributed.

First I'll solve the coin problem using a Bayes table.  Again, we start with four hypotheses with equal prior probability.  Just for fun, I'll use `Fraction` objects so the results are represented as rational numbers rather than floating-point.

from fractions import Fraction

hypos = ['HH', 'HT', 'TH', 'HH']
table = pd.DataFrame(index=hypos)
table['prior'] = Fraction(1, 4)
table

The data, in this example, is my report that there is at least one heads.  Assuming that I report honestly, we can compute the likelihood of the data under each hypothesis.

table['likelihood'] = [1, 1, 1, 0]
table

And we fill in the rest of the table in the usual way.

table['unnorm'] = table['prior'] * table['likelihood']
prob_data = table['unnorm'].sum()
table['posterior'] = table['unnorm'] / prob_data
table

In this example the remaining hypotheses have the same posterior probability because the likelihood of the data is the same under any of them.

For the Monty Hall problem, that is not the case.  We'll start with three hypotheses, one for each door, and equal priors.

hypos = ['Door 1', 'Door 2', 'Door 3']
table = pd.DataFrame(index=hypos)
table['prior'] = Fraction(1, 3)
table

Now, the data is that the host opens Door 3.  So the question is, what is the probability of the data under each hypothesis.  In reverse order:

* Door 3: If the car is behind Door 3 and you choose Door 1, the host has no choice but to open Door 2, so the probability that he opens Door 3 is 0.

* Door 2: If the car is behind Door 2 and you choose Door 1, the host has not choice but to open Door 3, so the probability that he opens Door 3 is 1.

* Door 1: If the car is behind Door 1 and you choose Door 1, the host has a choice, and the statement of the problem indicates that he chooses either Door 2 or Door 3 with equal probability, so the probability that he opens Door 3 is 1/2.

That's all we need to fill in the likelihoods:

table['likelihood'] = [Fraction(1,2), 1, 0]
table

And we fill in the rest of the table in the usual way.

table['unnorm'] = table['prior'] * table['likelihood']
prob_data = table['unnorm'].sum()
table['posterior'] = table['unnorm'] / prob_data
table

In this example the likelihood of the data is not the same under the remaining hypotheses, so the posterior probabilities are not the same.

**Exercise:** Here's a variation on the Monty Hall problem.  Suppose that whenever the host has a choice, he opens Door 3.  In that case, what are the posterior probabilities for the three doors?

# Solution

hypos = ['Door 1', 'Door 2', 'Door 3']
table = pd.DataFrame(index=hypos)
table['prior'] = Fraction(1, 3)

# Solution

table['likelihood'] = [1, 1, 0]
table

# Solution

table['unnorm'] = table['prior'] * table['likelihood']
prob_data = table['unnorm'].sum()
table['posterior'] = table['unnorm'] / prob_data
table

**Exercise:** Suppose that whenever the host has a choice he chooses Door 3 with probability `p` and Door 2 with probability `1-p`.  What are the posterior probabilities for the three doors?

Hint: If you use SymPy to create a symbol for `p`, it will carry through the computation.

from sympy import symbols

p = symbols('p')

# Solution

hypos = ['Door 1', 'Door 2', 'Door 3']
table = pd.DataFrame(index=hypos)
table['prior'] = Fraction(1, 3)

# Solution

table['likelihood'] = [p, 1, 0]
table

# Solution

table['unnorm'] = table['prior'] * table['likelihood']
prob_data = table['unnorm'].sum()
table['posterior'] = table['unnorm'] / prob_data
table

# Solution

table.loc['Door 1', 'posterior'].simplify()

# Solution

table.loc['Door 2', 'posterior'].simplify()

