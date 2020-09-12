# Bite Size Bayes

Copyright 2020 Allen B. Downey

License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Review

In [a previous notebook](https://colab.research.google.com/github/AllenDowney/BiteSizeBayes/blob/master/12_binomial.ipynb) we solved the Euro problem, which involved estimating the proportion of heads we get when we spin a coin on edge.

We used the posterior distribution to test whether the coin is fair or biased, but the answer is not entirely satisfying because it depends on how we define "biased".

In general, this kind of hypothesis testing is not the best use of a posterior distribution because it does not answer the question we really care about.  For practical purposes, it is less useful to know *whether* a coin is biased and more useful to know *how* biased.

In this notebook we solve the Bayesian bandit problem, which is similar in the sense that it involves estimating proportions, but different in the sense that we use the posterior distribution as part of a decision-making process.

## The Bayesian bandit problem

Suppose you have several "one-armed bandit" slot machines, and there's reason to think that they have different probabilities of paying off.

Each time you play a machine, you either win or lose, and you can use the outcome to update your belief about the probability of winning.

Then, to decide which machine to play next, you can use the "Bayesian bandit" strategy, explained below.

First, let's see how to do the update.

## The prior

If we know nothing about the probability of wining, we can start with a uniform prior.

def decorate_bandit(title):
    """Labels the axes.
    
    title: string
    """
    plt.xlabel('Probability of winning')
    plt.ylabel('PMF')
    plt.title(title)

xs = np.linspace(0, 1, 101)
prior = pd.Series(1/101, index=xs)

prior.plot()
decorate_bandit('Prior distribution')

## The likelihood function

The likelihood function that computes the probability of an outcome (W or L) for a hypothetical value of x, the probability of winning (from 0 to 1).

def update(prior, data):
    """Likelihood function for Bayesian bandit
    
    prior: Series that maps hypotheses to probabilities
    data: string, either 'W' or 'L'
    """
    xs = prior.index
    if data == 'W':
        prior *= xs
    else:
        prior *= 1-xs
        
    prior /= prior.sum()

bandit = prior.copy()
update(bandit, 'W')
update(bandit, 'L')
bandit.plot()
decorate_bandit('Posterior distribution, 1 loss, 1 win')

**Exercise 1:** Suppose you play a machine 10 times and win once.  What is the posterior distribution of $x$?

# Solution

bandit = prior.copy()

for outcome in 'WLLLLLLLLL':
    update(bandit, outcome)
    
bandit.plot()
decorate_bandit('Posterior distribution, 9 loss, one win')

## Multiple bandits

Now suppose we have several bandits and we want to decide which one to play.

For this example, we have 4 machines with these probabilities:

actual_probs = [0.10, 0.20, 0.30, 0.40]

The function `play` simulates playing one machine once and returns `W` or `L`.

from random import random
from collections import Counter

# count how many times we've played each machine
counter = Counter()

def flip(p):
    """Return True with probability p."""
    return random() < p

def play(i):
    """Play machine i.
    
    returns: string 'W' or 'L'
    """
    counter[i] += 1
    p = actual_probs[i]
    if flip(p):
        return 'W'
    else:
        return 'L'

Here's a test, playing machine 3 twenty times:

for i in range(20):
    result = play(3)
    print(result, end=' ')

Now I'll make four copies of the prior to represent our beliefs about the four machines.

beliefs = [prior.copy() for i in range(4)]

This function displays four distributions in a grid.

options = dict(xticklabels='invisible', yticklabels='invisible')

def plot(beliefs, **options):
    for i, b in enumerate(beliefs):
        plt.subplot(2, 2, i+1)
        b.plot(label='Machine %s' % i)
        plt.gca().set_yticklabels([])
        plt.legend()
        
    plt.tight_layout()

plot(beliefs)

**Exercise 2:** Write a nested loop that plays each machine 10 times; then plot the posterior distributions.  

Hint: call `play` and then `update`.

# Solution

for i in range(4):
    for _ in range(10):
        outcome = play(i)
        update(beliefs[i], outcome)

# Solution

plot(beliefs)

After playing each machine 10 times, we can summarize `beliefs` by printing the posterior mean and credible interval:

def pmf_mean(pmf):
    """Compute the mean of a PMF.
    
    pmf: Series representing a PMF
    
    return: float
    """
    return np.sum(pmf.index * pmf)

from scipy.interpolate import interp1d

def credible_interval(pmf, prob):
    """Compute the mean of a PMF.
    
    pmf: Series representing a PMF
    prob: probability of the interval
    
    return: pair of float
    """
    # make the CDF
    xs = pmf.index
    ys = pmf.cumsum()
    
    # compute the probabilities
    p = (1-prob)/2
    ps = [p, 1-p]
    
    # interpolate the inverse CDF
    options = dict(bounds_error=False,
                   fill_value=(xs[0], xs[-1]), 
                   assume_sorted=True)
    interp = interp1d(ys, xs, **options)
    return interp(ps)

for i, b in enumerate(beliefs):
    print(pmf_mean(b), credible_interval(b, 0.9))

## Bayesian Bandits

To get more information, we could play each machine 100 times, but while we are gathering data, we are not making good use of it.  The kernel of the Bayesian Bandits algorithm is that it collects and uses data at the same time.  In other words, it balances exploration and exploitation.

The following function chooses among the machines so that the probability of choosing each machine is proportional to its "probability of superiority".

def pmf_choice(pmf, n):
    """Draw a random sample from a PMF.
    
    pmf: Series representing a PMF
    
    returns: quantity from PMF
    """
    return np.random.choice(pmf.index, p=pmf)

def choose(beliefs):
    """Use the Bayesian bandit strategy to choose a machine.
    
    Draws a sample from each distributions.
    
    returns: index of the machine that yielded the highest value
    """
    ps = [pmf_choice(b, 1) for b in beliefs]
    return np.argmax(ps)

This function chooses one value from the posterior distribution of each machine and then uses `argmax` to find the index of the machine that chose the highest value.

Here's an example.

choose(beliefs)

**Exercise 3:** Putting it all together, fill in the following function to choose a machine, play once, and update `beliefs`:

def choose_play_update(beliefs, verbose=False):
    """Chose a machine, play it, and update beliefs.
    
    beliefs: list of Pmf objects
    verbose: Boolean, whether to print results
    """
    # choose a machine
    machine = ____
    
    # play it
    outcome = ____
    
    # update beliefs
    update(____)
    
    if verbose:
        print(i, outcome, beliefs[machine].mean())

# Solution

def choose_play_update(beliefs, verbose=False):
    """Chose a machine, play it, and update beliefs.
    
    beliefs: list of Pmf objects
    verbose: Boolean, whether to print results
    """
    # choose a machine
    machine = choose(beliefs)
    
    # play it
    outcome = play(machine)
    
    # update beliefs
    update(beliefs[machine], outcome)
    
    if verbose:
        print(i, outcome, beliefs[machine].mean())

Here's an example

choose_play_update(beliefs, verbose=True)

## Trying it out

Let's start again with a fresh set of machines and an empty `Counter`.

beliefs = [prior.copy() for i in range(4)]
counter = Counter()

If we run the bandit algorithm 100 times, we can see how `beliefs` gets updated:

num_plays = 100

for i in range(num_plays):
    choose_play_update(beliefs)
    
plot(beliefs)

We can summarize `beliefs` by printing the posterior mean and credible interval:

for i, b in enumerate(beliefs):
    print(pmf_mean(b), credible_interval(b, 0.9))

The credible intervals usually contain the true values (0.1, 0.2, 0.3, and 0.4).

The estimates are still rough, especially for the lower-probability machines.  But that's a feature, not a bug: the goal is to play the high-probability machines most often.  Making the estimates more precise is a means to that end, but not an end itself.

Let's see how many times each machine got played.  If things go according to plan, the machines with higher probabilities should get played more often.

for machine, count in sorted(counter.items()):
    print(machine, count)

**Exercise 4:**  Go back and run this section again with a different value of `num_play` and see how it does.

## Summary

The algorithm I presented in this notebook is called [Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling).  It is an example of a general strategy called [Bayesian decision theory](https://wiki.lesswrong.com/wiki/Bayesian_decision_theory), which is the idea of using a posterior distribution as part of a decision-making process, usually by choosing an action that minimizes the costs we expect on average (or maximizes a benefit).

In my opinion, this strategy is the biggest advantage of Bayesian methods over classical statistics.  When we represent knowledge in the form of probability distributions, Bayes's theorem tells us how to change our beliefs as we get more data, and Bayesian decisions theory tells us how to make that knowledge actionable.

