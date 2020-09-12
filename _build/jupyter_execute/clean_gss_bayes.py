# Bite Size Bayes


Copyright 2020 Allen B. Downey

MIT License: https://opensource.org/licenses/MIT

import pandas as pd
import numpy as np

The dataset includes variables I selected from the General Social Survey, available from this project on the GSS site: https://gssdataexplorer.norc.org/projects/54786

I also store the data in the GitHub repository for this book; the following cell downloads it, if necessary.

# Load the data file

import os

if not os.path.exists('gss_bayes.tar.gz'):
    !wget https://github.com/AllenDowney/BiteSizeBayes/raw/master/gss_bayes.tar.gz
    !tar -xzf gss_bayes.tar.gz

`utils.py` provides `read_stata`, which reads the data from the Stata format.

from utils import read_stata

gss = read_stata('GSS.dct', 'GSS.dat')
gss.rename(columns={'id_': 'caseid'}, inplace=True)
gss.index = gss['caseid']
gss.head()

def replace_invalid(series, bad_vals, replacement=np.nan):
    """Replace invalid values with NaN

    Modifies series in place.

    series: Pandas Series
    bad_vals: list of values to replace
    replacement: value to replace
    """
    series.replace(bad_vals, replacement, inplace=True)

The following cell replaces invalid responses for the variables we'll use.

replace_invalid(gss['feminist'], [0, 8, 9])
replace_invalid(gss['polviews'], [0, 8, 9])
replace_invalid(gss['partyid'], [8, 9])
replace_invalid(gss['indus10'], [0, 9997, 9999])
replace_invalid(gss['age'], [0, 98, 99])

def values(series):
    """Make a series of values and the number of times they appear.
    
    series: Pandas Series
    
    returns: Pandas Series
    """
    return series.value_counts(dropna=False).sort_index()

### feminist

https://gssdataexplorer.norc.org/variables/1698/vshow

This question was only asked during one year, so we're limited to a small number of responses.

values(gss['feminist'])

### polviews

https://gssdataexplorer.norc.org/variables/178/vshow


values(gss['polviews'])

### partyid

https://gssdataexplorer.norc.org/variables/141/vshow

values(gss['partyid'])

### race

https://gssdataexplorer.norc.org/variables/82/vshow

values(gss['race'])

### sex

https://gssdataexplorer.norc.org/variables/81/vshow

values(gss['sex'])

### age



values(gss['age'])

### indus10

https://gssdataexplorer.norc.org/variables/17/vshow

values(gss['indus10'])

## Select subset

Here's the subset of the data with valid responses for the variables we'll use.

varnames = ['year', 'age', 'sex', 'polviews', 'partyid', 'indus10']

valid = gss.dropna(subset=varnames)
valid.shape

subset = valid[varnames]
subset.head()

## Save the data

subset.to_csv('gss_bayes.csv')

!ls -l gss_bayes.csv

