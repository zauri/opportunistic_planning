# Opportunistic Planning Model
Implementation of an opportunistic planning model to generate action sequence predictions for human behavior in everyday activities. When generating predictions, the model assumes that for each next step, the lowest-cost action is chosen.

Equation for weighted cost (C):

![formula](https://render.githubusercontent.com/render/math?math=C_{p,q}%20=%20d(p,%20q)%20^%20k%20\cdot%20c)

Parameters:
- d: Euclidean distance between p and q
- k: factor for relational dependencies
- c: factor for topology (containment)

Predicted sequences are compared to observed sequences to evaluate model performance and to find the optimal parameter combination. Depending on which error function is used, the comparison either uses Damerau-Levenshtein edit distance as a similarity measure (having predicted the whole sequence in advance) or the accumulated prediction error for the prequential method (only one next action predicted in each step, which is then compared to the observed action).

## Input
- Spatial information about the task environment: item locations, subject location in each step, topology
- Context knowledge: relational dependencies between items
- Observed action sequence for the task

## Example use case
``` python
import pandas as pd
import numpy as np
import ast
from opportunistic_planning import processing, prediction

data = pd.read_csv(filename, header=0)
distances_dict = processing.generate_distances_dict(data)
results = processing.calculate_prediction_error(data, distances_dict=distances_dict, 
                                                error_function='prequential',
                                                dimensions=[[2, 'xy]], n=50)

lowest_mean, lowest_mean_idx, lowest_median, results_median = processing.get_lowest_error(results)

print(lowest_mean, lowest_mean_idx, lowest_median)

```
