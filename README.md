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
-> see test_data.csv 

## Example use case
Either use the provided main.py file, or run:

``` python
import pandas as pd
from opportunistic_planning import processing, visualization

# read data
data = pd.read_csv('test_data.csv', header=0)

# generate distances dictionary to reduce computation time
distances_dict = processing.generate_distances_dict(data)

# calculate prediction error (choose between 'prequential' or 'editdist')
results = processing.calculate_prediction_error(data, distances_dict=distances_dict, 
                                                error_function='prequential',
                                                n=10, dimensions=[[2, 'xy'],[3, 'xyz']])

# return parameter combination with lowest prediction error
lowest_mean, lowest_mean_idx, lowest_median, results_median = processing.get_lowest_error(results)

#print(lowest_mean, lowest_mean_idx, lowest_median)

# plot error values clustered by dimension
visualization.plot_dimensions(results)

```
