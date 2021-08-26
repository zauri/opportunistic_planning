#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from opportunistic_planning import processing, visualization
from scipy.stats import friedmanchisquare, wilcoxon

# read in data
data = pd.read_csv('test_data.csv', header=0)

# generate distances dictionary
distances_dict = processing.generate_distances_dict(data)

# calculate prediction error for all parameter values
results = processing.calculate_prediction_error(data, distances_dict, error_function='prequential',
          n=10, dimensions=[[2, 'xy'], [3, 'xyz']])

# return lowest parameter combination with lowest prediction error
lowest_mean, lowest_mean_idx, lowest_median, results_mean = processing.get_lowest_error(results)

# plot error values clustered by dimension
visualization.plot_dimensions(results)
