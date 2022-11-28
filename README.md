# Opportunistic Planning Model
Implementation of an opportunistic planning model to generate action sequence predictions for human behavior in everyday activities. When generating predictions, the model assumes that for each next step, the lowest-cost action is chosen.

Equation for weighted cost (C):

![formula](https://render.githubusercontent.com/render/math?math=C_{p,q}%20=%20d(p,%20q)%20^%20k%20\cdot%20c)

Parameters:
- d: Euclidean distance between p and q
- k: factor for relational dependencies
    - *food_k* indicates that the given dish has (warm) food on it and so is brought to the table at a later point in the sequence
    - *strong_k* is used for items that go below all other items, e.g., place mats or table cloths (normally taken first)
    - *mid_k* indicates items that need to be taken before other items (e.g., plate to define place setting on the table, saucer before cup, etc.) 
- c: factor for topology (containment)

Predicted sequences are compared to observed sequences to evaluate model performance and to find the optimal parameter combination. Depending on which error function is used, the comparison either uses Damerau-Levenshtein edit distance as a similarity measure (having predicted the whole sequence in advance) or the accumulated prediction error for the prequential method (only one next action predicted in each step, which is then compared to the observed action).

## Input
- Spatial information about the task environment: item locations, subject location in each step, topology
- Context knowledge: relational dependencies between items
- Observed action sequence for the task

:arrow_right: *'test_data.csv'* provides example data from table setting episodes in the needed format

## Example use case
Either use the provided *'main.py'* file, or run:

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

## Dataset references
- Damen, D. et al. (2018). Scaling egocentric vision: The EPIC-KITCHENS dataset. ECCV 2018,
720–736  ([EPIC-KITCHENS data set](https://epic-kitchens.github.io/2022))
- Meier, M., Mason, C., Porzel, R., Putze, F., & Schultz, T. (2018): Synchronized Multimodal Recording of a Table Setting Dataset. IROS 2018: Workshop on Latest Advances in Big Activity Data Sources for Robotics & New Challenges
- Rohrbach, M., Rohrbach, A., Regneri, M., Amin, S., Andriluka, M., Pinkal, M., Schiele, B. (2015): Recognizing Fine-Grained and Composite Activities using Hand-Centric Features and Script Data, IJCV 2015 ([MPII Cooking 2 Dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/human-activity-recognition/mpii-cooking-2-dataset))
- Rybok, L., Friedberger, S., Hanebeck, U. D., and Stiefelhagen, R. (2011): The KIT Robo-Kitchen Data set for the Evaluation of View-based Activity Recognition Systems, IEEE-RAS International Conference on Humanoid Robots, Bled, Slovenia, October 2011 ([KIT Robo-Kitchen Data set](https://cvhci.anthropomatik.kit.edu/~lrybok/projects/kitchen/))
- Tenorth, M., Bandouch, J., & Beetz, M. (2009): The TUM kitchen data set of everyday manipulation activities for motion tracking and action recognition. IEEE International
Workshop in conjunction with ICCV2009 ([TUM Kitchen Dataset](https://ias.in.tum.de/dokuwiki/software/kitchen-activity-data))
- Uhde, C., Berberich, N., Ramirez-Amaro, K., and Cheng, C. (2020): The Robot as Scientist: Using Mental Simulation to Test Causal Hypotheses Extracted from Human Activities in Virtual Reality, IROS 2020 ([HAVE DatA Set](https://www.ce.cit.tum.de/ics/ics-data-sets/have-data-set/))
