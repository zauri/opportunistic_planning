import numpy as np
import random
from fastDamerauLevenshtein import damerauLevenshtein
from collections import Counter

def filter_for_dimension(dimension, coordinates, start_coordinates):
    '''
    Filter coordinates and start coordinates for given dimension (e.g., xyz -> xy).

    Parameters
    ----------
    dimension : list of [int, str]
        Dimension for which to adapt coordinates (default before filtering: 3D).
    coordinates : dictionary
        Coordinates of objects in 3D.
    start_coordinates : list
        List of start coordinates where subject is standing before next picking_up action
        in 3D.

    Returns
    -------
    new_coords : dictionary
        Dictionary with filtered coordinates.
    new_start_coords : list
        List with filtered start coordinates.

    '''
    
    new_coords =  {}
    new_start_coords = []
    
    if dimension[0] == 3:  # no changes if 3D
        new_coords = coordinates
        new_start_coords = start_coordinates

    elif dimension[0] == 2:  # 2D: remove obsolete coordinate
        if dimension[1] == 'xy':
            new_coords = {key: value[:-1] for key, value in coordinates.items()}
            new_start_coords = [x[:-1] for x in start_coordinates]

        elif dimension[1] == 'xz':
            new_start_coords = [[x[0], x[-1]] for x in start_coordinates]

            for key, value in coordinates.items():
                new_value = (value[0], value[-1])
                new_coords[key] = new_value

        elif dimension[1] == 'yz':
            new_coords = {key: value[1:] for key, value in coordinates.items()}
            new_start_coords = [x[1:] for x in start_coordinates]

    elif dimension[0] == 1:  # 1D: choose appropriate coordinate
        if dimension[1] == 'x':
            new_coords = {key: value[0] for key, value in coordinates.items()}
            new_start_coords = [x[0] for x in start_coordinates]

        elif dimension[1] == 'y':
            new_coords = {key: value[1] for key, value in coordinates.items()}
            new_start_coords = [x[1] for x in start_coordinates]

        elif dimension[1] == 'z':
            new_coords = {key: value[2] for key, value in coordinates.items()}
            new_start_coords = [x[2] for x in start_coordinates]
            
    return new_coords, new_start_coords


def predict_editdist(distances_dict, ID, objects, coordinates, start_coordinates, sequence,
                     c, k, dimension=[3, ]):
    '''
    Predict sequence based on spatial properties of objects and environment.

    Parameters
    ----------
    distances_dict : dictionary
        Dictionary containing distances between objects in all dimensions.
    ID : str
        Identifier for episode.
    objects : list
        Objects in episode.
    coordinates : dictionary
        Coordinates of objects.
    start_coordinates : list
        List of coordinates where subject is standing before each picking-up action.
    sequence : str
        Observed sequence of objects in episode.
    c : dictionary
        Parameter values for containment for all objects.
    k : dictionary
        Parameter values for relational dependencies for all objects.
    dimension : list [int, str], optional
        Dimension in which to consider distances. The default is [3, ].

    Returns
    -------
    prediction : str
        Predicted sequence of objects.

    '''
    
    prediction = []
    possible_items = dict.fromkeys(objects, 0)  # generate dict from object list
    coord_index = 0
    
    new_coords, new_start_coords = filter_for_dimension(dimension, coordinates, start_coordinates)

    while bool(possible_items) == True:  # while dict not empty
        for obj in possible_items.keys():            
            try:
                position = tuple(new_start_coords[coord_index])
            except TypeError:
                position = str(new_start_coords[coord_index])
            
            possible_items[obj] = distances_dict[dimension[1]][ID][position][obj] ** k[obj] * c[obj]

        minval = min(possible_items.values())
        minval = [k for k, v in possible_items.items() if v == minval]
        minval = random.choice(minval)  # choose prediction randomly if multiple items have same cost
        prediction.append(minval)
        del possible_items[minval]
        coord_index += 1

    return prediction


def predict_prequential(distances_dict, ID, objects, coordinates, start_coordinates, sequence, 
                                 c, k, dimension=[3, ]):
    '''
    Predict sequence based on prequential method (predict one step, compare with observed behavior,
    error measure: 0 if predicted == observed, 1 if predicted != observed).

    Parameters
    ----------
    distances_dict : dictionary
        Dictionary containing distances between objects in all dimensions.
    ID : str
        Identifier for episode.
    objects : list
        Objects in episode.
    coordinates : dictionary
        Coordinates of objects.
    start_coordinates : list
        List of coordinates where subject is standing before each picking-up action.
    sequence : str
        Observed sequence of objects in episode.
    c : dictionary
        Parameter values for containment for all objects.
    k : dictionary
        Parameter values for relational dependencies for all objects.
    dimension : list [int, str], optional
        Dimension in which to consider distances. The default is [3, ].

    Returns
    -------
    errors : list
        List of error values for observed vs predicted sequence.

    '''
    
    i = 0
    errors = []
    possible_items = dict.fromkeys(objects, 0)  # generate dict from object list
    item_count = Counter(objects)
    
    coord_index = 0
    
    new_coords, new_start_coords = filter_for_dimension(dimension, coordinates, start_coordinates)

    while i < len(sequence) - 1:
        for obj in possible_items.keys():            
            try:
                position = tuple(new_start_coords[coord_index])
            except TypeError:
                position = str(new_start_coords[coord_index])
            
            possible_items[obj] = distances_dict[dimension[1]][ID][position][obj] ** k[obj] * c[obj]

        minval = min(possible_items.values())
        minval = [k for k, v in possible_items.items() if v == minval]
        minval = random.choice(minval)  # choose prediction randomly if multiple items have same cost
        
        prediction = minval
        observed = sequence[i]
        error = 1 - damerauLevenshtein(prediction, observed)
        
        errors.append(error)
        
        if item_count[sequence[i]] > 1:
            item_count[sequence[i]] = item_count[sequence[i]] - 1
        else:
            del possible_items[sequence[i]]
        
        coord_index += 1
        i += 1
    
    return errors


def get_median_error(error_function, row, ID, objects, coordinates, start_coordinates, c, k, dimension, sequence, 
                             distances_dict, n=1):
    '''
    Return median error for chosen error measure (editdist or prequential) for n trials.

    Parameters
    ----------
    error_function : function
        Error measure to use: editdist or prequential.
    row : int
        Row number in dataframe.
    ID : str
        Identifier for episode.
    objects : list
        Objects in episode.
    coordinates : dictionary
        Coordinates of objects.
    start_coordinates : list
        List of coordinates where subject is standing before each picking-up action.
    c : dictionary
        Parameter values for containment for all objects.
    k : dictionary
        Parameter values for relational dependencies for all objects.
    dimension : list [int, str]
        Dimension in which to consider distances. The default is [3, ].
    sequence : str
        Observed sequence of objects in episode.
    distances_dict : dictionary
        Dictionary containing distances between objects in all dimensions.
    n : int, optional
        Number of iterations. The default is 1.

    Returns
    -------
    median : float
        Median error value.

    '''

    error_list = []

    for x in range(0, n):
        # get median error using edit distance (predict whole sequence, then compare)
        if error_function == 'editdist':
        	# get predicted sequence for list of objects
            prediction = ''.join(predict_editdist(distances_dict, ID, objects, coordinates, 
                                          start_coordinates, sequence, c, k, dimension))

            # calculate normalized error between predicted and given sequence
            dl = 1 - damerauLevenshtein(sequence, prediction)

            error_list.append(dl)
        
        # get median summed error using prequential method (predict only for each next step)
        elif error_function == 'prequential':
            errors = predict_prequential(distances_dict, ID, objects, coordinates,
                                         start_coordinates, sequence, c, k, dimension)
            summed = sum(errors)
            error_list.append(summed)
                        
    median = np.nanmedian(error_list)
    return median

