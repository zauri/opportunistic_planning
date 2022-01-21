import ast
import numpy as np
import pandas as pd
from opportunistic_planning.prediction import get_median_error, filter_for_dimension


def calculate_prediction_error(data, distances_dict, error_function, n=10, 
                             dimensions=[[2, 'xy'], [3, 'xyz']], 
                             seqcol='sequence', coords='coordinates', error='error'):
    '''
    Calculates prediction error for all combinations of parameter values (c, k, dimension).

    Parameters
    ----------
    data : pandas.DataFrame
        Generated with read_data function from csv, contains information on objects and sequence.
    
    distances_dict : dictionary
        Contains distances between all objects in all possible dimension combinations.
    
    error_function : function
        Error function to use for prediction error.
        Options: editdist (Damerau-Levenshtein distance), prequential (prequential method)
    
    dimensions : list.
        Dimensions to use. The default is [[2, 'xy'], [3, 'xyz']].
    
    n : int
        Number of iterations for prediction. The default is 10.
    
    seqcol : str, optional
        Column of dataframe containing sequence. The default is 'sequence'.
    
    coords : str, optional
        Column of dataframe containing coordinates. The default is 'coordinates'.
    
    error : str, optional
        Column of dataframe containing error for random samping of sequence
        (only relevant when using editdist prediction). The default is 'error'.

    Returns
    -------
    results : pandas.DataFrame
        Median error over all iterations. Column names: parameter values.

    '''

    results = pd.DataFrame()
    
    for row in range(0, len(data)):
        # get episode information from input row
        objects = list(data.at[row, seqcol])
        coordinates = {key: ast.literal_eval(value) for key, value in
                       (elem.split(': ') for elem in data.at[row, coords].split(';'))}

        start_coordinates = list(ast.literal_eval(data.at[row, 'start_coordinates']))
        ID = str(data.at[row,'ID'])
        seq = str(data.at[row, seqcol])

        # get list of objects that have relational dependencies, if any (else set to empty list)
        try:
            strong_k = list(data.at[row, 'strong_k'].split(','))
        except AttributeError:
            strong_k = []

        try:
            mid_k = list(data.at[row, 'mid_k'].split(','))
        except AttributeError:
            mid_k = []

        try:
            food_k = list(data.at[row, 'food_k'].split(','))
        except AttributeError:
            food_k = []

        

        # go through parameter ranges
        # set k to current param if object has relational dependencies, else 1.0
        for k2 in np.arange(1.1, 2.0, 0.1):
            k_food = round(k2, 2)
            k1 = {obj: k_food if obj in food_k else 1.0 for obj in objects}

            for k in np.arange(0, 0.9, 0.1):
                k_strong = round(k, 2)
                k_mid = round(k + 0.1, 2)
                k1 = {obj: k_strong if obj in strong_k else k_mid if obj in mid_k else round(k1[obj], 2) for obj in
                      objects}

                for c in np.arange(1.0, 2.0, 0.1):
                    c = round(c, 1)
                    # set c to current param if object contained, else 1.0
                    c1 = {obj: c if obj in data.at[row, 'containment'] else 1.0 for obj in objects}

                    for dim in dimensions:
                        # get median error for parameter combination based on error function
                        median = get_median_error(error_function, row, ID, objects, coordinates, start_coordinates, 
                                                  c1, k1, dim,
                                                  seq, distances_dict, n)

                        # save parameter combination as column name in results
                        params = 'c: ' + str(c) + '; k: ' + str(k_strong) + ',' + str(k_mid) + ',' + str(
                            k_food) + '; ' + str(dim[1])

                        results.at[row, params] = median

        results.at[row, 'sequence'] = seq
        results.at[row, 'error'] = data.at[row, error]
        results.at[row, 'ID'] = ID

    return results


def get_lowest_error(results):
    '''
    Return lowest error in dataframe, index of lowest error, lowest median,
    and dataframe with mean/median.

    Parameters
    ----------
    results : pandas.DataFrame
        Resuts dataframe generated with calculate_prediction_error.

    Returns
    -------
    lowest_mean : float
        Lowest mean error.
    lowest_median : float
        Lowest median error.
    lowest_mean_idx : col index
        Column index where mean error is lowest.
    results : pandas.DataFrame
        Results dataframe with mean/median for each parameter combination calculated.

    '''

    for col in list(results):
        if col != 'sequence' and col != 'error' and col != 'ID':
            results.loc['mean', col] = results[col].mean()
            results.loc['median', col] = results[col].median()
    lowest_mean = min(results.loc['mean'])
    lowest_median = min(results.loc['median'])
    #mean = list(results.loc['mean'])
    lowest_mean_idx = results.columns[(results.loc['mean'] == lowest_mean)]

    return lowest_mean, lowest_mean_idx, lowest_median, results


def generate_distances_dict(data, dimensions=[[1, 'x'], [1, 'y'], [1, 'z'], [2, 'xy'], [2, 'xz'], [2, 'yz'], [3, 'xyz']]):
    '''
    Calculate all object distances in all dimensions (e.g., xy, xyz) to reduce computational effort
    in main optimization function (calculate_prediction_error).
    
    Parameters
    ----------
    data : dataframe with object information
    dimensions : list of dimensions to be considered, optional
                The default is [[1, 'x'], [1, 'y'], [1, 'z'], [2, 'xy'], [2, 'xz'], [2, 'yz'], [3, 'xyz']].

    Returns
    -------
    distances_dict : dictionary of all object distances for all dimensions

    '''
    distances_dict = {}
    
    for dim in dimensions:
        dimension = dim[1]
        distances_dict[dimension] = {}
    
        for row in range(0,len(data)):
            objects = list(data.at[row,'sequence'])
            ID = str(data.at[row,'ID'])
            start_coordinates = list(ast.literal_eval(data.at[row,'start_coordinates']))
            coordinates = {key: ast.literal_eval(value) for key, value in
                       (elem.split(': ') for elem in data.at[row,'coordinates'].split(';'))}
    
            distances_dict[dimension][ID] = {}
            
            new_coords, new_start_coords = filter_for_dimension(dim, coordinates, start_coordinates)
    
            for pos in new_start_coords:
                try:
                    position = tuple(pos)
                except TypeError:
                    position = str(pos)
                
                distances_dict[dimension][ID][position] = {}
                
                for obj in objects:
                    if obj not in distances_dict[dimension][ID][position]:
                        distances_dict[dimension][ID][position][obj] = np.linalg.norm(np.array(pos) -
                                                                     np.array(new_coords[obj]))
                
    return distances_dict


def read_data(file):
    '''
    Read in csv file with sequence + object information.
    
    Parameters
    ----------
    file : csv with sequence + object information

    Raises
    ------
    Exception if input data inconsistent (i.e., length of sequence != length of start_coordinate list,
                                          element in sequence not in coordinates dictionary)

    Returns
    -------
    df : dataframe with sequence + object information

    '''
    df = pd.read_csv(file, header=0)
    
    for row in range(0,len(df)):
        start_coordinates = list(ast.literal_eval(df.at[row, 'start_coordinates']))
        ID = str(df.at[row,'ID'])
        sequence = str(df.at[row, 'sequence'])
        coordinates = {key: ast.literal_eval(value) for key, value in
                       (elem.split(': ') for elem in df.at[row,'coordinates'].split(';'))}
        
        # check if nr. of items matches with nr. of start positions
        if len(sequence) != len(start_coordinates):
            raise Exception('Sequence length !=  nr. of start positions for ID {}'.format(ID))
        
        # check if coordinates for all items are given
        for elem in sequence:
            if elem not in coordinates.keys():
                raise Exception('No coordinates for object {}'.format(elem))
    
    return df

def read_results(file):
    '''
    Read in previously saved results from main calculate_prediction_error function.
    
    Parameters
    ----------
    file : csv

    Returns
    -------
    results : results as pandas dataframe

    '''
    results = pd.read_csv(file, header=0)
    results = results.T
    results.reset_index(drop=True, inplace=True)

    header = results.iloc[0]
    results = results[1:]
    results.columns = header

    results.drop(results.tail(1).index, inplace=True)
    
    # convert strings to numeric if possible
    results = results.apply(pd.to_numeric, errors='ignore')

    return results


def save_results(file, filepath):
    '''
    Save results dataframe to csv.

    Parameters
    ----------
    file : pandas dataframe
        Results dataframe generated by calculate_prediction_error.
    filepath : str
        Save path for file.

    Returns
    -------
    None.

    '''
    file.T.to_csv(filepath, header=True, index=True)


def check_to_float(x):
    try:
        return float(x)
    except ValueError:
        return x