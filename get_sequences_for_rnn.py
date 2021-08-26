#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 12:59:53 2021

@author: Petra Wenzl
"""

import argparse
import pandas as pd


def read_data(input_file):
    dataframe = pd.read_csv(input_file, header=0)
    
    return dataframe


def get_sequences(dataframe):
    return [dataframe.at[row, 'sequence'] for row in range(0, len(dataframe))]


def save_to_file(sequence_list):
    with open('test_data_rnn.txt', 'w') as file:
        for sequence in sequence_list:
            file.write(sequence)
            file.write('\n')
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', action='store', help='input csv file \
                        to process')
                        
    parsed_arguments = parser.parse_args()
    
    input_file = parsed_arguments.filename
    
    dataframe = read_data(input_file)
    sequence_list = get_sequences(dataframe)
    save_to_file(sequence_list)