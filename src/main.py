import datetime
import json
import logging
import logging.config
import os
import datetime
import src.util.FileProvider as FP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.datamodel.Column import DataDictionary as dd

import src.util.methods as methods
import src.util.LogReg as LogReg
import src.util.DataGenerator as DataGenerator



# Set a random seed for reproducibility
np.random.seed(42)

def main():
    logger = logging.getLogger('Main')
    logger.debug('======Start======')
    
    # Generate synthetic medical dataset with additional columns and binary outcome
    # Generate the dataset
    """
    :param treated_count: amount of treated patients
    :param untreated_count: amount of untreated patients
    :param num_params: number of numerical parameters (ie: age)
    :param cat_params: number of categorical parameters (ie: race)
    :param num_categories: number of categories to base the categorical parameters on
    :return: pd.DataFrame: returns the data frame created by the method
    """
    matched_df = pd.DataFrame()
    result_df = DataGenerator.generate_data(treated_count = 200, untreated_count = 800, num_params = 50, cat_params = 50, num_categories = 5)
    #cases is a selection mechanism that picks the numerical and categorical columns from the generated data frame
    #(1, 0) represents 1 numerical column and one categorical, the rest follow this schema
    cases = DataGenerator.generate_study_cases()
    target = [dd.treatment]
    for case in cases:
        #build the column labels to be passed to logistic regression for testing purposes
        combined_column_names = DataGenerator.filter_data(result_df, case, num_params=50)
        #calculate psm scores and return a new data frame of just the sample columns with patient id and psm scores
        data = LogReg.LogRegress(result_df, combined_column_names, target)
        #calculate the pairs and save them to file
        matched_df = methods.match_nearest_neighbors(data, replacement=True, caliper=0.02, k_neighbors=1, method='caliper')
        DataGenerator.save_dataset(matched_df, case)
        #plot the data
        DataGenerator.build_plot(data, combined_column_names, target, case)
    logger.debug('======Finish======')

