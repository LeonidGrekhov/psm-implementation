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


def generate_dataset(total_patients, treated_patients):
    # Generate random propensity scores between 0 and 1 for all patients
    propensity_scores = np.random.rand(total_patients)

    # Determine the treatment assignment (0 for control, 1 for treatment)
    # treatment_assignment = np.random.choice([0, 1], size=total_patients, replace=True, p=[0.80, 0.20])
    treatment_assignment = (np.random.rand(total_patients) > 0.8).astype(int)
    # Create a DataFrame
    data = pd.DataFrame({
        'Patient ID': range(total_patients),
        'Propensity Score': propensity_scores,
        'Treatment': treatment_assignment
    })
    return data

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
    treated_patients = 200
    untreated_count = 800
    
    num_params = 50
    cat_params = 50
    num_categories = 5
    
    matched_pairs = pd.DataFrame()
    result_df = DataGenerator.generate_data(treated_patients, untreated_count, num_params, cat_params, num_categories)
    print(result_df) 
    #num_params_samples = 50
    #cat_params_samples = 50
    #medical_data = LogisticRegression.LogRegress(result_df, num_params, num_params_samples, cat_params_samples) 
    #matched_pairs = methods.nnm2(medical_data, replacement=True, caliper=0.02, k_neighbors=1, method='caliper')
    samples = [(1, 0), (0, 1), (1, 1), (5, 0), (0, 5), (5, 5), (50, 0), (0, 50), (50, 50)]
    '''
    
    '''
    parameters = ['num_param_1', 'cat_param_1']
    target = [dd.treatment]
    folder_name = "build"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for treatment in samples:
        #build the column labels to be passed to logistic regression for testing purposes
        combined_column_names = []
        x, y = treatment 
        if x >  0:
            selected_numerical_column_names = result_df.columns[1:x+1]
            combined_column_names += selected_numerical_column_names.tolist()
        if y > 0:
            selected_categorical_column_names = result_df.columns[num_params+1:y+51]
            combined_column_names += selected_categorical_column_names.tolist()
        # Combine both sets of column names
        #create a file for each sample combination
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f'match_numerical_{x}_categorical_{y}_{timestamp}.csv'
        file_path = os.path.join(folder_name, file_name)
        #calculate psm scores and return a new dataframe of just the sample columns with patient id and psm scores
        data = LogReg.LogRegress(result_df, combined_column_names, target)
        #calculate the pairs and save them to file
        matched_pairs = methods.nnm2(data, replacement=True, caliper=0.02, k_neighbors=1, method='caliper')
        
        logger.debug(f'pairs: \n{matched_pairs}')
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Matched Patient:\n")
                f.write(matched_pairs.to_string(index=False) + "\n\n")
                f.write(f"Total matched pairs: {len(matched_pairs)}\n")
                f.flush()
        except Exception as e:
            logger.error(f"An error occurred while writing to the file: {e}")


        plot_enable = 1
        control_group = data[data[dd.treatment] == 0]
        treatment_group = data[data[dd.treatment] == 1]

        # Plot histograms of PSM scores for control and treatment groups
        
        if plot_enable:
            logging.getLogger('matplotlib.font_manager').disabled = True
            logging.getLogger('PIL.PngImagePlugin').disabled = True
            #Overlap Assessment (Propensity Score Distribution)
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            xLabel = f'\nParameters {len(combined_column_names)} target {target}'
            plt.figure()
            plt.hist(control_group[dd.propensity_scores], bins=20, alpha=0.5, color='blue', label='Control Group')
            plt.hist(treatment_group[dd.propensity_scores], bins=20, alpha=0.5, color='orange', label='Treatment Group')    
            plt.xlabel('Propensity Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title(f'Propensity Score Distribution (linspace){xLabel}')
            # Specify the directory where you want to save the plot
            save_directory = "build/"
            # Generate a timestamp
            

            # Generate a file name with the timestamp
            file_name = f'match_numerical_{x}_categorical_{y}_{timestamp}.png'

            # Save the plot to the generated file name
            plt.savefig(save_directory + file_name)
    logger.debug('======Finish======')

