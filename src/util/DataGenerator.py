import pandas as pd
import numpy as np
import logging.config
import datetime
import os
import matplotlib.pyplot as plt
import src.util.FileProvider as FP
from src.datamodel.Column import DataDictionary as dd

logger = logging.getLogger(__name__)

def generate_data(n_records: int, treatment_rate: float, n_params: int, numeric_params_rate: float, max_categories_n: int, ordered_cat_rate: float)  -> pd.DataFrame:
    """
    Function generates a data set based on input parameters
    :param n_records: int # number of all records
    :param treatment_rate: float # [0,1] percentage of treated patients (i.e. 0.2 means 20% of all records will be treated)
    :param n_params: int # number of columns
    :param numeric_params_rate: float # [0,1] percentage of numerical columns
    :param max_categories_n: int # maximal number of categories each categorical parameter may have
    :param ordered_cat_rate: float
    """
    logger.debug(f"DataGenerator. n_records {n_records}, treatment_rate: {treatment_rate}, n_params: {n_params}, "
                 f"numeric_params_rate {numeric_params_rate}, max_categories_n: {max_categories_n}, ordered_cat_rate: {ordered_cat_rate}")
    num_params, cat_params = round(n_params * numeric_params_rate), round(n_params * (1 - numeric_params_rate))
    
    # Calculate the number of ordered categorical and one-hot encoded categorical parameters
    ordered_cat_count = round(cat_params * ordered_cat_rate)
    one_hot_cat_count = cat_params - ordered_cat_count
    
    # Generate random numerical parameters ranging from 0 to 1 million
    numerical_data = np.random.randint(0, 1000000, size=(n_records, num_params))
    
    # Generate ordered categorical data for ordered_cat_count columns
    ordered_categorical_data = np.random.randint(1, max_categories_n + 1, size=(n_records, ordered_cat_count))
    
    # Generate random categorical data for one_hot_cat_count columns
    one_hot_categorical_data = np.random.randint(0, max_categories_n, size=(n_records, one_hot_cat_count))
    
    # Create column names for numerical, ordered categorical, and one-hot categorical parameters
    num_columns = [f'num_param_{i+1}' for i in range(num_params)]
    ordered_cat_columns = [f'ordered_cat_param_{i+1}' for i in range(ordered_cat_count)]
    one_hot_cat_columns = [f'one_hot_cat_param_{i+1}' for i in range(one_hot_cat_count)]
    
    # Add patient IDs
    patient_ids = [f'Patient_{i+1}' for i in range(n_records)]
    
    # Combine ordered categorical and one-hot categorical data with patient IDs and numerical data
    cat_df = pd.DataFrame(ordered_categorical_data, columns=ordered_cat_columns)
    
    # Perform one-hot encoding on the one-hot categorical data
    one_hot_df = pd.get_dummies(pd.DataFrame(one_hot_categorical_data, columns=one_hot_cat_columns), columns=one_hot_cat_columns)
    
    # Convert True/False values to 1/0
    one_hot_df = one_hot_df.astype(int)
    logger.debug(f'one hot encoding \n{one_hot_df}')
    # Combine all data
    data = np.concatenate((np.array(patient_ids).reshape(-1, 1), numerical_data, cat_df, one_hot_df), axis=1)
    columns = [dd.patientID] + num_columns + ordered_cat_columns + one_hot_df.columns.tolist()
    
    # Create the DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Add a column indicating treatment status
    treatment_assignment = (np.random.rand(n_records) < treatment_rate).astype(int)
    df[dd.treatment] = treatment_assignment
    
    return df    

def generate_study_cases():
    cases =[(1, 0), (0, 1), (1, 1), (5, 0), (0, 5), (5, 5), (50, 0), (0, 50), (50, 70)]
    return cases
def filter_data(df, case, num_params):
    combined_column_names = []
    # Combine both sets of column names
    x, y = case 
    if x >  0:
        selected_numerical_column_names = df.columns[1:x+1]
        combined_column_names += selected_numerical_column_names.tolist()
    if y > 0:
        selected_categorical_column_names = df.columns[num_params+1:y+51]
        combined_column_names += selected_categorical_column_names.tolist()
    return combined_column_names
def build_plot(data: pd.DataFrame, combined_column_names: list, target, case, name):
    x, y = case
    folder_name = FP.build_path
    control_group = data[data[dd.treatment] == 0]
    treatment_group = data[data[dd.treatment] == 1]
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
    plt.title(f'Propensity Score Distribution {xLabel}')
    # Specify the directory where you want to save the plot

    # Generate a file name with the timestamp
    file_name = f'{name}_numerical_{x}_categorical_{y}_{timestamp}.png'

    # Save the plot to the generated file name
    plt.savefig(folder_name + file_name)
    return
def save_dataset(matched_df: pd.DataFrame, case, name):
    #create a file for each sample combination
    folder_name = FP.build_path
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    x, y = case
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f'{name}_numerical_{x}_categorical_{y}_{timestamp}.csv'
    file_path = os.path.join(folder_name, file_name)
    try:
        matched_df.to_csv(file_path, index=False)
    except Exception as e:
        logger.error(f"An error occurred while writing to the file: {e}")

    return