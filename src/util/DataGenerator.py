import pandas as pd
import numpy as np
import logging.config
import datetime
import os
import matplotlib.pyplot as plt
import src.util.FileProvider as FP
from src.datamodel.Column import DataDictionary as dd

logger = logging.getLogger(__name__)


def generate_data(treated_count: int, untreated_count: int, num_params: int, cat_params: int, num_categories: int)  -> pd.DataFrame:
    """
    Function generates a data set based on input parameters
    :param treated_count: amount of treated patients
    :param untreated_count: amount of untreated patients
    :param num_params: number of numerical parameters (ie: age)
    :param cat_params: number of categorical parameters (ie: race)
    :param num_categories: number of categories to base the categorical parameters on
    :return: pd.DataFrame: returns the data frame created by the method
    """
    logger.debug(f"DataGenerator. treated_count {treated_count}, untreated_count: {untreated_count}, num_params: {num_params}, "
                 f"cat_params {cat_params}, num_categories: {num_categories}")
    total_patients = treated_count + untreated_count
    # Generate random numerical parameters ranging from 0 to 1 million
    numerical_data = np.random.randint(0, 1000000, size=(total_patients, num_params))
    #numerical_data = np.linspace(0, 1000000, num=numerical_data.size).reshape(numerical_data.shape).astype(int)
    
    # Generate random categorical parameters with 5 categories
    categorical_data = np.random.randint(0, num_categories, size=(total_patients, cat_params))
    #categorical_data = np.linspace(0, 5, num=categorical_data.size).reshape(categorical_data.shape).astype(int)

    # Create columns for numerical and categorical parameters
    num_columns = [f'num_param_{i+1}' for i in range(num_params)]
    cat_columns = [f'cat_param_{i+1}' for i in range(cat_params)]
    
    # Add patient IDs
    patient_ids = [f'Patient_{i+1}' for i in range(total_patients)]
    
    # Create a DataFrame
    data = np.concatenate((np.array(patient_ids).reshape(-1, 1), numerical_data, categorical_data), axis=1)
    columns = [dd.patientID] + num_columns + cat_columns
    df = pd.DataFrame(data, columns=columns)
    
    # Add a column indicating treatment status
    treatment_assignment = (np.random.rand(total_patients) > 0.8).astype(int)
    df[dd.treatment] = treatment_assignment
    
    return df

def generate_study_cases():
    cases =[(1, 0), (0, 1), (1, 1), (5, 0), (0, 5), (5, 5), (50, 0), (0, 50), (50, 50)]
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
def build_plot(data: pd.DataFrame, combined_column_names: list, target, case):
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
    plt.title(f'Propensity Score Distribution (linspace){xLabel}')
    # Specify the directory where you want to save the plot

    # Generate a file name with the timestamp
    file_name = f'match_numerical_{x}_categorical_{y}_{timestamp}.png'

    # Save the plot to the generated file name
    plt.savefig(folder_name + file_name)
    return
def save_dataset(matched_df: pd.DataFrame, case):
    #create a file for each sample combination
    folder_name = FP.build_path
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    x, y = case
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f'match_numerical_{x}_categorical_{y}_{timestamp}.csv'
    file_path = os.path.join(folder_name, file_name)
    try:
        matched_df.to_csv(file_path, index=False)
    except Exception as e:
        logger.error(f"An error occurred while writing to the file: {e}")

    return