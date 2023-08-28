import pandas as pd
import numpy as np
import logging.config
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

