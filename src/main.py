import datetime
import json
import logging
import logging.config
import os

import src.util.FileProvider as FP
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import src.util.methods as methods
import src.util.LogisticRegression as LogisticRegression
import src.util.DataGenerator as DataGenerator
from scipy.spatial.distance import cdist


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
    total_patients = 800
    treated_patients = 200
    num_params = 50
    cat_params = 50
    num_categories = 5
    num_params_samples = 50
    cat_params_samples = 50
    
    result_df = DataGenerator.generate_data(total_patients, treated_patients, num_params, cat_params, num_categories)
    print(result_df) 
    medical_data = LogisticRegression.LogRegress(result_df, num_params, num_params_samples, cat_params_samples)

    
    #medical_data = generate_dataset(total_patients, treated_patients)

    matched_pairs = methods.nnm2(medical_data, replacement=True, caliper=0.02, k_neighbors=1, method='caliper')
    
    folder_name = "build"
    file_name = "matched_pairs.txt"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    file_path = os.path.join(folder_name, file_name)
    logger.debug(f'pairs: \n{matched_pairs}')
    with open(file_path, "w") as f:      
        
        f.write(f"\nMatched Patient:\n{matched_pairs[::2]}\n")
        f.write(f"\nTreated Patient(s):\n{matched_pairs[1::2]}\n")
        f.write(f"Total matched pairs: {len(matched_pairs)}\n")
    
    
    logger.debug('======Finish======')

