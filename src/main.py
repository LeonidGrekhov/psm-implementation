import datetime
import json
import logging
import logging.config
import os

import util.FileProvider as FP
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import util.methods as methods
from scipy.spatial.distance import cdist

logging.basicConfig(filename='test.log', level=logging.DEBUG)

# Set a random seed for reproducibility
np.random.seed(42)

def generate_dataset(total_patients, treated_patients):
    # Generate random propensity scores between 0 and 1 for all patients
    propensity_scores = np.random.rand(total_patients)

    # Determine the treatment assignment (0 for control, 1 for treatment)
    #treatment_assignment = np.random.choice([0, 1], size=total_patients, replace=True, p=[0.80, 0.20])
    treatment_assignment = (np.random.rand(total_patients) > 0.8).astype(int)
    # Create a DataFrame
    data = pd.DataFrame({
        'Patient ID': range(1, total_patients + 1),
        'Propensity Score': propensity_scores,
        'Treatment': treatment_assignment
    })

    return data




if __name__ == '__main__':
    logging.config.fileConfig(FP.get_log_config_file())
    logger = logging.getLogger('Main')
    logger.debug('======Start======')
    # Generate synthetic medical dataset with additional columns and binary outcome
    # Generate the dataset
    total_patients = 100
    treated_patients = 20
    medical_data = generate_dataset(total_patients, treated_patients)
    logger.debug(medical_data[medical_data['Treatment'] == 1])
    matched_pairs = methods.nnm2(medical_data, replacement = 1, caliper = 0.02, k = 11)
    
    logging.debug('lenth of matched pairs')
    logging.debug(len(matched_pairs))
    count=1
    for row in matched_pairs:
        logging.debug(f"{count}\n")
        count+=1
        logging.debug(f"Treated Patient:\n{row[0]},\nMatched Patient(s):\n{row[1]}\n")
