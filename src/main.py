import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import methods
from scipy.spatial.distance import cdist

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
    # Generate synthetic medical dataset with additional columns and binary outcome
    # Generate the dataset
    total_patients = 100
    treated_patients = 20
    medical_data = generate_dataset(total_patients, treated_patients)
    """   
    treatment_group = medical_data[medical_data['Treatment'] == 1]
    control_group = medical_data[medical_data['Treatment'] == 0]
    print(len(treatment_group))
    print(medical_data)
    # Match treatment units with control units using nearest neighbors
    k = 5  # Number of nearest neighbors to match
    matched_pairs = []
    count = 1
    for _, treated_unit in treatment_group.iterrows():
        print(count)
        count+=1
        
        nearest_neighbors = methods.propensity_score_matching_caliper(treated_unit, control_group, k)
        #print(f"the control group before drop:{control_group}")
        #print(nearest_neighbors)
        control_group = control_group[~control_group.isin(nearest_neighbors)].dropna()
        
       
        
        #print(f"the control group after drop:{control_group}")
        matched_pairs.append((treated_unit, nearest_neighbors))
        #print(matched_pairs)
    #medical_data2d = pd.concat([pd.concat([treated, control], axis=1) for treated, control in matched_pairs])
    
    """
    matched_pairs = methods.nnm2(medical_data, replacement = 1, caliper = 0.02, k = 5)
    print(len(matched_pairs))
    count=1
    for row in matched_pairs:
        print(f"{count}\n")
        count+=1
        print(f"Treated Patient:\n{row[0]},\nMatched Patient(s):\n{row[1]}\n")