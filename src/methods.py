import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

def nnm2(medical_data, replacement, caliper, k):
    pairs = []
    treatment_group = medical_data[medical_data['Treatment'] == 1]
    control_group = medical_data[medical_data['Treatment'] == 0]
    count = 1

    if replacement == 1:
        print("Nearest Neighbor with replament")

    if caliper != 0:
        print("Nearest Neighbor with caliper")

    for _, treated_unit in treatment_group.iterrows():
        print(count)
        count+=1
        nn_model = NearestNeighbors(n_neighbors=k)
        nn_model.fit(control_group[['Propensity Score']])
        distances, indices = nn_model.kneighbors([[treated_unit['Propensity Score']]])  # Reshape to 2D array
        if caliper != 0:
            matched_control_data = control_group.iloc[indices[0]]
            propensity_diff = np.abs(matched_control_data['Propensity Score'] - treated_unit['Propensity Score'])
            nearest_neighbors = matched_control_data[propensity_diff <= caliper]

        else:
            nearest_neighbors = control_group.iloc[indices[0]]

        if replacement == 1:
            control_group = control_group[~control_group.isin(nearest_neighbors)].dropna()

        pairs.append((treated_unit, nearest_neighbors))

    return pairs
