import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

def nnm(treated_unit, control_group, k):
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(control_group[['Propensity Score']])
    distances, indices = nn_model.kneighbors([[treated_unit['Propensity Score']]])  # Reshape to 2D array
    return control_group.iloc[indices[0]]


def nnmNoReplacement(treated_unit, control_group, k):
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(control_group[['Propensity Score']])
    distances, indices = nn_model.kneighbors([[treated_unit['Propensity Score']]])  # Reshape to 2D array
    #control_group = control_group[~control_group.isin(control_group.iloc[indices[0]])].dropna()
    return control_group.iloc[indices[0]] 

def propensity_score_matching_caliper(treated_unit, control_group, k, caliper=0.02):
    # Find k-nearest neighbors in the control group for each treated individual
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(control_group[['Propensity Score']])
    if treated_unit['Propensity Score'] is not None:
        distances, indices = nn_model.kneighbors([[treated_unit['Propensity Score']]])
    
    #matched_indices = indices.reshape(-1)
    matched_control_data = control_group.iloc[indices[0]]

    # Apply caliper constraint
    propensity_diff = np.abs(matched_control_data['Propensity Score'] - treated_unit['Propensity Score'])
    #matched_data = pd.concat([treated_unit, matched_control_data[propensity_diff <= caliper]], axis=0)

    return matched_control_data[propensity_diff <= caliper]


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
"""
def nearest_neighbor_matching_without_replacement(data, treated_col='Treatment', ps_col='Propensity Score'):
    treated_data = data[data[treated_col] == 1]
    untreated_data = data[data[treated_col] == 0]
    matched_data = []

    while len(treated_data) > 0:
        distances = cdist(treated_data[[ps_col]], untreated_data[[ps_col]], metric='euclidean')
        min_distances = np.min(distances, axis=1)
        min_index = np.argmin(distances, axis=1)

        closest_index = np.argmin(min_distances)
        treated_index = treated_data.index[closest_index]
        untreated_index = untreated_data.index[min_index[closest_index]]

        matched_data.append(pd.concat([treated_data.loc[treated_index], untreated_data.loc[untreated_index]], axis=0))

        treated_data = treated_data.drop(treated_index)
        untreated_data = untreated_data.drop(untreated_index)

    matched_data = pd.concat(matched_data, ignore_index=True)
    return matched_data

def nearest_neighbor_matching_with_replacement(data, treated_col='Treatment', ps_col='Propensity Score'):
    treated_data = data[data[treated_col] == 1]
    untreated_data = data[data[treated_col] == 0]
    matched_data = []

    while len(treated_data) > 0:
        distances = cdist(treated_data[[ps_col]], untreated_data[[ps_col]], metric='euclidean')
        min_distances = np.min(distances, axis=1)
        min_index = np.argmin(distances, axis=1)

        closest_index = np.argmin(min_distances)
        treated_index = treated_data.index[closest_index]
        untreated_index = untreated_data.index[min_index[closest_index]]

        matched_data.append(pd.concat([treated_data.loc[treated_index], untreated_data.loc[untreated_index]], axis=0))

        treated_data = treated_data.drop(treated_index)

    matched_data = pd.concat(matched_data, ignore_index=True)
    return matched_data
"""