import datetime
import json
import logging
import logging.config
import os
from src.datamodel.Column import DataDictionary as dd
import src.util.FileProvider as FP
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


def nnm2(medical_data: pd.DataFrame, replacement: int, caliper: float, k: int) -> list:
    """
    Function description
    """
    logger = logging.getLogger('NNM2')
    pairs = []
    treatment_group = medical_data[medical_data[dd.treatment] == 1]
    control_group = medical_data[medical_data[dd.treatment] == 0]
    count = 1

    if replacement == 1:
        print("Nearest Neighbor with replacement")

    if caliper != 0:
        print("Nearest Neighbor with caliper")

    for _, treated_unit in treatment_group.iterrows():
        logger.debug(count)
        count += 1
        logger.debug([treated_unit[dd.propensity_scores]])
        logger.debug(len(control_group))
        nn_model = NearestNeighbors(n_neighbors=k)

        nn_model.fit(control_group[[dd.propensity_scores]])

        distances, indices = nn_model.kneighbors([[treated_unit[dd.propensity_scores]]])  # Reshape to 2D array
        if caliper != 0:
            matched_control_data = control_group.iloc[indices[0]]
            propensity_diff = np.abs(matched_control_data[dd.propensity_scores] - treated_unit[dd.propensity_scores])
            nearest_neighbors = matched_control_data[propensity_diff <= caliper]

        else:
            nearest_neighbors = control_group.iloc[indices[0]]

        if replacement == 1:
            control_group = control_group[~control_group.isin(nearest_neighbors)].dropna()

        pairs.append((treated_unit, nearest_neighbors))

    return pairs
