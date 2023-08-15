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
    Accepts dataframe with parameters of replacement, caliper and number of neighbors
    returns a list of pairs
    """
    logger = logging.getLogger('NNM2')
    pairs = []
    logger.debug(f'print medical data: {medical_data}')
    treatment_group = medical_data[medical_data[dd.treatment] == 1]
    control_group = medical_data[medical_data[dd.treatment] == 0]
    count = 0
    logger.debug(f'length of control group before execution  {len(control_group)}')
    if replacement == 1:
        print("Nearest Neighbor with replacement")

    if caliper != 0:
        print("Nearest Neighbor with caliper")
    
    for _, treated_unit in treatment_group.iterrows():
        propensity_diff = np.abs(control_group[dd.propensity_scores] - treated_unit[dd.propensity_scores])
        idx = propensity_diff.idxmin()
        count += 1
        diff = propensity_diff[idx]
        logger.debug(f'propensity_diff: {propensity_diff[idx]}')
        match = 0
        if caliper != 0:
            matched_control_data = control_group.loc[idx]
            if diff <= caliper:
                nearest_neighbors = matched_control_data
                match = 1  
            else:
                logger.debug("diff greater than caliper")
                match = 0
        else:
            match = 1
            nearest_neighbors = control_group.loc[idx]
        if replacement == 1 and match == 1:
            control_group = control_group.drop(control_group.loc[idx][0])   
        if match:
            pairs.append((treated_unit, nearest_neighbors))
    logger.debug(f'length of control group after execution  {len(control_group)}')   
    return pairs
