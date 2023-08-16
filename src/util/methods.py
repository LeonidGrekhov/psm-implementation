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
logger = logging.getLogger(__name__)
def nnm2(medical_data: pd.DataFrame, replacement: bool, caliper: float, k: int) -> list:
    """
    Accepts dataframe with parameters of replacement, caliper and number of neighbors
    returns a list of pairs (treated patient, untreated patient)
    """
    pairs = []
    treatment_group = medical_data[medical_data[dd.treatment] == 1]
    control_group = medical_data[medical_data[dd.treatment] == 0]
    logger.debug(f'length of control group before execution  {len(control_group)}')
    logger.debug(f"data size {medical_data.shape}, replacement: {replacement}, caliper: {caliper}")
    
    for _, treated_unit in treatment_group.iterrows():
        control_group['DIFF'] = np.abs(control_group[dd.propensity_scores] - treated_unit[dd.propensity_scores])
        idx = control_group['DIFF'].idxmin()
        diff = control_group['DIFF'][idx]
        match = 0
        
        if caliper:  
            if diff <= caliper:
                filtered_control_group = control_group[control_group['DIFF'] <= caliper]
                nearest_neighbors = filtered_control_group.sample(n=1)
                match = 1  
                logger.debug(f'propensity_diff: {diff}') 
                logger.debug(f'filtered_control_group:\n {filtered_control_group}')
                logger.debug(f'nearest_neighbors:\n {nearest_neighbors}')
            else:
                logger.warning("diff greater than caliper")
                match = 0
        else:
            control_group = control_group.sort_values(by="DIFF")
            filtered_k_rows = control_group.head(k)
            nearest_neighbors = filtered_k_rows.sample(n=1)
            match = 1
            logger.debug(f'filtered_k_rows:\n {filtered_k_rows}')
            logger.debug(f'nearest_neighbors:\n {nearest_neighbors}')
        if replacement == True and match == 1:
            control_group = control_group.drop(nearest_neighbors.iloc[0][0])   
        if match:
            pairs.append((treated_unit, nearest_neighbors))
    logger.debug(f'length of control group after execution  {len(control_group)}')   
    return pairs
