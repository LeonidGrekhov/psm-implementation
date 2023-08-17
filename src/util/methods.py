import logging.config
from src.datamodel.Column import DataDictionary as dd
import pandas as pd
import numpy as np
import logging.config

import numpy as np
import pandas as pd

from src.datamodel.Column import DataDictionary as dd

logger = logging.getLogger(__name__)


def nnm2(medical_data: pd.DataFrame, replacement: bool, caliper: float, k_neighbors: int, method: str) -> pd.DataFrame:
    """
    Accepts dataframe with parameters of replacement, caliper and number of neighbors
    returns a list of pairs (treated patient, untreated patient)
    :param medical_data:
    :param replacement:
    :param caliper:
    :param k_neighbors:
    :param method: possible values:"knn", "caliper". define which method for patient matching should be used
    :return:
    """
    logger.debug(f"nnm2. medical data frame size {medical_data.shape}, replacement: {replacement}, caliper: {caliper}, "
                 f"k_neighbors {k_neighbors}")

    all_matched_dfs = []
    treatment_group = medical_data[medical_data[dd.treatment] == 1]
    control_group = medical_data[medical_data[dd.treatment] == 0]
    logger.debug(f'length of control group before execution  {len(control_group)}')

    for _, treated_unit in treatment_group.iterrows():
        # get distance between treated unit and all control records
        control_group['DIFF'] = np.abs(control_group[dd.propensity_scores] - treated_unit[dd.propensity_scores])

        # get matched records
        if method == 'caliper':
            filtered_control_group = control_group[control_group['DIFF'] <= caliper]
            if filtered_control_group.empty:
                logger.warning(f'record {treated_unit[dd.patientID]} does not match any of control group')
                continue
            filtered_control_group = filtered_control_group.sort_values(by="DIFF")
        elif method == 'knn':
            control_group = control_group.sort_values(by="DIFF")
            filtered_control_group = control_group.head(k_neighbors)
        else:
            logger.warning(f"INVALID MATCHING METHOD {method}")
            return None

        # pick records for the result
        matched_records = filtered_control_group.head(1)
        if replacement:
            control_group = control_group[~control_group[dd.patientID].isin(matched_records[dd.patientID])]
        matched_records = pd.concat([matched_records, treated_unit.to_frame().transpose()])
        all_matched_dfs.append(matched_records)

    all_matched_df = pd.concat(all_matched_dfs)
    logger.debug(f'length of control group after execution  {len(control_group)}')
    logger.debug(f'matched data size  {all_matched_df.shape}')
    return all_matched_df
