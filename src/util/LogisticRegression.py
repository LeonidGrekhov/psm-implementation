import pandas as pd
import numpy as np
import logging.config
from src.datamodel.Column import DataDictionary as dd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

def LogRegress(medical_data: pd.DataFrame, num_params: int, num_params_samples: int, cat_params_samples: int) -> pd.DataFrame:
    """
    Function preforms on provided data frame and returns psm scores for said data frame
    :param medical_data: data frame generated earlier by DataGenerator.py
    :param num_params: number of numerical parameters (ie: age)
    :param num_params_samples: number of numerical sample columns we shit to work with
    :param cat_params_samples: number of categorical sample columns we shit to work with
    :return: pd.DataFrame: returns psm scores added to the data frame
    """
    logger.debug(f"LogRegress. medical data frame size {medical_data.shape}, num_params: {num_params}, num_params_samples: {num_params_samples}, "
                 f"cat_params_samples {cat_params_samples}")
    selected_numerical_column_names = medical_data.columns[1:num_params_samples+1]
    logger.debug(f'selected_numerical_column_names: {selected_numerical_column_names}')
    selected_categorical_column_names = medical_data.columns[num_params+1:cat_params_samples+51]
    logger.debug(f'selected_categorical_column_names: {selected_categorical_column_names}')
    # Combine both sets of column names
    combined_column_names = selected_numerical_column_names.tolist() + selected_categorical_column_names.tolist()
    logger.debug(f'lables: {combined_column_names}')
    features = combined_column_names
    target = dd.treatment
    
    X = medical_data[features]
    y = medical_data[target]
    model = LogisticRegression()
    model.fit(X, y) 
    propensity_scores = model.predict_proba(X)[:, 1]
    logger.debug(f'number of propensity_scores: {len(propensity_scores)}')
    medical_data[dd.propensity_scores] = propensity_scores

    return medical_data