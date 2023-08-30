import pandas as pd
import numpy as np
import logging.config
import sys
from src.datamodel.Column import DataDictionary as dd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import datetime

logger = logging.getLogger(__name__)
#def LogRegress(data: pd.DataFrame, num_params: int, num_params_samples: int, cat_params_samples: int) -> pd.DataFrame:
def LogRegress(data: pd.DataFrame, parameters: list, target: list) -> pd.DataFrame:
    """
    Function preforms on provided data frame and returns psm scores for said data frame
    :param data: data frame generated earlier by DataGenerator.py
    :param parameters: number of  parameters (ie: age)
    :param target: target
    :return: pd.DataFrame: returns psm scores added to the data frame
    """
    logger.debug(f"LogRegress.  data frame size {data.shape}, parameters: {len(parameters)}, target: {target}")
    
    X = data[parameters]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    if not X.empty:
        model.fit(X_train, y_train) 

        psm_train = model.predict_proba(X_train)[:, 1]
        psm_test = model.predict_proba(X_test)[:, 1]
        psm = model.predict_proba(X)[:, 1]
        logger.debug(f'number of propensity_scores: {len(psm_test)}')
        data[dd.propensity_scores] = psm
        

        y_pred_train = (psm_train > 0.5).astype(int)
        y_pred_test = (psm_test > 0.5).astype(int)

        # Calculate accuracy on the test set
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)

        precision_train = precision_score(y_train, y_pred_train)
        precision_test = precision_score(y_test, y_pred_test)

        recall_train = recall_score(y_train, y_pred_train)
        recall_test = recall_score(y_test, y_pred_test)

        f1_train = f1_score(y_train, y_pred_train)
        f1_test = f1_score(y_test, y_pred_test)
        

        logger.debug(f"Training Accuracy: {accuracy_train}", )
        logger.debug(f"Testing Accuracy: {accuracy_test}", )
        logger.debug(f"Training Precision: {precision_train}", )
        logger.debug(f"Testing Precision: {precision_test}", )
        logger.debug(f"Training Recall: {recall_train}", )
        logger.debug(f"Testing Recall: {recall_test}", )
        logger.debug(f"Training F1 Score: {f1_train}", )
        logger.debug(f"Testing F1 Score: {f1_test}", )
    else:
        logger.warning(f'Must select more than 0 columns to generate a psm score')
        sys.exit()
    selected = []
    parameters = parameters
    target = target
    patientID = data.columns[0]
    selected += [patientID] + parameters + target
    data_dict = {col: data[col] if col != dd.propensity_scores else psm for col in selected}


    # Create a new DataFrame from the dictionary
    new_df = pd.DataFrame(data_dict)
    new_df[dd.propensity_scores] = psm

    return new_df