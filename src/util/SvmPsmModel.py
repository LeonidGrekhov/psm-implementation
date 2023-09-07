import pandas as pd
import numpy as np
import logging.config
import sys
from src.datamodel.Column import DataDictionary as dd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime

logger = logging.getLogger(__name__)
#def LogRegress(data: pd.DataFrame, num_params: int, num_params_samples: int, cat_params_samples: int) -> pd.DataFrame:
def psm_scorer(treatment_labels, propensity_scores, response_variable):
    # Calculate the weighted mean difference in response_variable
    weighted_diff = np.sum((treatment_labels * propensity_scores - (1 - treatment_labels) * (1 - propensity_scores)) * response_variable) / np.sum(treatment_labels * propensity_scores + (1 - treatment_labels) * (1 - propensity_scores))

    return weighted_diff
def svm_for_psm(data: pd.DataFrame, parameters: list, target: list) -> pd.DataFrame:
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
    logger.debug(f'X scores: {X}')
    logger.debug(f'Y scores: {y}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    if not X.empty:
        # Train an SVM model
        svm_model = svm.SVC(probability=True, kernel='rbf') #poly, rbf, sigmoid
        svm_model.fit(X_train, y_train)

        # Predict propensity scores
        propensity_scores = svm_model.predict_proba(X_test)[:, 1]  # Probability of being treated
        psm = svm_model.predict_proba(X)[:, 1]
        
        logger.debug(f'number of propensity_scores: {len(propensity_scores)}')
        logger.debug("PSM propensity_scores:", propensity_scores)
        # Calculate the PSM score (absolute difference in propensity scores)
        
        y_pred_test = (propensity_scores > 0.19).astype(int)
        logger.debug("PSM y_pred_test:", y_pred_test)
        # Calculate metrics
        metrics_dict = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Testing': [accuracy_score(y_test, y_pred_test), precision_score(y_test, y_pred_test),
                        recall_score(y_test, y_pred_test), f1_score(y_test, y_pred_test)]
        }
        # Create a DataFrame
        metrics_df = pd.DataFrame(metrics_dict)
        # Log the DataFrame
        logger.debug(metrics_df)
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

    return new_df, metrics_df