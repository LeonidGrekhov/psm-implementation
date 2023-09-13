import pandas as pd
import numpy as np
import logging.config
import sys
from src.datamodel.Column import DataDictionary as dd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import datetime

logger = logging.getLogger(__name__)

def svm_for_psm(data: pd.DataFrame, parameters: list, target: list, constant: float, kernelMethod: str) -> pd.DataFrame:
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
    y = y.values.ravel()
    #logger.debug(f'X scores: {X}')
    #logger.debug(f'Y scores: {y}')
    param_grid = {
        'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1],
        'kernel': ['poly', 'rbf', 'sigmoid'],
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    if not X.empty:
        # Train an SVM model
        svm_model = svm.SVC(C=constant, random_state=1, probability=True, kernel=kernelMethod) #poly, rbf, sigmoid
        svm_model2 = svm.SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm_model2, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Get best hyperparameters
        best_params = grid_search.best_params_

        # Train the best SVM model on the entire training dataset
        best_svm = svm.SVC(**best_params, probability=True, random_state=42)
        best_svm.fit(X_train, y_train)
        svm_model.fit(X_train, y_train)

        # Predict propensity scores
        propensity_scores = best_svm.predict_proba(X_test)[:, 1]  # Probability of being treated
        psm = best_svm.predict_proba(X)[:, 1]
        
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
    logger.debug(f'best param: {best_params}')
    return new_df, metrics_df