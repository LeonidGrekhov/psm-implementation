import pandas as pd
import numpy as np
import logging.config
import sys
from src.datamodel.Column import DataDictionary as dd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler


logger = logging.getLogger(__name__)

def random_for_psm(data: pd.DataFrame, parameters: list, target: list) -> pd.DataFrame:
    """
    Function preforms on provided data frame and returns psm scores for said data frame
    :param data: data frame generated earlier by DataGenerator.py
    :param parameters: number of  parameters (ie: age)
    :param target: target
    :return: pd.DataFrame: returns psm scores added to the data frame
    """
    logger.debug(f"Random forest.  data frame size {data.shape}, parameters: {len(parameters)}, target: {target}")
    
    X = data[parameters]
    y = data[target]
    y = y.values.ravel()
    if not X.empty:
        #logger.debug(f'X scores: {X}')
        #logger.debug(f'Y scores: {y}')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = MinMaxScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        rf_model = RandomForestRegressor()

        # Define a grid of hyperparameters to search over
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Create a GridSearchCV object with cross-validation
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

        # Perform the grid search on the training data
        grid_search.fit(X_train, y_train)

        # Print the best hyperparameters found by the grid search
        best_params = grid_search.best_params_
        logger.debug("Best Hyperparameters:", best_params)

        # Get the best model
        best_rf_model = grid_search.best_estimator_

        # Evaluate the best model on the test set
        y_pred = best_rf_model.predict(X_test)

        # Calculate Mean Squared Error (MSE) as an evaluation metric
        mse = mean_squared_error(y_test, y_pred)
        logger.debug("Mean Squared Error:", mse)

        # You can also access other information such as feature importances if needed
        feature_importances = best_rf_model.feature_importances_
        logger.debug("feature_importances:", feature_importances)

        # Predict propensity scores
        propensity_scores = best_rf_model.predict(X_test)  # Probability of being treated
        psm = best_rf_model.predict(X)
        
        logger.debug(f'number of propensity_scores: {len(propensity_scores)}')
        logger.debug("PSM propensity_scores:", propensity_scores)
        # Calculate the PSM score (absolute difference in propensity scores)
        
        y_pred_test = (propensity_scores > 0.5).astype(int)
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