import logging
import logging.config
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.util import FileProvider as FP
from src.datamodel.Column import DataDictionary as dd
import src.util.randomForestPSMModel as randomForestPSMModel
import src.util.methods as methods
import src.util.LogReg as LogReg
import src.util.neuralNetwork as neuralNetwork
import src.util.SvmPsmModel as SvmPsmModel
import src.util.DataGenerator as DataGenerator

# Set a random seed for reproducibility
np.random.seed(42)

def main():
    logger = logging.getLogger('Main')
    logger.debug('======Start======')
    model_name = ['LogisticRegression', 'NeuralNetwork', 'RandomForest', 'SupportVectorMachine'] #'LogisticRegression', 'NeuralNetwork', 'RandomForest', 'SupportVectorMachine'  # Change this to the desired model (LogisticRegression, NeuralNetwork, RandomForest, SupportVectorMachine)
    all_results_dfs = []
    # Set the logging level for Matplotlib to INFO (or higher)
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # Generate synthetic medical dataset with additional columns and binary outcome
    # Generate the dataset
    """
    :param treated_count: amount of treated patients
    :param untreated_count: amount of untreated patients
    :param num_params: number of numerical parameters (ie: age)
    :param cat_params: number of categorical parameters (ie: race)
    :param num_categories: number of categories to base the categorical parameters on
    :return: pd.DataFrame: returns the data frame created by the method
    """
    for model in model_name:
        matched_df = pd.DataFrame()
        #result_df = DataGenerator.generate_data(treated_count = 200, untreated_count = 800, num_params = 50, cat_params = 50, num_categories = 5)
        #result_df = DataGenerator.generate_data(n_records=1000, treatment_rate=0.2, n_params=100, numeric_params_rate=0.5, max_categories_n=5, ordered_cat_rate=0.3)
        result_df = DataGenerator.import_data(path='src/data/psm_sample_data.csv')
        logger.debug(f'pre encode:{result_df.head()}')
        result_df = DataGenerator.encode_import_labels(result_df, label_columns = [dd.sex, dd.race, dd.ethnicity])
        encoded_df, encoded_columns = DataGenerator.encode_import_one_hot(result_df, one_hot_columns = [dd.race, dd.ethnicity])
        
        logger.debug(f'post encode:{result_df.head()}')

        file_path = os.path.join(FP.build_path, f'{model}_result_df.csv')
        #result_df.to_csv(file_path, index=False)
        #cases is a selection mechanism that picks the numerical and categorical columns from the generated data frame
        #(1, 0) represents 1 numerical column and one categorical, the rest follow this schema
        cases = DataGenerator.generate_study_cases()
        target = [dd.treatment]
        for case in cases:
            #build the column labels to be passed to logistic regression for testing purposes
            #combined_column_names = DataGenerator.filter_data(result_df, case, num_params=50)
            combined_column_names = ['sex', 'age','bmi_val'] + encoded_columns
            #calculate psm scores and return a new data frame of just the sample columns with patient id and psm scores

            
            data, metrics_df = select_model(model, encoded_df, combined_column_names, target)
            logger.debug(f'post metrics_df:{metrics_df}')
            #calculate the pairs and save them to file
            matched_df = methods.match_nearest_neighbors(data, replacement=True, caliper=0.5, k_neighbors=1, method='caliper')
            #matched_df = pd.concat([matched_df, metrics_df], ignore_index=True)
            DataGenerator.save_dataset(matched_df, case, model)
            #matched_df = DataGenerator.decode_import_one_hot(matched_df, one_hot_columns, encoded_columns)
            matched_df = matched_df.merge(result_df[['patient_id', 'race', 'ethnicity']],
                                on='patient_id',
                                how='left')

            #matched_df.to_csv(file_path, index=False)
            #plot the data
            result_diff, metrics_df = DataGenerator.stats(model, matched_df)
            all_results_dfs.append(metrics_df)
            DataGenerator.metrics(model, result_diff)
            DataGenerator.build_plot(data, combined_column_names, target, case, model)
        merged_results_df = pd.concat(all_results_dfs, ignore_index=True)
        DataGenerator.merged_df_plot(merged_results_df)
        file_name = f'merged_results_df.csv'
        merged_results_df.to_csv(FP.build_path + file_name, index=False)
        matched_df.to_csv(file_path, index=False)   
        logger.debug('======Finish======')

#select model to use for generating psm score and return modified data and metrics
def select_model(model_name, encoded_df, combined_column_names, target):
    if model_name == "LogisticRegression":
        data, metrics_df = LogReg.LogRegress(encoded_df, combined_column_names, target)
    elif model_name == "NeuralNetwork":
        data, metrics_df = neuralNetwork.nnModel(encoded_df, combined_column_names, target)
    elif model_name == "RandomForest":
        data, metrics_df = randomForestPSMModel.random_for_psm(encoded_df, combined_column_names, target)
    elif model_name == "SupportVectorMachine":
        data, metrics_df = SvmPsmModel.svm_for_psm(encoded_df, combined_column_names, target, constant=1.0, kernelMethod='rbf')
    else:
        raise ValueError("Invalid model name")

    return data, metrics_df