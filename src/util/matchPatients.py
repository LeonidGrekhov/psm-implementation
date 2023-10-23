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

def matchPatients(path:str, label_columns:list, one_hot_columns:list, target_column:list) -> pd.DataFrame:
    logger = logging.getLogger('MatchPatients')

    model_name = ['NN','LR', 'RF', 'SVM'] #'LR', 'NN', 'RF', 'SVM''LogisticRegression', 'NeuralNetwork', 'RandomForest', 'SupportVectorMachine'  # Change this to the desired model (LogisticRegression, NeuralNetwork, RandomForest, SupportVectorMachine)
    bmi_diff = []
    age_diff = []
    sex_diff = []
    race_diff = []
    eth_diff = []
    target = target_column
    # Set the logging level for Matplotlib to INFO (or higher)
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
   
    diffs = pd.DataFrame()

    for model in model_name:
        matched_df = pd.DataFrame()
        original_df = DataGenerator.import_data(path)
        lb_encode = DataGenerator.encode_import_labels(original_df, label_columns)
        logger.debug(f'lb_encode:\n{lb_encode}')

        encoded_df, encoded_columns = DataGenerator.encode_import_one_hot(lb_encode, one_hot_columns)
        combined_column_names = ['sex', 'age','bmi_val'] + encoded_columns
        #calculate psm scores and return a new data frame of just the sample columns with patient id and psm scores

        data, metrics_df, ps_results = select_model(model, original_df, encoded_df, combined_column_names, target)
        ps_results.to_csv()
        logger.debug(f'post data:\n{data}')

        logger.debug(f'post metrics_df:\n{metrics_df}')

        matched_df = methods.match_nearest_neighbors(ps_results, replacement=True, caliper=0.5, k_neighbors=1, method='caliper')

        DataGenerator.save_dataset(matched_df, model)

        age_diff.append(DataGenerator.find_diff(matched_df, dd.age, model))
        bmi_diff.append(DataGenerator.find_diff(matched_df, dd.bmi, model))
        sex_diff.append(DataGenerator.find_cat(matched_df, dd.sex, model))
        race_diff.append(DataGenerator.find_cat(matched_df, dd.race, model))
        eth_diff.append(DataGenerator.find_cat(matched_df, dd.ethnicity, model))
        
    sex_diff = pd.concat(sex_diff)
    race_diff = pd.concat(race_diff)
    eth_diff = pd.concat(eth_diff)
    age_diff = pd.concat(age_diff)
    bmi_diff = pd.concat(bmi_diff)

    DataGenerator.plot_boxplot(age_diff)      
    DataGenerator.plot_boxplot(bmi_diff)

    DataGenerator.plot_barplot(sex_diff)
    DataGenerator.plot_barplot(race_diff) 
    DataGenerator.plot_barplot(eth_diff)    

#select model to use for generating psm score and return modified data and metrics
def select_model(model_name, original_df, encoded_df, combined_column_names, target):
    if model_name == "LR": #LogisticRegression
        data, metrics_df, psm_results = LogReg.LogRegress(original_df, encoded_df, combined_column_names, target)
    elif model_name == "NN": #NeuralNetwork
        data, metrics_df, psm_results = neuralNetwork.nnModel(original_df, encoded_df, combined_column_names, target)
    elif model_name == "RF": #RandomForest
        data, metrics_df, psm_results = randomForestPSMModel.random_for_psm(original_df, encoded_df, combined_column_names, target)
    elif model_name == "SVM": #SupportVectorMachine
        data, metrics_df, psm_results = SvmPsmModel.svm_for_psm(original_df, encoded_df, combined_column_names, target, constant=1.0, kernelMethod='rbf')
    else:
        raise ValueError("Invalid model name")

    return data, metrics_df, psm_results