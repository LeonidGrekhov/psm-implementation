import pandas as pd
import numpy as np
import logging.config
import datetime
import os
import matplotlib.pyplot as plt
import src.util.FileProvider as FP
import seaborn as sns
from src.datamodel.Column import DataDictionary as dd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

def generate_data(n_records: int, treatment_rate: float, n_params: int, numeric_params_rate: float, max_categories_n: int, ordered_cat_rate: float)  -> pd.DataFrame:
    """
    Function generates a data set based on input parameters
    :param n_records: int # number of all records
    :param treatment_rate: float # [0,1] percentage of treated patients (i.e. 0.2 means 20% of all records will be treated)
    :param n_params: int # number of columns
    :param numeric_params_rate: float # [0,1] percentage of numerical columns
    :param max_categories_n: int # maximal number of categories each categorical parameter may have
    :param ordered_cat_rate: float
    """
    logger.debug(f"DataGenerator. n_records {n_records}, treatment_rate: {treatment_rate}, n_params: {n_params}, "
                 f"numeric_params_rate {numeric_params_rate}, max_categories_n: {max_categories_n}, ordered_cat_rate: {ordered_cat_rate}")
    num_params, cat_params = round(n_params * numeric_params_rate), round(n_params * (1 - numeric_params_rate))
    
    # Calculate the number of ordered categorical and one-hot encoded categorical parameters
    ordered_cat_count = round(cat_params * ordered_cat_rate)
    one_hot_cat_count = cat_params - ordered_cat_count
    
    # Generate random numerical parameters ranging from 0 to 1 million
    numerical_data = np.random.randint(0, 1000000, size=(n_records, num_params))
    
    # Generate ordered categorical data for ordered_cat_count columns
    ordered_categorical_data = np.random.randint(1, max_categories_n + 1, size=(n_records, ordered_cat_count))
    
    # Generate random categorical data for one_hot_cat_count columns
    one_hot_categorical_data = np.random.randint(0, max_categories_n, size=(n_records, one_hot_cat_count))
    
    # Create column names for numerical, ordered categorical, and one-hot categorical parameters
    num_columns = [f'num_param_{i+1}' for i in range(num_params)]
    ordered_cat_columns = [f'ordered_cat_param_{i+1}' for i in range(ordered_cat_count)]
    one_hot_cat_columns = [f'one_hot_cat_param_{i+1}' for i in range(one_hot_cat_count)]
    
    # Add patient IDs
    patient_ids = [f'{i+1}' for i in range(n_records)]
    
    # Combine ordered categorical and one-hot categorical data with patient IDs and numerical data
    cat_df = pd.DataFrame(ordered_categorical_data, columns=ordered_cat_columns)
    
    # Perform one-hot encoding on the one-hot categorical data
    one_hot_df = pd.get_dummies(pd.DataFrame(one_hot_categorical_data, columns=one_hot_cat_columns), columns=one_hot_cat_columns)
    
    # Convert True/False values to 1/0
    one_hot_df = one_hot_df.astype(int)
    logger.debug(f'one hot encoding \n{one_hot_df}')
    # Combine all data
    data = np.concatenate((np.array(patient_ids).reshape(-1, 1), numerical_data, cat_df, one_hot_df), axis=1)
    columns = [dd.patientID] + num_columns + ordered_cat_columns + one_hot_df.columns.tolist()
    
    # Create the DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Add a column indicating treatment status
    treatment_assignment = (np.random.rand(n_records) < treatment_rate).astype(int)
    df[dd.treatment] = treatment_assignment
    
    return df    
def import_data(path:str):
    df = pd.read_csv(path, dtype={dd.patientID: str, dd.sex: str, dd.ethnicity: str, dd.age: int, dd.bmi: float, dd.treatment: bool})
    return df
def generate_study_cases():
    cases = [(1, 0)]
    return cases
def filter_data(df, case, num_params):
    combined_column_names = []
    # Combine both sets of column names
    x, y = case 
    if x >  0:
        selected_numerical_column_names = df.columns[1:x+1]
        combined_column_names += selected_numerical_column_names.tolist()
    if y > 0:
        selected_categorical_column_names = df.columns[num_params+1:y+51]
        combined_column_names += selected_categorical_column_names.tolist()
    return combined_column_names

def encode_import_labels(df, label_columns):
    df = df.copy()
    label_encoder = LabelEncoder()
    df[dd.patientID] = df[dd.patientID].str.extract('(\d+)').astype(int)
    
    # Label encode the specified columns
    for column in label_columns:
        df[column] = label_encoder.fit_transform(df[column]).astype(int)
    return df

def encode_import_one_hot(df, one_hot_columns):
    # One-hot encode the specified columns and convert to int
    encoded_df = df.copy()
    encoded_columns = pd.get_dummies(encoded_df[one_hot_columns], columns=one_hot_columns, dtype=int)

    encoded_columns = encoded_columns.astype(int)
    # Concatenate the one-hot encoded columns with the original DataFrame
    encoded_df = pd.concat([encoded_df, encoded_columns], axis=1)
    logger.debug(f'new df: {encoded_df}')
    encoded_df.drop(columns=one_hot_columns, inplace=True)
    return encoded_df, encoded_columns.columns.tolist()

def decode_import_one_hot(encoded_df, one_hot_columns, encoded_columns):
     # Copy the encoded DataFrame to avoid modifying the original
    decoded_df = encoded_df.copy()

    # Create a dictionary to map the original columns to their respective one-hot encoded columns
    column_mapping = {}
    for column in one_hot_columns:
        for encoded_column in encoded_columns:
            if column in encoded_column:
                column_mapping[encoded_column] = column

    # Create a new DataFrame to hold the original columns
    original_columns_df = pd.DataFrame()
    for encoded_column, original_column in column_mapping.items():
        original_columns_df[original_column] = encoded_df[encoded_column]

    # Drop the one-hot encoded columns
    #decoded_df.drop(columns=encoded_columns, inplace=True)

    # Concatenate the original columns
    decoded_df = pd.concat([decoded_df, original_columns_df], axis=1)

    return decoded_df

def build_plot(data: pd.DataFrame, combined_column_names: list, target, case, model_name):
    x, y = case
    folder_name = FP.build_path
    control_group = data[data[dd.treatment] == 0]
    treatment_group = data[data[dd.treatment] == 1]
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL.PngImagePlugin').disabled = True
    #Overlap Assessment (Propensity Score Distribution)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    xLabel = f'\nParameters {len(combined_column_names)} target {target}'
    plt.figure()
    plt.hist(control_group[dd.propensity_scores], bins=20, alpha=0.5, color='blue', label='Control Group')
    plt.hist(treatment_group[dd.propensity_scores], bins=20, alpha=0.5, color='orange', label='Treatment Group')    
    plt.xlabel('Propensity Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(f'Propensity Score Distribution {xLabel}')
    # Specify the directory where you want to save the plot

    # Generate a file name with the timestamp

    file_name = f'{model_name}_{x}_categorical_{y}_{timestamp}.png'


    # Save the plot to the generated file name
    plt.savefig(folder_name + file_name)
    return
def save_dataset(matched_df: pd.DataFrame, model_name):
    #create a file for each sample combination
    folder_name = FP.build_path
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    file_name = f'{model_name}_results.csv'
    file_path = os.path.join(folder_name, file_name)
    try:
        matched_df.to_csv(file_path, index=False)
    except Exception as e:
        logger.error(f"An error occurred while writing to the file: {e}")
    return

def find_diff(df:pd.DataFrame, col:str, model_name):
    matched_grouped = df.groupby('matched_group')
    differences = matched_grouped[col].diff().abs()    
    result_df = differences.groupby(df['matched_group']).mean().reset_index()
    result_df['model'] = model_name
    return result_df

def find_cat(df:pd.DataFrame, col:str, model_name):   
    grouped_df = df.groupby('matched_group')
    results = []
    for group_name, group_data in grouped_df:
        diff = group_data[col].nunique() == 1
        results.append({
            'matched_group': group_name,
            f'{col}': diff,
            'model': model_name
        })
    result_df = pd.DataFrame(results)
    #logger.debug(f'find cat function: \n{result_df.head()}')
    return result_df

def plot_barplot(psm_results:pd.DataFrame):
    folder_name = FP.build_path  
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    # #logger.debug(f'psm results in plot barplot: \n{psm_results.head()}')
    col = psm_results.columns.tolist()
    plt.figure()
    plt.clf()
    psm_results.to_csv(folder_name + f'{col[1]}_diff.csv', index=False)
    sns.countplot(x='model', hue=col[1], data=psm_results)
    plt.xlabel('Model')
    plt.ylabel('Count')
    plt.title('Count of Records by Model and True/False Distribution')
    plt.savefig(folder_name + f'/{col[1]}_diff.png')
    plt.close()  
    return

def plot_boxplot(psm_results:pd.DataFrame):
    folder_name = FP.build_path
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    #logger.debug(f'psm results in plot boxplot: \n{psm_results.head()}')
    col = psm_results.columns.tolist()
    plt.clf()
    psm_results.to_csv(folder_name + f'{col[1]}_diff.csv', index=False)
    sns.boxplot(data= psm_results, y=f'{col[1]}', x='model')
    plt.savefig(folder_name + f'{col[1]}_diff.png')
    plt.close() 
    return

def statistics(df: pd.DataFrame):
    folder_name = FP.build_path  
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
    # #logger.debug(f'psm results in plot barplot: \n{psm_results.head()}')
    metrics = [col for col in df.columns if col != 'Model Name']
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        sns.barplot(x='Model Name', y=metric, data=df, ax=axes[row, col])
        axes[row, col].set_title(f'{metric} by Model')
    plt.tight_layout()
    # Save the plots as image files
    for metric in metrics:
        filename = f'{metric.replace(" ", "_").lower()}_plot.png'
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Model Name', y=metric, data=df)
        plt.title(f'{metric} by Model')
        plt.savefig(folder_name + filename)
        plt.close()

def calculate_metrics(df: pd.DataFrame, col:str):
    y_true = df.loc[df['treatment'] == 1, col].tolist()
    y_pred = df.loc[df['treatment'] == 0, col].tolist()
    #accuracy = accuracy_score(y_true, y_pred)
    #precision = precision_score(y_true, y_pred, average='micro')
    #recall = recall_score(y_true, y_pred, average='micro')
    #f1 = f1_score(y_true, y_pred, average='micro')
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    #return accuracy, precision, recall, f1, 
    return mse, mae, rmse, r2

def calculate_matches(group):
    group['sex_match'] = int(group['sex'].nunique() == 1)
    group['race_match'] = int(group['race'].nunique() == 1)
    group['ethnicity_match'] = int(group['ethnicity'].nunique() == 1)
    return group


def stats(model_name, df: pd.DataFrame):
    folder_name = FP.build_path
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    logger.debug(f'df in stats: \n{df.head()}')

    new_df = df.groupby('matched_group').apply(calculate_matches).reset_index(drop=True)
    race_match_frequency = new_df['race_match'].mean()
    ethnicity_match_frequency = new_df['ethnicity_match'].mean()
    sex_match_frequency = new_df['sex_match'].mean()
    
    #accuracy_age, precision_age, recall_age, f1_age, 
    mse_age, mae_age, rmse_age, r2_age = calculate_metrics(df, col='age')
    #accuracy_bmi, precision_bmi, recall_bmi, f1_bmi, 
    mse_bmi, mae_bmi, rmse_bmi, r2_bmi = calculate_metrics(df, col='bmi_val')

    file_name = f'{model_name}_results_final.csv'
    
    data = {
        'Model Name': [model_name],
        # 'Age Accuracy': [accuracy_age],
        # 'Age Precision': [precision_age],
        # 'Age Recall': [recall_age],
        # 'Age F1 Score': [f1_age],
        'Age mse': [mse_age],
        'Age mae': [mae_age],
        'Age rmse': [rmse_age],
        'Age r2': [r2_age],
     
        # 'BMI Accuracy': [accuracy_bmi],
        # 'BMI Precision': [precision_bmi],
        # 'BMI Recall': [recall_bmi],
        # 'BMI F1 Score': [f1_bmi],
        'BMI mse': [mse_bmi],
        'BMI mae': [mae_bmi],
        'BMI rmse': [rmse_bmi],
        'BMI r2': [r2_bmi],
        'race_match_frequency': [race_match_frequency],
        'ethnicity_match_frequency': [ethnicity_match_frequency],
        'sex_match_frequency': [sex_match_frequency],
        #'BMI mean': [bmi_mean],
        #'AGE mean': [age_mean]
    }
    metrics_df = pd.DataFrame(data)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'{model_name}_metrics_{timestamp}.csv'
    metrics_df.to_csv(folder_name + file_name, index=True)
    return  metrics_df
