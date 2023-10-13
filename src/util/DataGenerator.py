import pandas as pd
import numpy as np
import logging.config
import datetime
import os
import matplotlib.pyplot as plt
import src.util.FileProvider as FP
from src.datamodel.Column import DataDictionary as dd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, roc_auc_score
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
    df = pd.read_csv(path)
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
    label_encoder = LabelEncoder()
    df[dd.patientID] = df[dd.patientID].str.extract('(\d+)').astype(int)
    
    # Label encode the specified columns
    for column in label_columns:
        df[column] = label_encoder.fit_transform(df[column])
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
def save_dataset(matched_df: pd.DataFrame, case, model_name):
    #create a file for each sample combination
    folder_name = FP.build_path
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    x, y = case
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    file_name = f'{model_name}_{x}_categorical_{y}_{timestamp}.csv'


    file_path = os.path.join(folder_name, file_name)
    try:
        matched_df.to_csv(file_path, index=False)
    except Exception as e:
        logger.error(f"An error occurred while writing to the file: {e}")

    return

def stats(model_name, df: pd.DataFrame):
    folder_name = FP.build_path
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    race_matches = 0
    ethnicity_matches = 0
    sex_matches = 0
    '''
    df['age'].fillna(0, inplace=True)
    df['bmi_val'].fillna(0, inplace=True)
    df['sex'].fillna(0, inplace=True)
    df['race'].fillna(0, inplace=True)
    df['ethnicity'].fillna(0, inplace=True)
    '''
    logger.debug(f'df in stats: \n{df.head()}')
    pairs = [df.iloc[i:i+2] for i in range(0, len(df), 2)]

    # Sample threshold values for age and BMI differences
    age_threshold = 5
    bmi_threshold = 5

    results = []

    for pair in pairs:
        patient1 = pair.iloc[0]
        patient2 = pair.iloc[1]
        difference = abs(patient1['Propensity Score'] - patient2['Propensity Score'])
        age_difference = abs(patient1['age'] - patient2['age'])
        bmi_difference = abs(patient1['bmi_val'] - patient2['bmi_val'])

        similarity = int(age_difference <= age_threshold and bmi_difference <= bmi_threshold)

        
        #Check if 'race' and 'ethnicity' match between the two patients
        if patient1['race'] == patient2['race']:
            race_matches += 1
            race_difference = 0
        else:
            race_difference = 1

        if patient1['ethnicity'] == patient2['ethnicity']:
            ethnicity_matches += 1
            ethnicity_difference = 0
        else:
            ethnicity_difference = 1

        # Check if 'sex' matches between the two patients
        if patient1['sex'] == patient2['sex']:
            sex_matches += 1
            sex_difference = 0
        else:
            sex_difference = 1
        results.append([difference, age_difference, bmi_difference, similarity, sex_difference, race_difference, ethnicity_difference])
    total_pairs = len(pairs)

    race_match_frequency = race_matches / total_pairs
    ethnicity_match_frequency = ethnicity_matches / total_pairs
    sex_match_frequency = sex_matches / total_pairs
    # Convert results to a DataFrame for easier calculation
    results_df = pd.DataFrame(results, columns=['DIFF', 'AGE_DIFF', 'BMI_DIFF', 'Similarity','SEX_DIFF', 'RACE_DIFF', 'ETHNICITY_DIFF'])
    scaler = MinMaxScaler()
    #results_df[['AGE_DIFF', 'BMI_DIFF']] = 1 - scaler.fit_transform(results_df[['AGE_DIFF', 'BMI_DIFF']])

    # Calculate binary classification metrics
    y_true = results_df['Similarity']
    y_pred = [1] * len(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Calculate mean squared error
    mse_age = mean_squared_error(y_true, results_df['AGE_DIFF'])
    mse_bmi = mean_squared_error(y_true, results_df['BMI_DIFF'])

    # Normalize and invert the differences for ROC AUC calculation
    
    #results_df[['Age Difference', 'BMI Difference']] = 1 - scaler.fit_transform(results_df[['Age Difference', 'BMI Difference']])

    roc_auc_age = roc_auc_score(y_true, results_df['AGE_DIFF'])
    roc_auc_bmi = roc_auc_score(y_true, results_df['BMI_DIFF'])

    output_file_path = f'{model_name}_output_'
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    # Open the file in write mode
    with open(folder_name + output_file_path + timestamp, 'w') as file:
        file.write(f'Accuracy: {accuracy}\n')
        file.write(f'Precision: {precision}\n')
        file.write(f'Recall: {recall}\n')
        file.write(f'F1 Score: {f1}\n')
        file.write(f'MSE for Age: {mse_age}\n')
        file.write(f'MSE for BMI: {mse_bmi}\n')
        file.write(f'ROC AUC Score (Age): {roc_auc_age}\n')
        file.write(f'ROC AUC Score (BMI): {roc_auc_bmi}\n')
        file.write(f'race_match_frequency: {race_match_frequency}\n')
        file.write(f'ethnicity_match_frequency: {ethnicity_match_frequency}\n')
        file.write(f'sex_match_frequency: {sex_match_frequency}\n')
        bmi_mean = df['BMI_DIFF'].mean()
        age_mean = df['AGE_DIFF'].mean()
        file.write(f'BMI mean: {bmi_mean} AGE mean: {age_mean}\n')
    file_name = f'{model_name}_results_final.csv'
    results_df.to_csv(folder_name + file_name, index=False)

    return results_df

def metrics(model_name, df: pd.DataFrame):
    folder_name = FP.build_path
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    columns_to_plot = ['DIFF', 'AGE_DIFF', 'BMI_DIFF']

    for column in columns_to_plot:
        # Filter the DataFrame to include only rows where the current column is not null
        filtered_df = df.dropna(subset=[column])
        
        # Create and save the box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(filtered_df[column])
        ax.set_title(f'{model_name} {column} Box Plot')
        ax.set_ylabel(f'{column} Value')
        ax.set_xlabel(f'Patients')
        file_name = f'{model_name}_{column}_boxplot_{timestamp}.png'
        plt.savefig(os.path.join(folder_name, file_name))
        plt.close()
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cat_plot = ['SEX_DIFF', 'RACE_DIFF', 'ETHNICITY_DIFF']
    for column in cat_plot:
        # Create a categorical bar plot for 'SEX_DIFF'
        fig, ax = plt.subplots(figsize=(10, 6))
        filtered_df = df.dropna(subset=[column])
        
        # Count the occurrences of each category in the 'SEX_DIFF' column
        diff_counts = filtered_df[column].value_counts()
        
        # Create the bar plot
        diff_counts.plot(kind='bar', ax=ax)
        
        ax.set_title(f'{model_name} {column} Categorical Bar Plot')
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        file_name = f'{model_name}_{column}_barplot_{timestamp}.png'
        plt.savefig(os.path.join(folder_name, file_name))
        plt.close()

    print("Plots saved with timestamp:", timestamp)
    return
