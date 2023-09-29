import pandas as pd
import numpy as np
import logging.config
import sys
import random
from src.datamodel.Column import DataDictionary as dd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import datetime
import src.util.FileProvider as FP
logger = logging.getLogger(__name__)

    #create random variables
def set_random_seeds():
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    #convert cols to numeric
def convert_to_numeric(data):
    numeric_columns = data.select_dtypes(include=['object']).columns
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    #train test split
def split_data(data, parameters, target):
    X = data[parameters]
    y = data[target]
    y = y.values.ravel()
    return train_test_split(X, y, test_size=0.2, random_state=42)

    #training with early stopping
def train_neural_network(X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[1]
    model = create_model(input_dim)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping],
    )
    return model, history
#plot history of training loss and accuracy
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    #accuracy plot save
def save_accuracy_plot(folder_name, timestamp):
    file_name = f'nn_accuracy_plot_{timestamp}.png'
    plt.savefig(folder_name + file_name)

    #evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC on test data: {roc_auc:.4f}")

    #predict psm score
def predict_and_score_psm(model, X_test, y_test, X, dd, data, target):
    psm_test = model.predict(X_test)
    psm = model.predict(X)
    logger.debug(f'number of propensity_scores: {len(psm_test)}')
    data[dd.propensity_scores] = psm
    y_pred_test = (psm_test > 0.5).astype(int)

    metrics_dict = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Testing': [
            accuracy_score(y_test, y_pred_test),
            precision_score(y_test, y_pred_test),
            recall_score(y_test, y_pred_test),
            f1_score(y_test, y_pred_test),
        ],
    }

    metrics_df = pd.DataFrame(metrics_dict)
    logger.debug(metrics_df)

    #generate the dataframe to save
def generate_new_dataframe(data, dd, selected, psm):
    data_dict = {col: data[col] if col != dd.propensity_scores else psm for col in selected}
    new_df = pd.DataFrame(data_dict)
    new_df[dd.propensity_scores] = psm
    return new_df

def nnModel(data: pd.DataFrame, parameters: list, target: list) -> pd.DataFrame:
    """
    Function preforms neural network training on provided data frame and returns psm scores for said data frame
    :param data: data frame to process
    :param parameters: number of  parameters (ie: age)
    :param target: target
    :return: pd.DataFrame: returns psm scores added to the data frame
    """
    set_random_seeds()
    convert_to_numeric(data)
    
    X_train, X_test, y_train, y_test = split_data(data, parameters, target)

    if not X_train.empty:
        model, history = train_neural_network(X_train, y_train, X_test, y_test)
        plot_training_history(history)
        
        folder_name = FP.build_path
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_accuracy_plot(folder_name, timestamp)

        evaluate_model(model, X_test, y_test)

        metrics_df = predict_and_score_psm(model, X_test, y_test, data[parameters], dd, data, target)

    else:
        logger.warning(f'Must select more than 0 columns to generate a psm score')
        sys.exit()

    selected = [data.columns[0]] + parameters + target
    new_df = generate_new_dataframe(data, dd, selected, data[dd.propensity_scores])

    return new_df, metrics_df

def create_model(input_dim):
    #model to train
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5, seed=42),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5, seed=42),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5, seed=42),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5, seed=42),
        tf.keras.layers.Dense(32, activation='relu', kernel_initializer='glorot_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5, seed=42),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model