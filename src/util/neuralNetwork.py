import pandas as pd
import numpy as np
import logging.config
import sys
import random
from src.datamodel.Column import DataDictionary as dd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import datetime
import src.util.FileProvider as FP
logger = logging.getLogger(__name__)

def nnModel(data: pd.DataFrame, parameters: list, target: list) -> pd.DataFrame:
    numeric_columns = data.select_dtypes(include=['object']).columns

    # Set random seeds for TensorFlow and NumPy
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
# Convert selected columns to numeric
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    X = data[parameters]
    y = data[target]
    y = y.values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.debug(f'data: {data.info()}')
    if not X.empty:
        # Sample data generation
        # Define hyperparameters to search
        param_grid = {
            'batch_size': [32, 64, 128],
            'epochs': [50, 100, 200]
        }

        # Create the neural network model as a function
        #neural_network = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, input_dim=X_train.shape[1], verbose=0)

        # Perform grid search
        #model = GridSearchCV(estimator=neural_network, param_grid=param_grid, cv=3)
        #grid_result = model.fit(X_train, y_train)
        # Create the model
        input_dim = X_train.shape[1]
        model = create_model(input_dim)
        # Train the model with early stopping based on validation loss
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, callbacks=[early_stopping])

        # Plot training and validation loss and accuracy
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
         
        folder_name = FP.build_path
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f'nn_accuracy_plot_{timestamp}.png'

            # Save the plot to the generated file name
        plt.savefig(folder_name + file_name)

        # Evaluate the model on test data
        y_pred = model.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f"AUC-ROC on test data: {roc_auc:.4f}")

        # Print the best hyperparameters
        #print(f"Best: {grid_result.best_params_}, Score: {grid_result.best_score_}")

        # Make predictions on the test set
        
        psm_test = model.predict(X_test)
        psm = model.predict(X)
        logger.debug(f'number of propensity_scores: {len(psm_test)}')
        data[dd.propensity_scores] = psm
        y_pred_test = (psm_test > 0.5).astype(int)
        
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

def create_model(input_dim):
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