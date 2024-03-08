import logging
import pandas as pd
import os

base_path = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(base_path, 'processed.csv')


def load_input():
    """
    Loads input from csv file. Selects relevant features. Returns train data and target data.

    Args: none

    Returns:
        train: numpy ndarray of shape (n_samples, n_features)
        target: numpy ndarray of shape (n_samples)
        features: feature names list (n_features)
    """
    logging.info("Loading input for model")
    data = pd.read_csv(file_path)

    data = data.drop(
        columns=['ID', 'Start Smoking', 'Stop Smoking', 'COPD History', 'Taken Bronchodilators', 'Dominant Hand']
    )
    target = data.pop('Lung Cancer Occurrence')
    logging.debug(f"Inputs loaded: \n{data[0:2]}")

    return data.to_numpy(), target.to_numpy(), data.columns.values
