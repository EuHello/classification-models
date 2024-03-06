#!usr/bin/env python3
import numpy as np
import pandas as pd
import re
import logging
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

import utils.read_db as read_db

pd.set_option('display.width', 0)


def load_data():
    """
    Loads data from Database with pandas

    Args: None

    Returns:
        data: pandas dataframe
        shape : tuple (n_samples, n_features)
    """
    logging.info(f"Loading Data at path = {read_db.check_path()}")
    data = read_db.get_df()
    logging.info(f"Loaded Data shape = {data.shape}")
    return data, data.shape


def clean_age(df: pd.DataFrame):
    """
    Drops data with negative Age Values

    Args:
        df: pandas dataframe

    Returns:
        df: pandas dataframe
    """
    org_shape = df.shape
    logging.debug(f"Running clean_age method")
    count_neg_age = np.sum(df['Age'] <= 0)
    new_df = df[df['Age'] > 0]
    new_df = new_df.reset_index(drop=True)

    logging.debug(f"Dropped {count_neg_age} negative Age values. New data shape[0] is {new_df.shape[0]}. "
                  f"Expected = {org_shape[0] - count_neg_age}")
    return new_df


def clean_gender(df: pd.DataFrame):
    """
    Cleans Gender by capitalising all letters
    "Male" -> "MALE"
    "Female" -> "FEMALE"
    NAN -> ignore

    Args:
        df: pandas dataframe

    Returns:
        df: pandas dataframe
    """
    logging.debug(f"Running clean_gender method")
    logging.debug(df['Gender'].value_counts())
    df['Gender'] = df['Gender'].map(lambda x: x.strip().upper())
    df['Gender'] = df['Gender'].map({"MALE": "MALE", "FEMALE": "FEMALE", "NAN": "MALE"})
    logging.debug(f"Completed clean_gender method")
    logging.debug(df['Gender'].value_counts())
    return df


def clean_pollution_exposure(data: pd.DataFrame):
    """
    Cleans Air Pollution Exposure by backfilling NONE value

    Args:
        data: pandas dataframe

    Returns:
        data: pandas dataframe
    """
    logging.debug(f"Running clean_pollution_exposure method")
    logging.debug(f"missing vals = {data['Air Pollution Exposure'].isnull().sum()}")
    data['Air Pollution Exposure'] = data['Air Pollution Exposure'].bfill()
    logging.debug(f"missing vals = {data['Air Pollution Exposure'].isnull().sum()}")
    return data


def match_year_format(val):
    """
    Matches string to a year format YYYY

    Args:
        val: string

    Returns:
        True: string matches YYYY year format
        False: invalid string
    """
    pattern = r'^(19|20)\d\d$'
    if re.match(pattern, val):
        return True
    else:
        return False


def make_smoking_features(row: pd.Series):
    """
    Engineer Features from 'Start Smoking' & 'Stop Smoking'

    Args:
        row: row from df.apply()

    New Features:
        smoker_history:          1 or 0
        smoker_stopped_smoking:  1 or 0
        smoker_still_smoking:    1 or 0
        total_years_smoked:      years integer, based on latest data year 2024

    Returns:
        row: pandas dataframe row with above new features added
    """

    latest_year = 2024
    smoker_history = 0
    smoker_stopped_smoking = 0
    smoker_still_smoking = 0
    total_years_smoked = 0

    if match_year_format(row['Start Smoking']):
        year_start_smoking = int(row['Start Smoking'])
        smoker_history = 1
    elif row['Start Smoking'] == 'Not Applicable':
        smoker_history = 0

    if match_year_format(row['Stop Smoking']):
        year_stop_smoking = int(row['Stop Smoking'])
        smoker_stopped_smoking = 1
        smoker_still_smoking = 0
    elif row['Stop Smoking'] == 'Still Smoking':
        smoker_stopped_smoking = 0
        smoker_still_smoking = 1

    if smoker_history == 1 and smoker_stopped_smoking == 1:
        total_years_smoked = max((year_stop_smoking - year_start_smoking), 1)
    elif smoker_history == 1 and smoker_still_smoking == 1:
        total_years_smoked = max((latest_year - year_start_smoking), 1)
    elif smoker_history == 0:
        total_years_smoked = 0

    row['smoker_history'] = smoker_history
    row['smoker_stopped_smoking'] = smoker_stopped_smoking
    row['smoker_still_smoking'] = smoker_still_smoking
    row['total_years_smoked'] = total_years_smoked

    return row


def engineer_smoking_features(data: pd.DataFrame):
    """
    Apply (pandas method) to make new features from 'Start Smoking' & 'Stop Smoking'

    Args:
        data: pandas dataframe

    Returns:
        data dataframe with new smoking features below -
        smoker_history:          1 or 0
        smoker_stopped_smoking:  1 or 0
        smoker_still_smoking:    1 or 0
        total_years_smoked:      years integer, based on latest data year 2024
    """
    logging.debug("Running engineer_smoking_features method")

    data = data.apply(make_smoking_features, axis='columns')

    logging.debug(f"Completed engineer_smoking_features")
    logging.debug(f" total smokers with history={np.sum(data['smoker_history'])}, "
                  f"total still smoking={np.sum(data['smoker_still_smoking'])}, "
                  f"total stopped smoking={np.sum(data['smoker_stopped_smoking'])}, "
                  f"total years smoked>0 {np.sum(data['total_years_smoked'] > 0)}")
    return data


def ordinal_encoder(data: pd.DataFrame):
    """
    Label encodes data for the following:

    Gender: Binary data, 2 values
    Genetic Markers: Binary data, 2 values
    Air Pollution Exposure: Ordinal data, 4 values
    Frequency of Tiredness: Ordinal data, 3 values

    Args:
        data: pandas dataframe

    Returns:
        data: pandas dataframe
    """
    logging.debug("Running ordinal_encoder method")
    binary_ordinal_features = ['Gender', 'Genetic Markers', 'Air Pollution Exposure', 'Frequency of Tiredness']
    gender_categories = ['FEMALE', 'MALE']
    genetic_marker_categories = ['Not Present', 'Present']
    air_pollution_categories = ['Low', 'Medium', 'High']
    tiredness_categories = ['None / Low', 'Medium', 'High']

    enc = OrdinalEncoder(
        categories=[gender_categories, genetic_marker_categories, air_pollution_categories, tiredness_categories]
    )

    enc.fit(data[binary_ordinal_features])
    logging.debug(f"Ordinal encoder categories = {enc.categories_}")
    data[binary_ordinal_features] = enc.transform(data[binary_ordinal_features])
    logging.debug(data[binary_ordinal_features].describe())

    return data


def one_hot_encoder(data: pd.DataFrame):
    """
    One Hot encodes data for the following:

    COPD History: Yes, No - ignore None. None shows as [0,0]
    Taken Bronchodilators: Yes, No - ignore None. None shows as [0,0]
    Dominant Hand: Nominal data, 3 values

    Args:
        data: pandas dataframe

    Returns:
        data_encoded: input pandas dataframe with encoded features added in
    """
    logging.debug("Running one_hot_encoder method")
    features = ['COPD History', 'Taken Bronchodilators', 'Dominant Hand']
    copd_categories = ['No', 'Yes']
    bronchodilators_categories = ['No', 'Yes']
    hand_categories = ['Right', 'Left', 'RightBoth']

    enc = OneHotEncoder(
        categories=[copd_categories, bronchodilators_categories, hand_categories],
        handle_unknown='ignore',
        sparse_output=False
    )

    enc.fit(data[features])
    logging.debug(f"One Hot encoder categories = {enc.categories_}")
    encoded = enc.transform(data[features])
    logging.debug(f"encoded output shape={encoded.shape}")

    new_features = enc.get_feature_names_out()
    one_hot_df = pd.DataFrame(data=encoded, columns=new_features)

    data_encoded = pd.concat([data, one_hot_df], axis=1)
    logging.debug(f"data_encoded.shape={data_encoded.shape}")

    return data_encoded


def scale_features(data: pd.DataFrame):
    """
    Scales numeric features with sklearn StandardScaler(). Age, Last Weight, Current Weight, total_years_smoked

    Args:
        data: pandas dataframe

    Returns:
        data: pandas dataframe with features scaled
    """
    logging.debug("Running scale_features method")
    scaler = StandardScaler()
    features = ['Age', 'Last Weight', 'Current Weight', 'total_years_smoked']
    scaled_data = scaler.fit_transform(data[features])

    scaled_df = pd.DataFrame(scaled_data, columns=features)
    logging.debug(f"scaler output.shape={scaled_df.shape}")

    data[features] = scaled_df
    logging.debug(f"scaled data.shape={data.shape}")
    logging.debug(f"describing the scaled features ={data[features].describe()}")

    return data


def main():
    logging.basicConfig(level=logging.INFO)

    data, org_shape = load_data()

    logging.info("Pre-processing data now..")
    data = clean_age(data)
    data = clean_gender(data)
    data = clean_pollution_exposure(data)
    data = engineer_smoking_features(data)
    logging.debug(data.head())

    data = ordinal_encoder(data)
    data = one_hot_encoder(data)
    data = scale_features(data)

    logging.debug(data.head())
    logging.info(f"Final Data shape = {data.shape}")
    data.to_csv("processed.csv", index=False)
    logging.info("Pre-processing COMPLETED")

    return 0


if __name__ == "__main__":
    main()
