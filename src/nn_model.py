#!usr/bin/env python3
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

pd.set_option('display.width', 0)
input_file = 'processed.csv'


def load_inputs():
    data = pd.read_csv(input_file)
    return data


def select_xy(data: pd.DataFrame):
    """
    Selects the relevant features for X by dropping the un-necessary columns.

    Args:
        data: pandas dataframe

    Returns:
        data: pandas dataframe
        y_train: pandas series with Target labels
    """
    logging.debug("Running select_xy method")
    data = data.drop(
        columns=['ID', 'Start Smoking', 'Stop Smoking', 'COPD History', 'Taken Bronchodilators', 'Dominant Hand'])
    target = data.pop('Lung Cancer Occurrence')
    logging.debug(f"Final data columns = \n{data[0:2]}")

    return data.to_numpy(), target.to_numpy()


def model_builder(hp):
    hp_units = hp.Int('units', min_value=8, max_value=128, step=8)
    hp_hidden = hp.Int('hidden', min_value=4, max_value=64, step=4)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3, 8e-3, 1e-2, 5e-2, 8e-2, 1e-1, 5e-1])
    hp_lambda = hp.Choice('l2_lambda', values=[1e-06, 5e-06, 1e-05, 5e-05, 1e-4, 5e-4, 1e-3, 1e-2, 1e-1, 1.0])

    model = Sequential([
        tf.keras.Input(shape=(18,)),
        Dense(units=hp_units, activation='relu', name='layer1', kernel_regularizer=tf.keras.regularizers.l2(hp_lambda)),
        Dense(units=hp_hidden, activation='relu', name='layer2',
              kernel_regularizer=tf.keras.regularizers.l2(hp_lambda)),
        Dense(units=1, activation='linear', name='layer3', kernel_regularizer=tf.keras.regularizers.l2(hp_lambda))
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=['accuracy'])

    return model


def tune_model(X_train, y_train, X_test, y_test):
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=50,
                         factor=3,
                         directory='nn_tuner',
                         project_name='classification',
                         overwrite=False)

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. 
    The optimal number of units in the first densely-connected layer is {best_hps.get('units')}, 
    the optimal number of units in the second densely-connected layer {best_hps.get('hidden')}, 
    the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}, 
    and the optimal lambda for the L2 regularizer is {best_hps.get('l2_lambda')}.
    """)

    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)
    eval_result = hypermodel.evaluate(X_test, y_test)
    logging.info(f"Test Loss, Test Accuracy: {eval_result}")


def main():
    logging.basicConfig(level=logging.DEBUG)

    logging.info("Loading inputs")
    data_loaded = load_inputs()
    data, target = select_xy(data_loaded)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=1)
    logging.info(f"Training test split. X_train shape = {X_train.shape}, y_train shape = {y_train.shape}, "
                 f"X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")

    tune_model(X_train, y_train, X_test, y_test)

    return 0


if __name__ == '__main__':
    main()
