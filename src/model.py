#!usr/bin/env python
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

pd.set_option('display.width', 0)
input_file = 'processed.csv'


def load_processed_inputs():
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


def evaluate_model_accuracy(y_pred, y_true):
    """
    Evaluate model's accuracy and prints F1 score, precision, recall

    Args:
        y_pred: numpy ndarray
        y_true: numpy ndarray

    Returns:
        accuracy: float
    """
    y_pred = y_pred.reshape(-1)
    accuracy = np.mean(y_pred == y_true)
    logging.info(f"Accuracy Score = {accuracy}")
    cm = confusion_matrix(y_true, y_pred)
    logging.debug(f"Confusion Matrix = \n{cm}")
    tn, fp, fn, tp = cm.ravel()
    logging.info(f"tp = {tp}, fp = {fp}, fn = {fn}, tn = {tn}")
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    f1_score_cal = 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(y_true, y_pred)
    logging.info(f"precision = {precision}, recall = {recall}, f1_score_cal = {f1_score_cal}, f1 score = {f1}")
    return accuracy


def build_logistic_model(X_train, y_train, X_cv, y_cv):
    """
    Build Sklearn logistic regression

    Args:
        X_train: X training array (n_samples, n_features)
        y_train: y training array (n_samples)
        X_cv: X cross validation array (n_samples, n_features)
        y_cv: y cross validation array (n_samples)

    Returns:
        y_pred: predictions list
        score: model's score float
    """
    logging.info(f"Running Logistics Model")
    lr_model = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = lr_model.predict(X_cv)

    score = lr_model.score(X_cv, y_cv)
    logging.info(f"Logistics Model score = {score}")
    evaluate_model_accuracy(y_pred, y_cv)
    return y_pred, score


def build_neural_network(X_train, y_train, X_cv, y_cv, alpha_, lambda_):
    """
    Build Tensorflow neural network

    Args:
        X_train: X training array (n_samples, n_features)
        y_train: y training array (n_samples)
        X_cv: X cross validation array (n_samples, n_features)
        y_cv: y cross validation array (n_samples)
        alpha_: learning rate float
        lambda_: regularization float

    Returns:
        loss: loss from training set, float
        val_loss: loss from validation set, float
        diff: loss minus val_loss, float
    """
    BATCH_SIZE = 32
    EPOCHS = 80

    logging.info(f"Running Neural Network: Hyper Parameters alpha_ = {alpha_}, lambda = {lambda_}")
    model = Sequential([
        tf.keras.Input(shape=(18,)),
        Dense(units=20, activation='relu', name='layer1', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        Dense(units=4, activation='relu', name='layer2', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        Dense(units=1, activation='linear', name='layer3', kernel_regularizer=tf.keras.regularizers.l2(lambda_))
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=alpha_),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_cv, y_cv))

    # prediction
    y_logits = model.predict(X_cv)
    y_hat = tf.math.sigmoid(y_logits).numpy()
    y_hat = np.where(y_hat >= 0.5, 1, 0)
    evaluate_model_accuracy(y_hat, y_cv)

    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    return loss, val_loss, loss - val_loss


def build_nn_models(X_train, y_train, X_cv, y_cv, alphas, lambdas):
    """
    Iterate to tune hyperparameters for NN, by providing a list of alphas and/or lambdas. Prints out the best values.

    Args:
        X_train: X training array (n_samples, n_features)
        y_train: y training array (n_samples)
        X_cv: X cross validation array (n_samples, n_features)
        y_cv: y cross validation array (n_samples)
        alphas: learning rate list
        lambdas: regularization list

    Returns: none
    """
    logging.info(f"Tuning Hyperparameters for NN")
    hist = []

    if len(alphas) > 1:
        logging.info(f"begin iterating for alphas = {alphas}")
        for i in range(len(alphas)):
            alpha_ = alphas[i]
            lambda_ = lambdas[0]
            loss, val_loss, diff = build_neural_network(X_train, y_train, X_cv, y_cv, alpha_, lambda_)
            hist.append([alpha_, lambda_, loss, val_loss, diff])
        logging.debug(f"finished iterating alphas = {alphas}")

    if len(lambdas) > 1:
        logging.debug(f"begin iterating for lambdas = {lambdas}")
        for i in range(len(lambdas)):
            alpha_ = alphas[0]
            lambda_ = lambdas[i]
            loss, val_loss, diff = build_neural_network(X_train, y_train, X_cv, y_cv, alpha_, lambda_)
            hist.append([alpha_, lambda_, loss, val_loss, diff])
        logging.debug(f"finished iterating lambdas = {lambdas}")

    logging.info("Summarising results")
    val_loss = []
    loss_minus_val_loss = []
    for i in range(len(hist)):
        logging.info(f"History index {i}: alpha, lambda, loss, val_loss, loss-val_loss = {hist[i]}")
        val_loss.append(hist[i][3])
        loss_minus_val_loss.append(hist[i][4])
    logging.info(f"Best cv loss: index = {np.argmin(val_loss)}")
    logging.info(f"Best loss-val_loss: index = {np.argmax(loss_minus_val_loss)}")


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading inputs")
    data_loaded = load_processed_inputs()

    data, target = select_xy(data_loaded)
    X_train, X_cv, y_train, y_cv = train_test_split(data, target, test_size=0.10, random_state=1)
    logging.info(f"Training and CV split complete. "
                 f"X_train shape = {X_train.shape}, y_train shape = {y_train.shape}, "
                 f"X_cv shape = {X_cv.shape}, y_cv shape = {y_cv.shape}")

    # Run Logistic Model
    log_y_pred, log_score = build_logistic_model(X_train, y_train, X_cv, y_cv)

    # Untuned Neural Network
    untuned_train_loss, untuned_cv_loss, _ = build_neural_network(X_train, y_train, X_cv, y_cv, 0.001, 0)

    # Tuning hyperparameters
    # Tune alpha
    # build_nn_models(X_train, y_train, X_cv, y_cv,
    #                 [0.0001, 0.0005, 0.001, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1, 0.5, 10.], [0])

    # Tune lambda
    # build_nn_models(X_train, y_train, X_cv, y_cv, [0.03],
    #                 [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 10.])

    # Tuned Neural Network
    tuned_train_loss, tuned_cv_loss, _ = build_neural_network(X_train, y_train, X_cv, y_cv, 0.03, 0.00001)
    logging.info(f"Untuned Neural Network: Training Loss: {untuned_train_loss}, CV Loss: {untuned_cv_loss}")
    logging.info(f"Tuned Neural Network: Training Loss: {tuned_train_loss}, CV Loss: {tuned_cv_loss}")

    logging.info("Models finished")

    return


if __name__ == '__main__':
    main()
