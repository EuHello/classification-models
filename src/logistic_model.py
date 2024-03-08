#!usr/bin/env python3
import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from utils.load_input import load_input
from utils.measure import plot_roc_curve, plot_confusion_matrix, get_score

pd.set_option('display.width', 0)


def main():
    logging.basicConfig(level=logging.INFO)

    data, target, features = load_input()
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=1)
    del data, target
    logging.info(f"Training test split. X_train shape = {X_train.shape}, y_train shape = {y_train.shape}, "
                 f"X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")

    logging.info(f"Running Grid Search for Logistics Model")
    lr_model = LogisticRegression()
    # print(lr_model.get_params())

    params = {
        'C': [0.0001, 0.001, 0.01, 0.05, 0.08, 0.1, 0.3, 0.5, 0.8, 1, 10],
        # 'penalty': ['l1', 'l2'],
        'solver': ['lbfgs', 'liblinear']
    }

    grid = GridSearchCV(estimator=lr_model, param_grid=params, scoring='accuracy', cv=5)
    grid.fit(X_train, y_train)

    logging.info(f"The best score is {grid.best_score_}")
    logging.info(f"The best params are {grid.best_params_}")

    # Prediction
    optimised_lr = grid.best_estimator_
    y_pred = optimised_lr.predict(X_test)

    weights = list(zip(features, optimised_lr.coef_[0]))
    logging.info(f"The weights are: \n{weights}")

    get_score(y_test, y_pred)
    plot_roc_curve(y_test, y_pred, 'Logistic Regression')
    plot_confusion_matrix(y_test, y_pred, optimised_lr.classes_)

    return 0


if __name__ == '__main__':
    main()
