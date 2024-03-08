#!/bin/bash

url=https://techassessment.blob.core.windows.net/aiap16-assessment-data/lung_cancer.db
local_dir="./data"
db_file_path="./data/lung_cancer.db"

if [ -f "$db_file_path" ]; then
  echo "DB file exists."
  else
    echo "Downloading dataset from $url"
    wget -P "$local_dir" "$url"
fi


python3 --version


if [ -z "$1" ]; then
  echo "No args. Default: Run pre-processing, followed by Logistic Regression."
  echo "Use args to run for neural network."
  python3 ./src/preprocess.py
  python3 ./src/logistic_model.py
  exit 0

elif [ "$1" = "-p" ]; then
  echo "-p provided. Pre-processing only."
  python3 ./src/preprocess.py
  exit 0

elif [ "$1" = "-lr" ]; then
  echo "-lr provided. Running Logistic Regression Model, tuning, cross-validation, prediction scoring."
  python3 ./src/logistic_model.py
  exit 0

elif [ "$1" = "-nn" ]; then
  echo "-nn provided. Running Neural Network Model, tuning, cross-validation, prediction and scoring."
  python3 ./src/nn_model.py
  exit 0

else
  echo "Unknown argument $1"
  echo "  No argument: run pre-processing and logistic regression model."
  echo "  -p: run pre-processing only."
  echo "  -lr: run Logistic Regression model only."
  echo "  -nn: run Neural Network model only."
  exit 1
fi
