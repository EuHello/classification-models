#!/bin/bash

url=https://techassessment.blob.core.windows.net/aiap16-assessment-data/lung_cancer.db
local_dir="./data"
db_file_path="./data/lung_cancer.db"

if [ -f "$db_file_path" ]; then
  echo "DB file exists."
  else
    echo "Downloading DB from $url."
    wget -P "$local_dir" "$url"
fi


python3 --version


if [ -z "$1" ]; then
  echo "Run pre-processing and models."
  python3 ./src/preprocess.py
  python3 ./src/model.py
fi


if [ "$1" = "-p" ]; then
  echo "-p provided. Preprocessing"
  python3 ./src/preprocess.py

elif [ "$1" = "-ta" ]; then
  echo "-ta provided. Running Tuning for Alpha in Neural Network"
  python3 ./src/model.py "$1"

elif [ "$1" = "-tl" ]; then
  echo "-ta provided. Running Tuning for Lambda in Neural Network"
  python3 ./src/model.py "$1"

elif [ "$1" = "-m" ]; then
  echo "-m provided. Running both models"
  python3 ./src/model.py "$1"

else
  echo "Unknown argument $1"
  echo "  No argument: run pre-processing and models."
  echo "  -p: run pre-processing only."
  echo "  -m: run models only."
  echo "  -ta or -tl: iterate for alphas or lambdas respectively for neural network."
  exit 1
fi
