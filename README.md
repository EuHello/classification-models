# Intro


# Overview of Structure


# Instruction for executing pipeline


# Description of pipeline flow
1. Load Data
2. Cleaning - Age, Gender, Air Pollution Exposure
3. Feature Engineering
4. Encoding - ordinal encoding and one hot encoding
5. Scaling and Normalisation
6. Split into train and cross-validation sets
7. Build Models
8. Evaluate Models


# Key findings from EDA


# Feature processing
| Feature                      | Processing          | Details                                                                                                | Reason                                                                                                                                                                |
|------------------------------|---------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Age                          | Clean               | Drop negative values                                                                                   | Negative values are invalid without further information                                                                                                               |
| Gender                       | Clean               | Map all to M and F. Missing value backfilled                                                           | Only 1 instance of missing value in training data. This seems to be rare occurrence.                                                                                  |
| Air Pollution Exposure       | Clean               | Missing value backfilled                                                                               | Only 3 instances of missing value. Rare occurrence. Backfill is okay as low impact                                                                                    |
| Start Smoking + Stop Smoking | Feature Engineering | Create 4 new features - 3 binary(history, present smoker, past smoker), 1 nominal (total years smoked) | The two features are a mix of integers (years) and categorical strings, which make it difficult to be interpreted by humans and as input for models                   |
| Gender                       | Ordinal Encode      | Encode to 0, 1 as this is a binary data after cleaning                                                 |                                                                                                                                                                       |
| Genetic Markers              | Ordinal Encode      | Encode to 0, 1 as this is a binary data                                                                |                                                                                                                                                                       |
| Air Pollution Exposure       | Ordinal Encode      | Encode Low, Medium, High to 0, 1, 2 as this is ordinal data                                            |                                                                                                                                                                       |
| Frequency of Tiredness       | Ordinal Encode      | Encode None/Low, Medium, High to 0, 1, 2 as this is ordinal data                                       |                                                                                                                                                                       |
| COPD History                 | One Hot Encode      | 3 values Yes, No, None.  None is not dropped, and treated as information [0,0] in one-hot encoding     | For None - ~10% of training data, implies this is common, for inference too. Instead of dropping it, this information should be captured as [0,0] in one-hot encoding |
| Taken Bronchodilators        | One Hot Encode      | 3 values Yes, No, None.  None is not dropped, and treated as information [0,0] in one-hot encoding     | Same as above.                                                                                                                                                        |
| Dominant Hand                | One Hot Encode      | 3 nominal values - low cardinality                                                                     |                                                                                                                                                                       |
| Age                          | Z score scaling     | After scaling, values min/max = -3.22 to 3.92                                                          |                                                                                                                                                                       |
| Last Weight                  | Z score scaling     | After scaling, values min/max = -1.69 to 1.71                                                          |                                                                                                                                                                       |
| Current Weight               | Z score scaling     | After scaling, values min/max = -1.97 to 2.69                                                          |                                                                                                                                                                       |
| total_years_smoked           | Z score scaling     | After scaling, values min/max = -1.04 to 4.3                                                           |                                                                                                                                                                       |


# Model Choice

## 1. Logistic Regression

score = 0.6640316205533597

## 2. Neural Network
   a. Tuning Hyper-parameters

| NN architecture | Cross Validation Loss |
|-----------------|-----------------------|
| 20,12,1         | 0.5338                |
| 20,6,1          | 0.5301                |
| 20,4,1          | 0.5261                |
| 20,1            | 0.5267                |
| 19,1            | 0.5264                |
| 12,1            | 0.5309                |

Tune Alpha
alpha params = [0.0001, 0.0005, 0.001, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1, 0.5, 10.]
Selected alpha = 0.03

Tune Lambda
lambda params = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 10.])
Selected lambda = 0.00001


   b. final results

| NN (80 epochs) | Training Loss       | Cross Validation Loss |
|----------------|---------------------|-----------------------|
| Untuned NN     | 0.47575679421424866 | 0.5230705738067627    |
| Tuned NN       | 0.4582006633281708  | 0.49096259474754333   |



# Model Evaluation

## Confusion Matrix

### Logistic Regression

| Total = 1012    | Predicted Positive | Predicted Negative |
|-----------------|--------------------|--------------------|
| Actual Positive | TP = 393           | FN = 143           |
| Actual Negative | FP = 197           | TN = 279           |


### Tuned NN

| Total = 1012    | Predicted Positive | Predicted Negative |
|-----------------|--------------------|--------------------|
| Actual Positive | TP = 447           | FN = 89            |
| Actual Negative | FP = 189           | TN = 287           |



### Accuracy vs Recall vs F1

| Model               | Accuracy (CV) | Precision TP/(TP+FP) | Recall TP/(TP+FN) | F1     |
|---------------------|---------------|----------------------|-------------------|--------|
| Logistic Regression | 0.6640        | 0.6661               | 0.7332            | 0.6980 |
| Tuned NN            | 0.7322        | 0.702830             | 0.8339            | 0.7627 |


Tuned NN is superior. 