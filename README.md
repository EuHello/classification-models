# Intro
Lung cancer remains the leading cause of cancer death worldwide, including Singapore, claiming millions of lives each year. 
Early detection is crucial for improving survival rates, but current diagnostic methods like chest X-rays and CT scans can be expensive, time-consuming, and prone to misinterpretation.

## Data
 The dataset provided contains patients’ medical information collected from all public hospitals in Singapore. 
 Do note that there could be synthetic features in the dataset. 

You can query the datasets using the following URL: 
https://techassessment.blob.core.windows.net/aiap16-assessment-data/lung_cancer.db


| Attribute              | Description                                                                |
|------------------------|----------------------------------------------------------------------------|
| ID                     | PatientID                                                                  |
| Age                    | Age of the patient                                                         |
| Gender                 | Gender of the patient                                                      |
| COPD History           | Whether the patient has a history of Chronic Obstructive Pulmonary Disease |
| Genetic Markers        | Presence of any genetic markers known to increase the risk of lung cancer  |
| Air Pollution Exposure | Level of air pollution exposure in the patient’s daily life                |
| Last Weight            | Last officially recorded weight of patient                                 |
| Current Weight         | Current officially recorded weight of patient                              |
| Start Smoking          | Year that the patient starts smoking                                       |
| Stop Smoking           | Year that the patient stops smoking                                        |
| Taken Bronchodilators  | Whether the patient is previously prescribed Bronchodilator medications    |
| Frequency of Tiredness | Frequency of patient feeling tiredness in a day                            |
| Dominant Hand          | Dominant hand of the patient                                               |
| Lung Cancer Occurrence | Whether the patient has lung cancer or not. Lung Cancer = 1.               |


# Overview of Structure

Project
- .gitHub -> gitHub actions
- src     -> preprocessing and models
- README.md
- eda.ipynb -> Exploratory Data Analysis
- requirements.txt -> Python packages
- run.sh -> script to run


# Instruction for executing pipeline
| Command and argument | Action                                                     |
|----------------------|------------------------------------------------------------|
| ./run.sh             | Preprocess data, run models.                               |
| ./run.sh -p          | Preprocess data only.                                      |
| ./run.sh -m          | Run both models - logistic regression, and neural network. |
| ./run.sh -ta         | Iterate list of alphas for neural network and find best.   |
| ./run.sh -tl         | Iterate list of lambdas for neural network and find best.  |



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
Refer to eda.ipynb

**Data Size**: 10348 points, of which 229 datapoints are invalid (negative Age)

**Features in data:**
 - Numerical features = 3 
 - Year features = 2
 - Categorical features = 7

**Proposed Engineered Features:**
  - 1 numerical feature (total years smoked)
  - 3 categorical feature (smoking history; smoker stopped smoking; smoker still smoking)

**Target:** Fairly balanced, not skewed. Lung Cancer 54.41%, non Lung Cancer 45.59%.

**Observed relation with Target:**
 - Generally, all features show relation with target. Especially categorical features.

**Cleaning is required for Age and Gender:**
 - Age: to drop datapoints with negative values, as examples seem invalid. They do not look like placeholder values.
 - Gender: to clean up dupes (e.g. MALE and male). 1 'NAN' value to be backfilled

**Missing values**
- Air Pollution Exposure: missing count = 3 or 0.03% -> as there are only 3 missing values, it can be easily handled with backfill. 
- COPD History: missing values count = 1112 or 10.75%
- Taken Bronchodilators: missing count = 1061 or 10.25%

**Pre-processing and feature engineering is required for 'Start Smoking' and 'Stop Smoking':**

This is because these features contain both integer year values, together with categorical values 'Not Applicable', and 'Still Smoking'.

As mentioned above, the engineered features are:
- 1 numerical feature (total years smoked)
- 3 categorical feature (smoking history; smoker stopped smoking; smoker still smoking)

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

1. Logistic Regression Model, tuning for C and solver. 
2. Neural Network, 3 layers(1 input, 1 hidden, 1 output), with binarycrossentrophy loss


# Model Evaluation

## Confusion Matrix

### Logistic Regression, Tuned

| Total = 1012    | Predicted Positive | Predicted Negative |
|-----------------|--------------------|--------------------|
| Actual Positive | TP = 415           | FN = 121           |
| Actual Negative | FP = 210           | TN = 266           |


### Neural Network, Tuned

| Total = 1012    | Predicted Positive | Predicted Negative |
|-----------------|--------------------|--------------------|
| Actual Positive | TP = 451           | FN = 85            |
| Actual Negative | FP = 187           | TN = 289           |



### Accuracy vs Recall vs F1

| Tuned Model         | Accuracy (CV) | Precision TP/(TP+FP) | Recall TP/(TP+FN) | F1     |
|---------------------|---------------|----------------------|-------------------|--------|
| Logistic Regression | 0.6729        | 0.6664               | 0.7743            | 0.7149 |
| Neural Network      | 0.7322        | 0.7069               | 0.8414            | 0.7683 |


Tuned Neural Network is superior. 
However, further feature engineering may help improve Logistic Regression model further. 