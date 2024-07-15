# Telco_Customer_Churn_Prediction_With_Python
Predicting customer churn using the Telco Customer Churn dataset with an XGBoost classifier. This project involves data preprocessing, feature engineering, and hyperparameter tuning to build an effective model for identifying at-risk customers.

### Import Libraries

First, I imported the necessary libraries for data manipulation, model training, and evaluation.

```python
import pandas as pd 
import numpy as np 
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
```

### Load and Preview Data

Next, I loaded the dataset and previewed its structure to understand the data I was working with.

```python
df = pd.read_csv('Telco_customer_churn.csv')
df.head()
```

### Data Cleaning

I dropped columns that wouldn't be useful for the prediction model to simplify the data and remove noise.

```python
df.drop(['Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'], axis=1, inplace=True)
df.head()
```

### Inspecting Unique Values

I checked the unique values of some columns to understand their content, helping in identifying categorical variables and understanding data distribution.

```python
print(df['Count'].unique())
print(df['Country'].unique())
print(df['State'].unique())
print(df['City'].unique())
```

### Further Column Dropping

I dropped additional columns that were not necessary for the analysis, such as IDs or columns with constant values.

```python
df.drop(['CustomerID', 'Count', 'Country', 'State', 'Lat Long'], axis=1, inplace=True)
```

### Data Preprocessing

I replaced spaces in city names with underscores for consistency, which is important for later encoding steps.

```python
df['City'].replace(' ', '_', regex=True, inplace=True)
df.head()
```

I also renamed columns to have underscores instead of spaces for easier handling.

```python
df.columns = df.columns.str.replace(' ', '_')
df.head()
```

I then checked the data types of columns to ensure they were as expected, identifying if any column needed type conversion.

```python
df.dtypes
```

I inspected unique values for specific columns to check for inconsistencies or unexpected values.

```python
print(df['Phone_Service'].unique())
print(df['Total_Charges'].unique())
```

I converted 'Total_Charges' to numeric, handling any non-numeric entries by converting them to NaN and then filling them with 0.

```python
df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')
df['Total_Charges'].fillna(0, inplace=True)
```

Finally, I replaced any remaining spaces with underscores to ensure all categorical variables were consistent.

```python
df.replace(' ', '_', regex=True, inplace=True)
df.head()
```

### Splitting Data

I defined the feature matrix `X` and target vector `y`. The feature matrix contains all the predictor variables, and the target vector contains the response variable.

```python
X = df.drop('Churn_Value', axis=1).copy()
y = df['Churn_Value'].copy()
print(X.head())
print(y.head())
```

### Encoding Categorical Variables

I converted categorical variables into dummy/indicator variables using one-hot encoding. This converts categorical data into a format that can be provided to machine learning algorithms for better prediction.

```python
X_encoded = pd.get_dummies(X, columns=['City', 'Gender', 'Senior_Citizen', 'Partner', 'Dependents', 'Phone_Service', 'Multiple_Lines', 'Internet_Service', 'Online_Backup', 'Online_Security', 'Device_Protection', 'Tech_Support', 'Streaming_TV', 'Streaming_Movies', 'Contract', 'Paperless_Billing', 'Payment_Method'])
X_encoded.head()
```

### Target Variable Analysis

I checked the unique values in the target variable to understand the classification problem.

```python
print(y.unique())
```

I also calculated the class imbalance to understand if the dataset was balanced or if one class was more prevalent than the other, which can affect model performance.

```python
sum(y) / len(y)
```

### Train-Test Split

I split the data into training and testing sets, using stratified sampling to ensure the class distribution in the splits was the same as the original dataset.

```python
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)
```

### Model Training with XGBoost

I trained the XGBoost classifier with initial parameters. XGBoost is an efficient and scalable implementation of the gradient boosting framework by Friedman.

```python
clf_xgb = xgb.XGBClassifier(objective='binary:logistic', eval_metric='aucpr', early_stopping_rounds=10, seed=42)
clf_xgb.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])
```
    [0]	validation_0-aucpr:0.63785
    [1]	validation_0-aucpr:0.64982
    [2]	validation_0-aucpr:0.65387
    [3]	validation_0-aucpr:0.65349
    [4]	validation_0-aucpr:0.65982
    [5]	validation_0-aucpr:0.65564
    [6]	validation_0-aucpr:0.65947
    [7]	validation_0-aucpr:0.65758
    [8]	validation_0-aucpr:0.65258
    [9]	validation_0-aucpr:0.65392
    [10]validation_0-aucpr:0.65230
    [11]validation_0-aucpr:0.65334
    [12]validation_0-aucpr:0.65225
    [13]validation_0-aucpr:0.65724
    [14]validation_0-aucpr:0.65439
    

### Model Evaluation

I displayed the confusion matrix for the classifier, which is a useful tool for analyzing how well the classifier is performing.

```python
ConfusionMatrixDisplay.from_estimator(clf_xgb, X_test, y_test, display_labels=["Did not leave", "Left"])
```
					Predicted Values
	
				Did_not_leave		Left
Actual Value	Did_not_leave	1179			115
		Left		241			226

the accuracy of the confusion matrix is 0.798 .

### Hyperparameter Tuning

Hyperparameter tuning is a crucial step in improving model performance. It involves selecting the best set of hyperparameters for a learning algorithm.

#### Round 1

I performed a grid search to find the optimal hyperparameters. GridSearchCV exhaustively considers all parameter combinations to find the best one.

```python
param_grid = {'max_depth': [3, 4, 5], 'learning_rate': [0.1, 0.5, 1], 'gamma': [0, 0.25, 1.0], 'reg_lambda': [0, 1.0, 10.0], 'scale_pos_weight': [1, 3, 5]}

optimal_params = GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic', seed=42, subsample=0.9, early_stopping_rounds=10, eval_metric='auc', colsample_bytree=0.5), param_grid=param_grid, scoring='roc_auc', verbose=0, n_jobs=10, cv=3)

optimal_params.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(optimal_params.best_params_)
```
{'gamma': 0, 'learning_rate': 0.1, 'max_depth': 4, 'reg_lambda': 10.0, 'scale_pos_weight': 3}

#### Round 2

I refined the search with a narrower range of parameters to hone in on the best parameters.

```python
param_grid = {'max_depth': [4], 'learning_rate': [0.1, 0.5, 1], 'gamma': [0.25], 'reg_lambda': [10.0, 20, 100], 'scale_pos_weight': [3]}

optimal_params = GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic', seed=42, subsample=0.9, early_stopping_rounds=10, eval_metric='auc', colsample_bytree=0.5), param_grid=param_grid, scoring='roc_auc', verbose=0, n_jobs=10, cv=3)

optimal_params.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(optimal_params.best_params_)
```
{'gamma': 0.25, 'learning_rate': 0.1, 'max_depth': 4, 'reg_lambda': 20, 'scale_pos_weight': 3}

### Final Model Training

I trained the final model with the best parameters found to ensure that the model was using the most optimal settings.

```python
clf_xgb = xgb.XGBClassifier(seed=42, objective='binary:logistic', gamma=0.25, learning_rate=0.1, max_depth=4, reg_lambda=10, scale_pos_weight=3, subsample=0.9, colsample_bytree=0.5

, eval_metric='aucpr', early_stopping_rounds=10)
clf_xgb.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])
```
    [0]	validation_0-aucpr:0.54553
    [1]	validation_0-aucpr:0.56880
    [2]	validation_0-aucpr:0.62032
    [3]	validation_0-aucpr:0.62974
    [4]	validation_0-aucpr:0.63486
    [5]	validation_0-aucpr:0.63784
    [6]	validation_0-aucpr:0.63663
    [7]	validation_0-aucpr:0.63713
    [8]	validation_0-aucpr:0.63883
    [9]	validation_0-aucpr:0.63841
    [10]validation_0-aucpr:0.63878
    [11]validation_0-aucpr:0.63934
    [12]validation_0-aucpr:0.65377
    [13]validation_0-aucpr:0.65006
    [14]validation_0-aucpr:0.65731
    [15]validation_0-aucpr:0.65439
    [16]validation_0-aucpr:0.65138
    [17]validation_0-aucpr:0.64968
    [18]validation_0-aucpr:0.65106
    [19]validation_0-aucpr:0.65699
    [20]validation_0-aucpr:0.65686
    [21]validation_0-aucpr:0.65957
    [22]validation_0-aucpr:0.65944
    [23]validation_0-aucpr:0.65426
    [24]validation_0-aucpr:0.65330
    [25]validation_0-aucpr:0.65394
    [26]validation_0-aucpr:0.65279
    [27]validation_0-aucpr:0.65241
    [28]validation_0-aucpr:0.65364
    [29]validation_0-aucpr:0.65406
    [30]validation_0-aucpr:0.65464
    [31]validation_0-aucpr:0.65414
    

### Final Model Evaluation

I displayed the confusion matrix for the final model to understand how well it was performing.

```python
ConfusionMatrixDisplay.from_estimator(clf_xgb, X_test, y_test, display_labels=["Did not leave", "Left"])
```
					Predicted Values
	
				Did_not_leave		Left
Actual Value	Did_not_leave	934			360
		Left		78			389

the accuracy of the confusion matrix is 0.751 .

### Feature Importance

I analyzed the importance of features in the model. Feature importance helps in understanding which features are contributing the most to the predictions.

```python
clf_xgb = xgb.XGBClassifier(seed=42, objective='binary:logistic', gamma=0.25, learning_rate=0.1, max_depth=4, reg_lambda=10, scale_pos_weight=3, subsample=0.9, colsample_bytree=0.5, n_estimators=1)
clf_xgb.fit(X_train, y_train)
bst = clf_xgb.get_booster()
for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))

node_params = {'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#78cbe'}
leaf_params = {'shape': 'box', 'style': 'filled', 'fillcolor': '#e48038'}
xgb.to_graphviz(clf_xgb, num_trees=0, size="10,10", condition_node_params=node_params, leaf_node_params=leaf_params)
```

It can be seen that after hyper-parameter tuning the XGBoost model the accuracy of the model is reducing , but it should be noted that we are concerned more about minimizing the False Negative Rates as our primary goal is to accurately identify the most at-risk customers without over-allocating retention resources to those who are not actually not going to churn.

### Conclusion

In this Telco Customer Churn Prediction project, I applied several machine learning concepts and techniques to build an effective model for predicting customer churn. Here’s a summary of the steps and key concepts used:

1. **Data Cleaning**: Dropped irrelevant columns and handled missing values to prepare the data for analysis.
2. **Data Preprocessing**: Applied transformations to ensure consistency, such as replacing spaces with underscores and converting columns to appropriate data types.
3. **Encoding Categorical Variables**: Used one-hot encoding to convert categorical variables into a format suitable for machine learning algorithms.
4. **Train-Test Split**: Split the dataset into training and testing sets using stratified sampling to maintain the class distribution.
5. **Model Training**: Used XGBoost, a powerful gradient boosting algorithm, for initial model training.
6. **Model Evaluation**: Evaluated model performance using a confusion matrix and balanced accuracy score to understand how well the model was predicting churn.
7. **Hyperparameter Tuning**: Performed grid search for hyperparameter optimization to improve model performance.
8. **Feature Importance**: Analyzed feature importance to understand which features had the most impact on predictions.

The project demonstrated the importance of careful data preprocessing and the effectiveness of XGBoost for classification tasks. Despite some trade-offs in model accuracy, the focus on minimizing false negatives was essential for accurately identifying at-risk customers. This approach ensures that retention efforts are directed towards those most likely to churn, optimizing resource allocation and improving customer retention strategies.

---
