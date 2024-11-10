#!/usr/bin/env python
# coding: utf-8

# ## Model for Predicting Stroke -  

# loading all the basic libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import xgboost as xgb
import pickle


# Parameters - 
# specifying the number of folds to be used:
n_splits = 5

output_file = f'model.bin' 


# Data Preparation -
# importing the data from csv file into Pandas DataFrame:
data = pd.read_csv('healthcare-dataset-stroke-data.csv')
data.head()


# Data Cleaning and Formatting -   
# formatting column names and row values to lower case:
data.columns = data.columns.str.lower()
data['smoking_status'] = data['smoking_status'].str.lower()
data['work_type'] = data['work_type'].str.lower()

# imputing missing values in BMI column with mean BMI values:
data['bmi'] = data['bmi'].fillna(np.mean(data['bmi']))

# dropping id column from dataset:
data.drop(columns=['id'], inplace=True)

# getting the mode of gender column:
gender_mode = list(data.gender.mode().values)[0]

# replacing the 'Other' gender category row to mode of gender column:
data['gender'] = data['gender'].replace('Other', gender_mode)


# Separating numerical variable columns and categorical variable columns:
numerical = ['age', 'avg_glucose_level', 'bmi']

# remaining columns are categorical variable columns:
categorical = ['gender','hypertension','heart_disease', 'ever_married', 'work_type', 
                        'residence_type','smoking_status']


# Splitting the Data and getting the Feature Matrix & Target variables - 
# splitting the dataset using sklearn into 60-20-20:
# Step 1 - splitting dataset into full train and test subsets first:
df_full_train, df_test = train_test_split(data, test_size=0.2,random_state=1)

# Step 2 - splitting full train subset again into training set and validation set:
df_train, df_val = train_test_split(df_full_train, test_size=0.25,random_state = 1)

# Resetting indices for each of the subset: 
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Getting our target variable column ('stroke') subsets as respective Numpy arrays:
y_train = df_train.stroke.values
y_val = df_val.stroke.values
y_test = df_test.stroke.values

# deleting 'stroke' column from feature matrix subsets:
del df_train['stroke']
del df_val['stroke']
del df_test['stroke']

df_train.shape, df_val.shape, df_test.shape


# Predicting on Test data using our Final Model (XGBoost for Classification) - 
# resetting indices of full_train DataFrame:
df_full_train = df_full_train.reset_index(drop=True)

# slicing the target variable column for full_train dataset:
y_full_train = (df_full_train.stroke).astype(int).values

# turning the full train df into dictionaries:
dicts_full_train = df_full_train.to_dict(orient='records')

# instantiating the vectorizer instance:
dv = DictVectorizer(sparse=False)

# turning list of dictionaries into full train feature matrix
X_full_train = dv.fit_transform(dicts_full_train)

# turning the test df into dictionaries:
dicts_test = df_test.to_dict(orient='records')

# turning list of dictionaries into testing feature matrix
X_test = dv.transform(dicts_test)

# converting full train and test matrices into DMatrix datastructure for using in XGBoost model:
dfulltrain = xgb.DMatrix(X_full_train, label = y_full_train, feature_names = dv.get_feature_names())
dtest = xgb.DMatrix(X_test, feature_names = dv.get_feature_names())


xgb_params = {'eta': 0.1, 
              'max_depth': 3, 
              'min_child_weight': 20,
             'objective': 'binary:logistic',
              'eval_metric':'auc',
              
             'nthread': 8,
             'seed': 1,
             'verbosity': 1}


# training our best model XGBoost on our full train set:
model = xgb.train(xgb_params, dfulltrain, num_boost_round=200) 

# predicting the XGBoost model on the testing set:
y_pred = model.predict(dtest)

# computing the AUC score on testing set:
print('AUC on test set: %.3f' % roc_auc_score(y_test, y_pred))


# Training -
## Using KFold Cross-Validation on our Final Model for making Predictions - 

# Step 1 -
# Function 1 - Creating a function to train our DataFrame:
def train(df_train, y_train):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    # converting full train and test matrices into DMatrix datastructure for using in XGBoost model:
    dtrain = xgb.DMatrix(X_train, label = y_train, feature_names = dv.get_feature_names())
    model = xgb.train(xgb_params,dtrain, num_boost_round=200)     
    
    return dv, model


# Step 2 - 
# Function 2 - Creating another function to predict:
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')  # converts df to list of dictionaries
    
    X = dv.transform(dicts)  # creates a feature matrix using the vectorizer
    
    X_Dmat = xgb.DMatrix(X, feature_names = dv.get_feature_names())
    y_pred = model.predict(X_Dmat)  # uses the model
    
    return y_pred


# Validation -
print(f'doing validation')

# Performing K-fold Cross validation and evaluating the AUC scores after each iteration:
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold = 0
        
for train_idx, val_idx in kfold.split(df_full_train):
        
    # Selecting part of dataset as 3 subsets for model:
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.stroke.values   # our target variable values as Numpy array for train and validation sets
    y_val = df_val.stroke.values

    dv, model = train(df_train, y_train)   # using train function created
    y_pred = predict(df_val, dv, model)   # using predict function created

    # compute auc scores for each iteration or fold in KFold:
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold= fold + 1

        
# Computing mean of AUC scores and spread of AUC score:
print('validation results:')
print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))



# Training the Final Model -
print('training the final model')

# Now, Training our Final Model on Full train dataset and evaluating on test dataset -
dv, model = train(df_full_train, df_full_train.stroke.values)   # using train function created
y_pred = predict(df_test, dv, model)   # using predict function created

# compute auc for ROC Curve:
auc = roc_auc_score(y_test, y_pred)
print(f'auc={auc}')



# Saving the Model -
# Step 1 - taking our model and writing it to a file - 
# creating a file where we'll write it:
output_file


# write a Binary file using pickle - alternative to open and close codes we use with open to automatically open-close a file:
with open(output_file, 'wb') as f_out:    # file output
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')





