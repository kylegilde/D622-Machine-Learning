#!/usr/bin/env/python3
"""
Created on Sept 16 2018
@author: Kyle Gilde
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline
import pickle

try:
    train_df, test_df = pd.read_csv('train.csv'), pd.read_csv('test.csv')
except Exception as e:
    print(e)
    print('Failed to load train and test CSVs')
else:
    # Combine data for munging
    train_rows = train_df.shape[0]
    all_data = pd.concat([train_df, test_df], sort=False)

    # Data Munge
    all_data.Pclass = all_data.Pclass.astype(str) # Pclass should be treated as a categorical variable
    all_data['Deck'] = (all_data.Cabin.str.slice(0, 1) # Create Deck variable
                        .fillna('Unk')) # replace NaNs with 'Unk'
    all_data['Title'] = all_data.Name.str.extract(r'^.*?, (.*?)\..*$') #extract title from name; Source: https://www.kaggle.com/gzw8210/predicting-survival-on-the-titanic
    all_data['Embarked'] = all_data.Embarked.fillna('S') # Replace NaNs with the mode value
    all_data = all_data.drop(['PassengerId', 'Cabin', 'Ticket', 'Name'], axis=1) # drop unneeded variables
    all_data = pd.get_dummies(all_data)

    #figure out which dummy levels to drop
    pd.set_option('display.max_columns', 50)
    all_data.describe(include='all')
    # drop dummy levels
    all_data = all_data.drop(['Pclass_3', 'Sex_female', 'Embarked_S',
                              'Deck_A', 'Title_Mr'], axis=1)

    # split back into training test sets
    train_df, test_df = all_data[ :train_rows], all_data[train_rows: ]
    X_test = test_df.drop('Survived', axis=1)
    # create holdout set to test accuracy
    X, y = train_df.drop('Survived', axis=1), train_df.Survived
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3,
                                                              random_state=42, stratify=y)

    # Create the ML Pipeline
    steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
             ('scaler', StandardScaler()),
             ('knn', KNeighborsClassifier(n_neighbors=7))]

    # Instantiate pipeline
    pipeline = Pipeline(steps)
    # fit model pipeline
    pipeline.fit(X_train, y_train)

    # Save test & holdout sets
    y_holdout_df = pd.DataFrame()
    y_holdout_df['y_holdout'] = y_holdout
    file_outputs = ['X_test', 'y_holdout_df', 'X_holdout']
    for file_output in file_outputs:
        eval(file_output).to_csv('%s.csv' % file_output, index=False)

    # Pickle model for another day
    model_filename = 'knn_pipeline.pkl'
    with open(model_filename, 'wb') as f: #source: https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
        pickle.dump(pipeline, f)

    file_outputs.append(model_filename)
    print('The following files should now be found be in the working directory: ' + ', '.join(file_outputs))
    # Source: https://stackoverflow.com/questions/5618878/how-to-convert-list-to-string

