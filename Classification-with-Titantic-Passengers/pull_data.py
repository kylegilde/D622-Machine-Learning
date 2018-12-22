#!/usr/bin/env/python3
"""
Created on Sept 15 2018
@author: Kyle Gilde
"""

# Training dataset url: https://www.kaggle.com/c/titanic/download/train.csv
# Scoring dataset url: https://www.kaggle.com/c/titanic/download/test.csv
import pandas as pd

# Initiate constants & function
GITHUB_URL = 'https://raw.githubusercontent.com/kylegilde/D622-Machine-Learning/master/titanic-data/'
FILE_NAMES = ['train.csv', 'test.csv']


def validate_df(df):
    """
    Validates that the object is a dataframe and is not an empty dataframe
    Sources: https://stackoverflow.com/questions/14808945/check-if-variable-is-dataframe
    https://pandas.pydata.org/pandas-docs/version/0.18/generated/pandas.DataFrame.empty.html
    """
    if isinstance(df, pd.DataFrame) and not df.empty:
        print('The dataframe loaded correctly!')
    else:
        print('The dataframe did NOT load correctly!')

# Read csvs
try:
    train_df, test_df = pd.read_csv(GITHUB_URL + FILE_NAMES[0]), pd.read_csv(GITHUB_URL + FILE_NAMES[1])
except Exception as e:
    print(e)
    print('Github file not read')
else:
    # Validate dataframes
    validate_df(train_df)
    validate_df(test_df)

# Create local files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print('train.csv & test.csv should now be found in the working directory' )
