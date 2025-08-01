import pandas as pd 
import numpy as np

df = pd.read_csv('HR-Employee-Attrition.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Display the shape of the DataFrame
print(df.shape)

# Display the data types of each column
print(df.dtypes)

# Display the summary statistics of the DataFrame
print(df.describe())

# Check for missing values in the DataFrame
print(df.isnull().sum())

# Check for duplicate rows in the DataFrame
print(df.duplicated().sum())

# drop the 'EmployeeCount' and 'Over18' and 'StandardHours' columns as they are not useful for analysis
df.drop(['EmployeeCount', 'Over18','StandardHours'], axis=1, inplace=True)

# Convert 'Attrition' column to numerical values
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

