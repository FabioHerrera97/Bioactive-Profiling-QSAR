# import the required libraries

import numpy as np
from numpy import unique
import pandas as pd
from numpy import arange
from pandas import read_csv

# Upload the file with the raw data

file = 'path of the file with the data'
data = pd.read_excel(file, sheet_name='name of the worksheet where the data are')
data

# Take a look at the dimensionality of the dataset

print(data.shape)

# Find columns with unique values

counts = data.nunique()
to_del = [i for i,v in enumerate(counts) if v == 1]
print(to_del)
print(len(to_del))

# Drop columns with unique values

data_filtered_1 = data.drop(data.columns[to_del], axis=1)
data_filtered_1
print(data_filtered_1.shape)

# Perform a correlation analysis 

cor_matrix = data_filtered_1.corr().abs()
cor_matrix
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
print(upper_tri)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
print(); print(to_drop)
print(len(to_drop))

# Drop the columns with a correlation higher than 0.9
# Check manually the dropping columns to avoid deleting critical information 

data_cleaned = data_filtered_1[data_filtered_1.columns.difference(to_drop)]
data_cleaned

# Save the data in xlsx format

data_cleaned.to_excel('data_cleaned.xlsx')


