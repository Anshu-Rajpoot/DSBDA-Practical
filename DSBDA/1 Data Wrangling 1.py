import pandas as pd
import numpy as np
import requests
from io import StringIO

# URL of the dataset
url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"

# Fetching data from URL
response = requests.get(url)

# Converting data into DataFrame
df = pd.read_csv(StringIO(response.text))

# Displaying the first few rows of the dataframe
print("First few rows of the dataframe:")
print(df.head())

# Checking for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Getting initial statistics
print("\nInitial statistics of the dataset:")
print(df.describe())

# Variable descriptions
print("\nVariable descriptions:")
print("The dataset contains information about iris flowers with the following variables:")
print("1. Sepal length (in cm)")
print("2. Sepal width (in cm)")
print("3. Petal length (in cm)")
print("4. Petal width (in cm)")
print("5. Species (categorical variable indicating the species of iris)")

# Dimensions of the dataframe
print("\nDimensions of the dataframe:")
print(df.shape)

# Summarizing types of variables
print("\nTypes of variables:")
print(df.dtypes)

# No need for further conversion in this dataset as 'Species' is already categorical
