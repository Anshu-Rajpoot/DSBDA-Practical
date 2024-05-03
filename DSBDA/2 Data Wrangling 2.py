import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Create a sample dataset
data = {
    'Student_ID': [101, 102, 103, 104, 105],
    'Age': [20, 21, 19, 20, 22],
    'Math_Score': [85, 90, 75, 95, 70],
    'Science_Score': [78, 85, 80, 88, 72],
    'English_Score': [80, 82, 78, np.nan, 85]
}

# Step 1: Handling missing values and inconsistencies
df = pd.DataFrame(data)
df['English_Score'].fillna(df['English_Score'].mean(), inplace=True)

# Step 2: Handling outliers
numeric_columns = df.select_dtypes(include=np.number).columns
for column in numeric_columns:
    z_scores = np.abs(stats.zscore(df[column]))
    outliers = (z_scores > 3)
    df[column][outliers] = df[column].mean() + 3 * df[column].std()

# Step 3: Data transformation
# Applying a logarithmic transformation to 'Math_Score' to change the scale
df['Transformed_Math_Score'] = np.log(df['Math_Score'])

# Visualize the distribution before and after transformation
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='Math_Score', kde=True)
plt.title('Before Transformation')

plt.subplot(1, 2, 2)
sns.histplot(data=df, x='Transformed_Math_Score', kde=True)
plt.title('After Transformation')

plt.tight_layout()
plt.show()

# Display the modified dataset
print(df)
