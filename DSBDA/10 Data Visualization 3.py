import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset from the UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris = pd.read_csv(url, names=iris_columns)

# 1. List down the features and their types
features_types = iris.dtypes
print("Features and their types:")
print(features_types)

# 2. Create a histogram for each feature
iris.hist(figsize=(10, 6), bins=20)
plt.suptitle("Histograms of Iris Dataset Features", y=1.02)
plt.tight_layout()
plt.show()

# 3. Create a boxplot for each feature
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris)
plt.title("Boxplots of Iris Dataset Features")
plt.xticks(rotation=45)
plt.show()

# 4. Compare distributions and identify outliers
# Outliers can be identified from the boxplot where points beyond the whiskers are considered outliers.
