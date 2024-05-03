import seaborn as sns

# Load the Iris dataset from seaborn
iris_df = sns.load_dataset("iris")

# 1. Summary statistics grouped by species
summary_stats = iris_df.groupby('species').describe()

# 2. Basic statistical details for each species
setosa_stats = iris_df[iris_df['species'] == 'setosa'].describe()
versicolor_stats = iris_df[iris_df['species'] == 'versicolor'].describe()
virginica_stats = iris_df[iris_df['species'] == 'virginica'].describe()

print("Summary Statistics Grouped by Species:")
print(summary_stats)

print("\nBasic Statistical Details for Iris-setosa:")
print(setosa_stats)

print("\nBasic Statistical Details for Iris-versicolor:")
print(versicolor_stats)

print("\nBasic Statistical Details for Iris-virginica:")
print(virginica_stats)
