import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Sample data similar to Titanic dataset
data = {
    'survived': [0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    'pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
    'sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female'],
    'age': [22, 38, 26, 35, 35, None, 54, 2, 27, 14],
    'fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708]
}

# Convert sample data to DataFrame
titanic = pd.DataFrame(data)

# Display the first few rows of the dataset
print(titanic.head())

# Plotting histogram to visualize the distribution of ticket prices
plt.figure(figsize=(8, 6))
sns.histplot(data=titanic, x='fare', bins=30, kde=True)
plt.title('Distribution of Ticket Prices')
plt.xlabel('Ticket Fare')
plt.ylabel('Frequency')
plt.show()
