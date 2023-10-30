from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel("embeddingsdata.xlsx")

# Assume 'embed_1' is the independent variable
independent_variable = df['embed_1']
dependent_variable = df['embed_0']

# Reshape the data as sklearn's LinearRegression model expects 2D array
independent_variable = independent_variable.values.reshape(-1, 1)
dependent_variable = dependent_variable.values.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()
model.fit(independent_variable, dependent_variable)

# Predict the values
predicted_values = model.predict(independent_variable)

# Calculate mean squared error
mse = mean_squared_error(dependent_variable, predicted_values)
print(f'Mean Squared Error: {mse}')

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(independent_variable, dependent_variable, color='blue', alpha=0.5)

# Plot the regression line
plt.plot(independent_variable, predicted_values, color='red')

# Add labels and title
plt.xlabel('embed_0')
plt.ylabel('embed_1')
plt.title('Linear Regression Model: embed_0 vs embed_1')

# Show plot
plt.grid(True)
plt.show()
