import pandas as pd
import matplotlib.pyplot as plt

#Load the dataset
df = pd.read_excel("embeddingsdata.xlsx")

#Loading 2 features having numeric values
feature_a = df['embed_0']
feature_b = df['embed_1']

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(feature_a, feature_b, color='blue', alpha=0.5)

# Add labels and title
plt.xlabel('embed_0')
plt.ylabel('embed_1')
plt.title('Scatter Plot of embed_0 vs embed_1')

# Show plot
plt.grid(True)
plt.show()
