import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Load the dataset
df = pd.read_excel("embeddingsdata.xlsx")

# Assuming X_train, y_train, X_test, y_test are your training and test sets

# Initialize the Logistic Regression model
logistic_model = LogisticRegression()

binary_dataframe = df[df['Label'].isin([0, 1])]
X = binary_dataframe[['embed_1', 'embed_2']]
y = binary_dataframe['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Logistic Regression model on the training data
logistic_model.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
predictions = logistic_model.predict(X_test)

# Calculate accuracy by comparing predicted labels to actual labels in the test set
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of Logistic Regression on the test set: {accuracy * 100:.2f}%")
