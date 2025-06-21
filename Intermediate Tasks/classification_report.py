# Step 1: Install required packages (if not already installed)
# !pip install scikit-learn

# Step 2: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 3: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a classification model (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Generate and display the classification report
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("Classification Report:\n")
print(report)