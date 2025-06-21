# Step 1: Install necessary packages (if not already installed)
# !pip install scikit-learn

# Step 2: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Step 3: Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 4: Initialize the model
model = DecisionTreeClassifier()

# Step 5: Define k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Step 6: Evaluate the model using cross-validation
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# Step 7: Display results
print("Cross-Validation Accuracy Scores for Each Fold:", scores)
print("Mean Accuracy:", np.mean(scores))
print("Standard Deviation:", np.std(scores))