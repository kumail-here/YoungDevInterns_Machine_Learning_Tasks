# Step 1: Install required libraries (if not already installed)
# !pip install scikit-learn pandas numpy

# Step 2: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 3: Create a sample dataset with missing values and categorical features
data = {
    'age': [25, 30, np.nan, 40, 35],
    'salary': [50000, 60000, 52000, np.nan, 58000],
    'department': ['IT', 'HR', 'IT', 'Finance', np.nan],
    'target': [0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Step 4: Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Step 5: Identify numerical and categorical columns
num_cols = ['age', 'salary']
cat_cols = ['department']

# Step 6: Define preprocessing for numerical data (impute + scale)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Step 7: Define preprocessing for categorical data (impute + encode)
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])

# Step 8: Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Step 9: Apply transformations
X_processed = preprocessor.fit_transform(X)

# Step 10: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Step 11: Show results
print("Processed Feature Matrix (Training Set):\n", X_train.toarray() if hasattr(X_train, 'toarray') else X_train)
print("\nTarget Values (Training Set):", y_train.values)