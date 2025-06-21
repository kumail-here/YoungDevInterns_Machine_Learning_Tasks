# Step 1: Install necessary packages (if not already installed)
# !pip install scikit-learn matplotlib

# Step 2: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 3: Create sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Features must be 2D
y = np.array([2, 4, 5, 4, 5])                # Target values

# Step 4: Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Step 5: Make predictions
y_pred = model.predict(X)

# Step 6: Print slope and intercept
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Step 7: Visualize the regression line
plt.scatter(X, y, color='blue', label='Actual Points')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
