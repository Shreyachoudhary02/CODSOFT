import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dummy Sales Dataset Create Karna (For Task Submission)
np.random.seed(42)
data_size = 200
data = {
    'TV': np.random.uniform(10, 300, size=data_size),
    'Radio': np.random.uniform(5, 50, size=data_size),
    'Newspaper': np.random.uniform(5, 100, size=data_size),
    'Sales': np.random.uniform(5, 25, size=data_size)
}
# Sales ko thoda realistic banane ke liye TV/Radio par depend kar dete hain
data['Sales'] = 0.05 * data['TV'] + 0.18 * data['Radio'] + np.random.normal(2, 1, size=data_size)

df = pd.DataFrame(data)

# 2. Features aur Target ko alag karna
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 3. Train aur Test split karna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Train karna (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict aur Evaluate karna
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2 * 100:.2f}%")
