import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Dummy Titanic Dataset Create Karna (For Task Submission)
np.random.seed(42)
data_size = 200
data = {
    'Pclass': np.random.choice([1, 2, 3], size=data_size, p=[0.24, 0.21, 0.55]),
    'Sex': np.random.choice([0, 1], size=data_size, p=[0.65, 0.35]),  # 0 for Male, 1 for Female
    'Age': np.random.randint(1, 80, size=data_size),
    'SibSp': np.random.choice([0, 1, 2, 3], size=data_size, p=[0.70, 0.23, 0.05, 0.02]),
    'Parch': np.random.choice([0, 1, 2], size=data_size, p=[0.78, 0.15, 0.07]),
    'Fare': np.random.uniform(10, 150, size=data_size),
    'Survived': np.random.choice([0, 1], size=data_size, p=[0.62, 0.38])
}
df = pd.DataFrame(data)

# 2. Features aur Target ko alag karna
X = df.drop('Survived', axis=1)
y = df['Survived']

# 3. Train aur Test split karna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, random_state=42)

# 4. Model Train karna (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Predict aur Evaluate karna
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
