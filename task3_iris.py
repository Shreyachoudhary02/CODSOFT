import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Dummy Iris Dataset Create Karna (For Task Submission)
np.random.seed(42)
data_size = 150
data = {
    'SepalLengthCm': np.random.uniform(4.3, 7.9, size=data_size),
    'SepalWidthCm': np.random.uniform(2.0, 4.4, size=data_size),
    'PetalLengthCm': np.random.uniform(1.0, 6.9, size=data_size),
    'PetalWidthCm': np.random.uniform(0.1, 2.5, size=data_size),
    'Species': np.random.choice(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], size=data_size)
}
df = pd.DataFrame(data)

# 2. Features aur Target ko alag karna
X = df.drop('Species', axis=1)
y = df['Species']

# 3. Train aur Test split karna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Train karna (K-Nearest Neighbors)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 5. Predict aur Evaluate karna
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
