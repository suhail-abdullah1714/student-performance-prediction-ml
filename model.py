import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv("data/student_data.csv")

X = data[['Hours_Studied', 'Attendance', 'Previous_Score']]
y = data['Final_Score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
score = r2_score(y_test, predictions)
print("Model R2 Score:", score)
