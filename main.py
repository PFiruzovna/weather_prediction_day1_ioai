import pandas as pd
import numpy as np

df = pd.read_csv("weather.csv")
df["date"] = pd.to_datetime(df["date"])

df["yesterday"] = df["temperature"].shift(1)

df["3days_before"] = df["temperature"].rolling(window=3).mean()

df.dropna(inplace=True)

X1 = df["yesterday"].to_numpy()
X2 = df["3days_before"].to_numpy()
X = np.column_stack((X1, X2))
y = df["temperature"].to_numpy()
X_mat = np.column_stack((np.ones(len(X)), X)) 
w = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y
y_pred = X_mat @ w

rmse = np.sqrt(np.mean((y - y_pred)**2))
print("xatolik:", rmse)

for i in range(5):
    print(f"{i+1}-day: Real = {round(y[i], 2)}°C | Prediction = {round(y_pred[i], 2)}°C")
