import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ev_fiyat.csv")


df['fiyat'] = np.log1p(df['fiyat'])  

X = df.drop(columns="fiyat")
y = df['fiyat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)



from sklearn.metrics import mean_absolute_percentage_error

test_loss = mean_absolute_percentage_error(y_test, y_pred)
print(f'Test Loss (MAPE): {test_loss:.4f}')

import joblib
joblib.dump(model, "random_forest_model_compressed.pkl", compress=3)
joblib.dump(scaler, "scaler.pkl")

