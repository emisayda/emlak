import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ev_fiyat.csv")

# Log-transform the target variable if it has large values
df['fiyat'] = np.log1p(df['fiyat'])  # log1p is log(1+x), handles zero values

X = df.drop(columns="fiyat")
y = df['fiyat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

#test_loss = mean_squared_error(y_test, y_pred)
#print(f'Test Loss (MSE): {test_loss:.4f}')

from sklearn.metrics import mean_absolute_percentage_error

test_loss = mean_absolute_percentage_error(y_test, y_pred)
print(f'Test Loss (MAPE): {test_loss:.4f}')
'''
plt.scatter(np.expm1(y_test), np.expm1(y_pred)) 
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
'''