import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_california_housing 
data = fetch_california_housing() 
df = pd.DataFrame(data.data, columns=data.feature_names) 
df['PRICE'] = data.target 
print("Dataset Loaded Successfully!") 
print(df.head()) 
X = df.drop('PRICE', axis=1) 
y = df['PRICE'] 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 
from sklearn.linear_model import LinearRegression 
lr = LinearRegression() 
lr.fit(X_train, y_train) 
y_pred_lr = lr.predict(X_test) 
from sklearn.metrics import mean_squared_error 
print("\nLinear Regression RMSE:", 
np.sqrt(mean_squared_error(y_test, y_pred_lr))) 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
model = Sequential() 
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1])) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(32, activation='relu')) 
model.add(Dense(1)) 
model.compile(optimizer='adam', 
loss='mean_squared_error', 
metrics=['mae']) 
model.fit(X_train, y_train, epochs=50, validation_split=0.1) 
loss, mae = model.evaluate(X_test, y_test) 
print("\nNeural Network MAE:", mae) 
print("\nEnter values for prediction:") 
user_input = [] 
for col in df.columns[:-1]: 
val = float(input(f"Enter value for {col}: ")) 
user_input.append(val) 
user_input = np.array(user_input).reshape(1, -1) 
user_input = sc.transform(user_input) 
prediction = model.predict(user_input) 
print("\nPredicted House Price:", prediction[0][0]) 
