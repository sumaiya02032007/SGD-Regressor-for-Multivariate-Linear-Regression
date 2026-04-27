# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and select features.

2.Split data into training and testing sets.

3.Scale the input features.

4.Train the multi-output regression model.

5.Predict outputs and calculate error.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: s.sumaiya
RegisterNumber:  212225040437
*/
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

data = fetch_california_housing()

x = data.data[:, :3]
y = np.c_[data.target, data.data[:, 6]]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
y_train = scaler.fit_transform(y_train)

model = MultiOutputRegressor(SGDRegressor(random_state = 42))
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)

print(f"Mean Squared Error (MSE) : {mse}")
print("Sample Prediction (House Price,Population) :",)
print(y_pred[:5])

```

## Output:
<img width="749" height="231" alt="Screenshot 2026-04-27 155534" src="https://github.com/user-attachments/assets/42354c72-e3da-4f11-9401-66eba4dce09c" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
