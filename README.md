### NAME: SURYA P <br>
### REG NO: 212224230280

# IMPLEMENTATION OF SIMPLE LINEAR REGRESSION MODEL FOR PREDICTING THE MARKS SCORED 

## AIM :

To write a program to predict the marks scored by a student using the simple linear regression model.

## EQUIPMENTS REQUIRED :

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM : 

1. Load the dataset into a DataFrame and explore its contents to understand the data structure.
2. Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.
3. Create a linear regression model and fit it using the training data.
4. Predict the results for the testing set and plot the training and testing sets with fitted lines.
5. Calculate error metrics (MSE, MAE, RMSE) to evaluate the model’s performance.

## PROGRAM : 
```
# Program to implement the simple linear regression model for predicting the marks scored.

# Developed by: Surya P
# RegisterNumber: 212224230280


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student_scores.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("Last 5 rows of the dataset:")
print(df.tail())

X = df.iloc[:, :-1].values  # Assuming the 'Hours' column is the first column
Y = df.iloc[:, 1].values    # Assuming the 'Scores' column is the second column

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

print("Predicted values:")
print(Y_pred)
print("Actual values:")
print(Y_test)

plt.scatter(X_train, Y_train, color="red", label="Actual Scores")
plt.plot(X_train, regressor.predict(X_train), color="blue", label="Fitted Line")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.show()

plt.scatter(X_test, Y_test, color='green', label="Actual Scores")
plt.plot(X_train, regressor.predict(X_train), color='red', label="Fitted Line")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.show()

mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print('Mean Squared Error (MSE) =', mse)
print('Mean Absolute Error (MAE) =', mae)
print('Root Mean Squared Error (RMSE) =', rmse)
```

## OUTPUT : 

![image](https://github.com/user-attachments/assets/616f3cdb-f307-44fe-b08f-954ba7fddea2)

![image](https://github.com/user-attachments/assets/9592205e-d09f-4935-ad22-49dcc70ccd32)

![image](https://github.com/user-attachments/assets/faf751d1-7d73-4ac2-83fe-a2974f76db41)

![image](https://github.com/user-attachments/assets/bb1f41e2-1a5a-480e-b9ab-14a0cd44db28)

## RESULT : 

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
