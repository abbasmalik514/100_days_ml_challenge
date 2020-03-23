# Reading The Data from the file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('drive/My Drive/Salary_Data.csv')
X = dataset.iloc[ : ,   :-1 ].values
Y = dataset.iloc[ : , 1 ].values

# Test Train Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0)

# Fiting a regression model
from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()
regression_model = regression_model.fit(X_train,Y_train)

# Predicting Results
prediction = regression_model.predict(X_test)

# Plotting Train Results
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regression_model.predict(X_train), color ='blue')

# Plotting Test Results
plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regression_model.predict(X_test), color ='blue')
