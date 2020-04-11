# Reading data from the file
import numpy as np
import pandas as pd
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values
print(X)

# Encoding the lables
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
print(X)

# Creating Dummy Variables
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   # The column numbers to be transformed (here is [3] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X = ct.fit_transform(X)
print(X)

# Spliting data into train & test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Fitting the regresson model
from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()
regression_model.fit(X_train, Y_train)

# Predicting Values
predictions = regression_model.predict(X_test)

# Comparison
print(Y_test)
print(predictions)

# Doing Backpropagation
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regression_model_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regression_model_OLS.summary()

# Removing non significant features by comparing to the p values
X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
regression_model_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regression_model_OLS.summary()
X_opt = np.array(X[:, [0, 3, 4, 5]], dtype=float)
regression_model_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regression_model_OLS.summary()
X_opt = np.array(X[:, [0, 3, 5]], dtype=float)
regression_model_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regression_model_OLS.summary()

# Fitting regression model
regression_model2 = LinearRegression()
regression_model2.fit(X_train[:,[0,3,5]], Y_train)
predictions2 = regression_model2.predict(X_test[:, [0, 3, 5]])

# Comparisons
print(Y_train)
print(predictions)
print(predictions2)
predictions3
