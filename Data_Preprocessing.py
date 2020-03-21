# Loading Dataset
import numpy as np
import pandas as pd
dataset = pd.read_csv('/content/drive/My Drive/Data.csv',error_bad_lines=False)
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values

# Handling Missing Data
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer()
imputer.fit(X[:,1:3 ])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])

# Creating Dummy Variables
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X = ct.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
