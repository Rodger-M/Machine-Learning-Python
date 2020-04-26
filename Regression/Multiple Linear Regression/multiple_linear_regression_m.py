# Multiple Linear Regression

# Importing the libraries
# Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Importando os dados
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Codificando dados categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap - Skipping one encoded column
# Evitando "Dummy Variable Trap" - Pulando uma coluna codificada
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
# Dividindo os dados em conjunto de treino e teste
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (We don't use this time)
# Recurso de escalonamento (Não usamos desta vez)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training Set
# Ajustando a Regressão Linear Simples ao conjunto de treino
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
# Predizendo os resultados do conjunto de teste
y_pred = regressor.predict(X_test)