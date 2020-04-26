# Polynomial Regression

# Importing the libraries
# Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Importando os dados
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear Regression to the Dataset (for comparisson)
# Ajustando os dados para regressão linear (para comparação)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the Dataset
# Ajustando os dados para regressão polinomial
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
# Visualizando os resultados da Regressão Linear
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position leve')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
# Visualizando os resultados da Regressão Polinomial
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position leve')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
# Predizendo os resultados com Regressão Linear
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
# Predizendo os resultados com Regressão polinomial
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))