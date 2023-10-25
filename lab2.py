import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('temperatures.csv')
df.head()

#input data
x = df['YEAR']

#output data
y = df['ANNUAL']

plt.title('Temperature of INDIA')
plt.xlabel('Year')
plt.ylabel('Annual Average Temperature')
plt.scatter(x,y)

x.shape

x = x.values

x = x.reshape(117,1)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x,y)

regressor.coef_

regressor.intercept_

regressor.predict([[2024]])

predicted = regressor.predict(x)

import numpy as np

#Mean Absolute Error
np.mean(abs(y-predicted))

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y,predicted)

np.mean((y-predicted)**2)

from sklearn.metrics import mean_squared_error
mean_squared_error(y,predicted)

from sklean.metrics import r2_score
r2_score(y,predicted)

############

plt.title('Temperature plot of INDIA')
plt.xlabel('Year')
plt.ylabel('Annual Average Temperature')
plt.scatter(x,y,label='actual',color ='r',marker='.')
plt.plot(x,predicted, label='predicted',color='g')
plt.legend()