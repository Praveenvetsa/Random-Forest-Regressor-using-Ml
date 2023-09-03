# Random Forest Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\fsds materials\fsds\3. Aug\22nd\EMP SAL.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor()
#regressor = RandomForestRegressor(n_estimators=50,criterion='absolute_error')
#regressor = RandomForestRegressor(n_estimators=50)
regressor = RandomForestRegressor(criterion='poisson', random_state=0)
regressor.fit(X,y)

y_pred = regressor.predict([[6.5]])
y_pred


%matplotlib inline
plt.scatter(X,y,color = 'red')
plt.plot(X,regressor.predict(X),color ='blue')
plt.title('Truth or Bluff (Decision tree regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X,regressor.predict(X),color ='blue')
plt.title('Truth or Bluff (Decision tree regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()