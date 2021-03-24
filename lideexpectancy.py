import numpy as np
from numpy import nan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score,explained_variance_score,mean_squared_error
from sklearn.feature_selection import SelectFromModel



missing_value_formats = ["?","NA"]
dataset = pd.read_csv('LifeExpectancyData.csv', na_values = missing_value_formats)

y = dataset.iloc[:,2:3].values
y = y.reshape((len(y),1))
x_f = dataset.iloc[:,1:2]
x_m = dataset.iloc[:, 3:]
X = pd.concat([x_f,x_m], axis=1)
sum_1 = X.isnull().sum()


x_f = X.iloc[:,:2]
x_m = X.iloc[:, 5:6]
x_mm = X.iloc[:,7:8]
x_l = X.iloc[:,9:10]
x_ll = X.iloc[:,11:]
X = pd.concat([x_f,x_m, x_mm, x_l, x_ll], axis=1)
sum_2 = X.isnull().sum()

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

y = imputer.fit_transform(y)

imputer.fit(X.iloc[:, 1:4])
X.iloc[:, 1:4] = imputer.transform(X.iloc[:, 1:4])

imputer.fit(X.iloc[:, 7:])
X.iloc[:, 7:] = imputer.transform(X.iloc[:, 7:])
sum_3 = X.isnull().sum()

le_1 = LabelEncoder()
X.iloc[:,:1] = le_1.fit_transform(X.iloc[:,:1])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
pred_multi = regressor.predict(X_test)


poly_reg = PolynomialFeatures(degree = 8)
X_poly = poly_reg.fit_transform(X_train)

regressor_poly = LinearRegression()
regressor_poly.fit(X_poly, y_train)
pred_poly = regressor_poly.predict(poly_reg.fit_transform(X_test))



X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(X, y, test_size = 0.33, random_state = 0)
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_svr = sc_X.fit_transform(X_train_svr)
y_train_svr = sc_y.fit_transform(y_train_svr)

regressor_svr = SVR(kernel = 'rbf')
regressor_svr.fit(X_train_svr, y_train_svr)
pred_svr = sc_y.inverse_transform(regressor_svr.predict(sc_X.transform(X_test_svr)))


regressor_dt = DecisionTreeRegressor(random_state = 0)
regressor_dt.fit(X_train, y_train)
pred_dt = regressor_dt.predict(X_test)


regressor_rf = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor_rf.fit(X_train, y_train)
pred_rf = regressor_rf.predict(X_test)


r_multi = r2_score(y_test, pred_multi)
r_poly = r2_score(y_test, pred_poly)
r_svr = r2_score(y_test_svr, pred_svr)
r_dt = r2_score(y_test, pred_dt)
r_rf = r2_score(y_test, pred_rf)



from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
pred_ridge = ridge_model.predict(X_test)
r_ridge = r2_score(y_test, pred_ridge)
RIDGE = mean_squared_error(y_test,pred_ridge)**(0.5)

lasso_model = Lasso(alpha=0.00000001)
lasso_model.fit(X_train, y_train)
pred_lasso = lasso_model.predict(X_test)
r_lasso = r2_score(y_test, pred_lasso)
LASSO = mean_squared_error(y_test,pred_lasso)**(0.5)


MLR = mean_squared_error(y_test,pred_multi)**(0.5)
PR = mean_squared_error(y_test,pred_poly)**(0.5)
SVR = mean_squared_error(y_test_svr,pred_svr)**(0.5)
DT = mean_squared_error(y_test,pred_dt)**(0.5)
RF = mean_squared_error(y_test,pred_rf)**(0.5)


y = pd.DataFrame(y)
new_data = pd.concat([y,X],axis=1)
plt.figure(figsize=(12,10))
cor = new_data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


sel = SelectFromModel(regressor_rf, threshold=0.05)
sel.fit(X_train, y_train)

sel.get_support()
selected_feat= X_train.columns[(sel.get_support())]


X_train_sel = sel.transform(X_train)
X_test_sel = sel.transform(X_test)

regressor_rf_sel = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor_rf_sel.fit(X_train_sel, y_train)
pred_rf_sel = regressor_rf_sel.predict(X_test_sel)

r_rf_sel = r2_score(y_test, pred_rf_sel)
RF_sel = mean_squared_error(y_test,pred_rf_sel)**(0.5)



print('RF Mean Sqr:',RF, 'R2:', r_rf)
print('RF_sel Mean Sqr:', RF_sel,'R2:', r_rf_sel)


sns.set()
cols = ['LifeExpectancy','status', 'AdultMortality', 'HepatitisB', 'BMI', 'Polio', 'Diphtheria', 'HIV/AIDS','GDP','Population','Thinness1-19Years','Thinness5-9Years','IncomeCompositionOfResources','Schooling']
sns.pairplot(dataset[cols], size = 2.5)
plt.show()


sns.set()
cols = [0, 'AdultMortality', 'HIV/AIDS','IncomeCompositionOfResources','Schooling']
sns.pairplot(new_data[cols], size = 2.5)
plt.show()






