import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
# Import data set from URL and store data in df
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
# Linear regression to predict car price based on highway mpg
lm = LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)
Yhat=lm.predict(X)  # prediction
lm.intercept_   # intercept (a)
lm.coef_        # slope (b)
# Multiple Linear Regression model using four variables as predictor variables
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])
lm.intercept_   # intercept (a)
lm.coef_        # slope coefficient (b1, b2, b3, b4)
lm2 = LinearRegression()
lm2.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])
lm2.intercept_   # intercept (a)
lm2.coef_        # slope coefficient (b1, b2, b3, b4)
# Regression plot to visualize horsepower as predictor variable of price
width = 6   # window width and height
height = 5
#plt.figure(figsize=(width, height))
#sns.regplot(x="highway-mpg", y="price", data=df)
#plt.ylim(0,)
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
print('Correlation between peak rpm, highway mpg and price:')
print(df[["peak-rpm","highway-mpg","price"]].corr())
# Residual plot
#plt.figure(figsize=(width, height))
#sns.residplot(df['highway-mpg'], df['price'])
# Multiple Linear Regression
Y_hat = lm.predict(Z)
#plt.figure(figsize=(width, height))
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
#sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)
#plt.title('Actual vs Fitted Values for Price')
#plt.xlabel('Price (in dollars)')
#plt.ylabel('Proportion of Cars')
# Polynomial Regression and Pipelines
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit for Price')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
x = df['highway-mpg']
y = df['price']
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
PlotPolly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
PlotPolly(p1,x,y, 'Highway MPG')
# Create a pipeline using Pipeline module
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)
pipe=Pipeline(Input)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]
Input1=[('scale',StandardScaler()),('model',LinearRegression())]        # pipeline using linear regression model
pipe1=Pipeline(Input1)
pipe1.fit(Z,y)
ypipe1=pipe1.predict(Z)
# Measures for evaluating models
lm.fit(X, Y)                # simple linear regression
print('The R-square is: ', lm.score(X, Y))
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)
lm.fit(Z, df['price'])      # multiple linear regression
print('The R-square is: ', lm.score(Z, df['price']))
Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))
r_squared = r2_score(y, p(x))           # polynomial fit
print('The R-square value is: ', r_squared)
# Prediction
new_input=np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, Y)
yhat=lm.predict(new_input)
plt.plot(new_input, yhat)
