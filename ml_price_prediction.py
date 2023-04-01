### Name: ML for Price Prediction on upcoming auction lots

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import statsmodels.api as sm 
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pymongo
from forex_python.converter import CurrencyRates
### Pre processing Train Data
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["final_project"]
table = db["past_auctions"]
data = []
i = 1
for x in table.find({},{'Lot':1,
                        'Primary description':1,
                       'Secondary description':1,
                       'Url':1,
                       'Estimate':1,
                       'Price Realise':1,
                       'Auction Info':1}):
    data.append([[list(pair) for pair in x.items()][1],
    [list(pair) for pair in x.items()][2],
    [list(pair) for pair in x.items()][3],
    [list(pair) for pair in x.items()][4],
    [list(pair) for pair in x.items()][5],
    [list(pair) for pair in x.items()][6],
    [list(pair) for pair in x.items()][7]])
    i+=1
    
data2 = pd.DataFrame(data)
Lot = []
for i in data2[0]:
    Lot.append(i[1])
    
Primary_description = []
for i in data2[1]:
    Primary_description.append(i[1])

Secondary_description = []
for i in data2[2]:
    Secondary_description.append(i[1])
    
Url = []
for i in data2[3]:
    Url.append(i[1])    
    
Estimate = []
for i in data2[4]:
    Estimate.append(i[1])  
    
Price_Realise = []
for i in data2[5]:
    Price_Realise.append(i[1])      
    
Auction_Name = []
Auction_Location = []
Sales_Total = []
Auction_Date = []
Department = []

for i in data2[6]:
    Auction_Name.append([list(pair) for pair in i[1].items()][0][1])
    Auction_Location.append([list(pair) for pair in i[1].items()][1][1])
    Sales_Total.append([list(pair) for pair in i[1].items()][2][1])
    Auction_Date.append([list(pair) for pair in i[1].items()][3][1])
    Department.append([list(pair) for pair in i[1].items()][4][1])
    
Estimate_Currency = []
Estimate_Lower = []
Estimate_Upper = []

for i in data2[4]:
    Estimate_Currency.append(i[1].split(" ")[0])
    try:
        Estimate_Lower.append(int(i[1].split(" ")[1].replace(",","")))
    except:
        Estimate_Lower.append(0)
    try:
        Estimate_Upper.append(int(i[1].split(" ")[3].replace(",","")))
    except:
        Estimate_Upper.append(0)

Realized_Currency = []
Realized_Final = []

for i in data2[5]:
    try:
        Realized_Currency.append(i[1].split(" ")[0])
        Realized_Final.append(int(i[1].split(" ")[1].replace(",","")))
    except:
        Realized_Currency.append("No realised price available")
        Realized_Final.append("No realised price available")
        
Sales_Total_Currency = []
Sales_Total_Final = []

for i in Sales_Total:
    try:
        Sales_Total_Currency.append(i.split(" ")[0])
        Sales_Total_Final.append(int(i.split(" ")[1].replace(",","")))
    except:
        Sales_Total_Currency.append("No sales total available")
        Sales_Total_Final.append("No sales total available")        
        
TrainData = pd.DataFrame(list(zip(Lot, Primary_description, Secondary_description, Url, Estimate, Estimate_Currency, Estimate_Lower, Estimate_Upper, Price_Realise, Realized_Currency, Realized_Final, Auction_Name, Auction_Location, Sales_Total, Sales_Total_Currency, Sales_Total_Final, Auction_Date, Department)), columns =['Lot', 'Primary description', 'Secondary description', 'Url', 'Estimate', 'Estimate Currency', 'Estimate Lower', 'Estimate Upper', 'Price Realise', 'Realized Currency', 'Realized Final', 'Auction Name', 'Auction Location', 'Sales Total', 'Sales Total Currency', 'Sales Total Final', 'Auction Date', 'Department'])
TrainData
Conversion_Foreign_USD = []
Conversion_Rate = []

for i in list(set(TrainData["Estimate Currency"])):
    try:
        Conversion_Rate.append(CurrencyRates().convert(i, "USD", 1))
        Conversion_Foreign_USD.append(i)
    except:
        0
        
currency_conversion = pd.DataFrame(list(zip(Conversion_Foreign_USD, Conversion_Rate)), columns = ['Conversion_Foreign_USD','Conversion_Rate'])
currency_conversion
TrainData["Estimate Lower USD"] = 0
TrainData["Estimate Upper USD"] = 0
TrainData["Realized Final USD"] = 0
TrainData["Sales Total Final USD"] = 0

for i in np.arange(len(TrainData)):
    for j in np.arange(len(currency_conversion)):
        if TrainData["Estimate Currency"][i]==currency_conversion["Conversion_Foreign_USD"][j]:
            try:
                TrainData["Estimate Lower USD"][i] = round(TrainData["Estimate Lower"][i]*currency_conversion["Conversion_Rate"][j])
            except: 
                TrainData["Estimate Lower USD"][i] = 0
            try:
                TrainData["Estimate Upper USD"][i] = round(TrainData["Estimate Upper"][i]*currency_conversion["Conversion_Rate"][j])
            except:
                TrainData["Estimate Upper USD"][i] = 0
            try:
                TrainData["Realized Final USD"][i] = round(TrainData["Realized Final"][i]*currency_conversion["Conversion_Rate"][j])
            except:
                TrainData["Realized Final USD"][i] = 0 
            try:
                TrainData["Sales Total Final USD"][i] = round(TrainData["Sales Total Final"][i]*currency_conversion["Conversion_Rate"][j])
            except:
                TrainData["Sales Total Final USD"][i] = 0
                

TrainData["Lot Sold"] = np.where(TrainData["Price Realise"]==0, 0, 1)
TrainData["Lot x"] = 1
TrainData
### Pre processing Test Data
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["final_project"]
table2 = db["upcoming_auctions"]
table2
test_data = []
i = 1
for x in table2.find({},{'Lot':1,
                        'Primary description':1,
                       'Secondary description':1,
                       'Url':1,
                       'Estimate':1,
                       'Auction Info':1}):
    test_data.append([[list(pair) for pair in x.items()][1],
    [list(pair) for pair in x.items()][2],
    [list(pair) for pair in x.items()][3],
    [list(pair) for pair in x.items()][4],
    [list(pair) for pair in x.items()][5],
    [list(pair) for pair in x.items()][6]])
    i+=1

test_data
data2 = pd.DataFrame(test_data)
Lot = []
for i in data2[0]:
    Lot.append(i[1])
    
Primary_description = []
for i in data2[1]:
    Primary_description.append(i[1])

Secondary_description = []
for i in data2[2]:
    Secondary_description.append(i[1])
    
Url = []
for i in data2[3]:
    Url.append(i[1])    
    
Estimate = []
for i in data2[4]:
    Estimate.append(i[1])      
    
Auction_Name = []
Auction_Location = []
Auction_Date = []
Department = []

for i in data2[5]:
    Auction_Name.append([list(pair) for pair in i[1].items()][0][1])
    Auction_Location.append([list(pair) for pair in i[1].items()][1][1])
    Auction_Date.append([list(pair) for pair in i[1].items()][2][1])
    Department.append([list(pair) for pair in i[1].items()][3][1])
    
Estimate_Currency = []
Estimate_Lower = []
Estimate_Upper = []

for i in data2[4]:
    Estimate_Currency.append(i[1].split(" ")[0])
    try:
        Estimate_Lower.append(int(i[1].split(" ")[1].replace(",","")))
    except:
        Estimate_Lower.append(0)
    try:
        Estimate_Upper.append(int(i[1].split(" ")[3].replace(",","")))
    except:
        Estimate_Upper.append(0)
        
TestData = pd.DataFrame(list(zip(Lot, Primary_description, Secondary_description, Url, Estimate, Estimate_Currency, Estimate_Lower, Estimate_Upper, Auction_Name, Auction_Location, Auction_Date, Department)), columns =['Lot', 'Primary description', 'Secondary description', 'Url', 'Estimate', 'Estimate Currency', 'Estimate Lower', 'Estimate Upper', 'Auction Name', 'Auction Location', 'Auction Date', 'Department'])
TestData
TestData["Estimate Lower USD"] = 0
TestData["Estimate Upper USD"] = 0

for i in np.arange(len(TestData)):
    for j in np.arange(len(currency_conversion)):
        if TestData["Estimate Currency"][i]==currency_conversion["Conversion_Foreign_USD"][j]:
            try:
                TestData["Estimate Lower USD"][i] = round(TestData["Estimate Lower"][i]*currency_conversion["Conversion_Rate"][j])
            except: 
                TestData["Estimate Lower USD"][i] = 0
            try:
                TestData["Estimate Upper USD"][i] = round(TestData["Estimate Upper"][i]*currency_conversion["Conversion_Rate"][j])
            except:
                TestData["Estimate Upper USD"][i] = 0

TestData            
### EDA
TrainData.columns
print("The percentage lots unsold: ", round(len(TrainData[TrainData["Price Realise"]==0])/len(TrainData)*100,2),"%")
import plotly.express as px

fig = px.pie(TrainData, values='Lot x', names='Lot Sold',
                title="Percentage of Lots sold")
fig.show()
df = TrainData.groupby(['Auction Location']).aggregate({'Auction Name':'nunique', 'Lot':'count', 'Lot Sold' : 'sum', 'Estimate Lower USD': 'mean', 'Estimate Upper USD': 'mean'}).reset_index()
df["Estimate Lower USD"] = round(df["Estimate Lower USD"])
df["Estimate Upper USD"] = round(df["Estimate Upper USD"])
df
fig = px.bar(df, x='Auction Location', y='Auction Name',
            labels={
                     "Auction Location": "Auction Location",
                     "Auction Name": "Number of Auctions"},
                title="Location wise auction count")
fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total ascending'})
fig.show()
fig = px.bar(df, x='Auction Location', y='Lot',
                 labels={
                     "Auction Location": "Auction Location",
                     "Lot": "Number of Lots"},
                title="Location wise Lots count")
fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total ascending'})
fig.show()
fig = px.bar(df, x='Auction Location', y='Estimate Lower USD',
                 labels={
                     "Auction Location": "Auction Location",
                     "Estimate Lower USD": "Mean Estimate Lower USD"},
                title="Location wise Lots count")
fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total ascending'})
fig.show()
### Data cleaning
### Removing Worldwide as that would overlap with other auction locations
TrainData = TrainData[TrainData["Auction Location"] != "Worldwide"]

### Removing the lots which did not sell since we are doing price prediction
TrainData = TrainData[TrainData["Price Realise"] != 0]
TrainData
### Train Test Split
Y = TrainData['Realized Final USD']
X = pd.get_dummies(TrainData[['Estimate Lower USD', 'Estimate Upper USD','Auction Location', 'Department']])
TestData2 = pd.concat([pd.DataFrame(columns=X.columns), TestData])[X.columns].fillna(0)
TestData2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import RepeatedKFold

X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=.2)
### Basic Linear Regression
# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# RMSE
MSE = np.square(np.subtract(y_test,reg.predict(X_test))).mean() 
lr_RMSE = math.sqrt(MSE)

print('RMSE: {}'.format(round(lr_RMSE),2))

# plot for residual error

## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,color = "red", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()
### Ridge Regression
#Ridge Regression Model
ridgeReg = Ridge(alpha=10)

ridgeReg.fit(X_train,y_train)

# RMSE
MSE = np.square(np.subtract(y_test,ridgeReg.predict(X_test))).mean() 
ridge_RMSE = math.sqrt(MSE)

print('RMSE: {}'.format(round(ridge_RMSE),2))

# plot for residual error

## plotting residual errors in training data
plt.scatter(ridgeReg.predict(X_train), ridgeReg.predict(X_train) - y_train,color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(ridgeReg.predict(X_test), ridgeReg.predict(X_test) - y_test,color = "red", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()
### Lasso Regression
#Lasso Regression Model
lasso = Lasso(alpha = 10)
lasso.fit(X_train,y_train)

# RMSE
MSE = np.square(np.subtract(y_test, lasso.predict(X_test))).mean() 
lasso_RMSE = math.sqrt(MSE)

print('RMSE: {}'.format(round(lasso_RMSE),2))

# plot for residual error

## plotting residual errors in training data
plt.scatter(lasso.predict(X_train), lasso.predict(X_train) - y_train,color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(lasso.predict(X_test), lasso.predict(X_test) - y_test,color = "red", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()
### RidgeCV Regression
#define cross-validation method to evaluate model
cv = RepeatedKFold(n_splits=20, n_repeats=3, random_state=9496)

#Lasso Cross validation
ridge_cv = RidgeCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10], cv=cv).fit(X_train, y_train)

#fit model
ridge_cv.fit(X_train, y_train)

# RMSE
MSE = np.square(np.subtract(y_test, ridge_cv.predict(X_test))).mean() 
rcv_RMSE = math.sqrt(MSE)

print('RMSE: {}'.format(round(rcv_RMSE),2))

# plot for residual error

## plotting residual errors in training data
plt.scatter(ridge_cv.predict(X_train), ridge_cv.predict(X_train) - y_train,color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(ridge_cv.predict(X_test), ridge_cv.predict(X_test) - y_test,color = "red", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()
### LassoCV Regression
#define cross-validation method to evaluate model
cv = RepeatedKFold(n_splits=20, n_repeats=3, random_state=9496)

#define model
lassocvmodel = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cv)

#fit model
lassocvmodel.fit(X_train, y_train)

# RMSE
MSE = np.square(np.subtract(y_test, lassocvmodel.predict(X_test))).mean() 
lcv_RMSE = math.sqrt(MSE)

print('RMSE: {}'.format(round(lcv_RMSE),2))

# plot for residual error

## plotting residual errors in training data
plt.scatter(lassocvmodel.predict(X_train), lassocvmodel.predict(X_train) - y_train,color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(lassocvmodel.predict(X_test), lassocvmodel.predict(X_test) - y_test,color = "red", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()
### Regression Tree
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=9496)
dtr.fit(X_train, y_train)
predictions = dtr.predict(X_test)

# RMSE
MSE = np.square(np.subtract(y_test, predictions)).mean() 
drt_RMSE = math.sqrt(MSE)

print('RMSE: {}'.format(round(drt_RMSE),2))

# plot for residual error

## plotting residual errors in training data
plt.scatter(dtr.predict(X_train), dtr.predict(X_train) - y_train,color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(dtr.predict(X_test), dtr.predict(X_test) - y_test,color = "red", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()
### Random Forest
df1 = pd.DataFrame([1,2,3], columns = [1])
df2 = pd.DataFrame([4,5,6])

df2
pd.concat([df1,df2])
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

# RMSE
MSE = np.square(np.subtract(y_test, predictions)).mean() 
rf_RMSE = math.sqrt(MSE)

print('RMSE: {}'.format(round(rf_RMSE),2))
# plot for residual error

## plotting residual errors in training data
plt.scatter(rf.predict(X_train), rf.predict(X_train) - y_train,color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(rf.predict(X_test), rf.predict(X_test) - y_test,color = "red", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()
Model = ["Basic Linear Regression", "Ridge", "Lasso", "RidgeCV", "LassoCV", "Regression Tree", "Random Forest"]
RMSE = [round(lr_RMSE), round(ridge_RMSE), round(lasso_RMSE), round(rcv_RMSE), round(lcv_RMSE), round(drt_RMSE), round(rf_RMSE)]

pd.DataFrame(list(zip(Model, RMSE)), columns=['Model', 'RMSE']).sort_values(['RMSE'])
### From the Models we have built we can see that Random Forest has the lowest RMSE amongst all the Models. We will go ahead with this model and predict the Price Realised (Hammer Price) for the upcoming auction items.
### Price prediction for upcoming auction items using the model build.
pd.concat([TestData, pd.DataFrame(rf.predict(TestData2),columns = ["Predicted Price Realised"])], axis = 1)
### Now we have the predicted price realised, we will wait for the auction to play out to see how accurately we have predicted the price!
