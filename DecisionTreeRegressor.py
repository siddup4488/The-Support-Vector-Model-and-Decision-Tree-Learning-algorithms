""" Step 1 - Import the required modules"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVR

""" Step 2 - Read the data source"""
SourceData=pd.read_excel("Supplier Past Performance.xlsx") # Load the training data into Pandas DataFrame
Testdata=pd.read_excel("Defect Predict.xlsx") # Load the test data 

""" Step 3  - Declare the independent and dependent train data from the sample"""
SourceData_train_independent= SourceData.drop(["Defect Percent"], axis=1) # Drop depedent variable from training dataset
SourceData_train_dependent=SourceData["Defect Percent"].copy() #  New dataframe with only independent variable value for training dataset

""" Step 4  - Scale the independent test and train data"""
sc_X = StandardScaler()
X_train=sc_X.fit_transform(SourceData_train_independent.values) # scale the independent variables
y_train=SourceData_train_dependent # scaling is not required for dependent variable
X_test=sc_X.transform(Testdata.values)

""" Step 5  - Fit the test data in maching learning model - Support Vector Regressor"""
svm_reg = SVR(kernel="linear", C=1)
svm_reg.fit(X_train, y_train) # fit and train the model
predictions = svm_reg.predict(X_test)

print("Defect percent prediction by Support Vector model for the order value of 95827 GBP with 851 pallets sent 55 days before delivery data is " ,round(predictions[0],2) , "%")

""" Step 6 - Fit the test data in maching learning model - Decision Tree Model"""
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train) # fit and train the model
decision_predictions = tree_reg.predict(X_test) # Predict the value of dependent variable
print("Defect percent prediction by Decision Tree model for the order value of 95827 GBP with 851 pallets sent 55 days before delivery data is " ,round(decision_predictions[0],2) , "%")
