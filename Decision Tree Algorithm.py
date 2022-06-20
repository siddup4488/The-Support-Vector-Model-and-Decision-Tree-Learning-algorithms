# Importing the required modules 
import pandas as pd
import numpy as np
from sklearn.model_selection
import StratifiedShuffleSplit #import to have equal weigtage samples in training dataset
from sklearn.tree
import DecisionTreeRegressor # import for Decision Tree Algorithm
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR #import for support vector regressor
from sklearn.metrics import mean_squared_error  # import to calculate root mean square
SourceData=pd.read_excel("Supplier Past Performance.xlsx") # Load the data into Pandas DataFrame
SourceData_independent= SourceData.drop(["Defect Percent"], axis=1) # Drop depedent variable from training dataset
SourceData_dependent=SourceData["Defect Percent"].copy() # New dataframe with only independent variable value for training dataset
SourceData["PO Category"]=pd.cut(SourceData["PO Amount "],
                                     bins=[0., 30000, 60000, 90000,
np.inf],                                     
labels=[1, 2, 3, 4])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
for train_index, test_index in split.split(SourceData, SourceData["PO Category"]):
    strat_train_set = SourceData.loc[train_index]  # stratfied train dataset 
    strat_test_set = SourceData.loc[test_index] #stratified test dataset
for set_ in (strat_train_set, strat_test_set): 
    set_.drop("PO Category", axis=1, inplace=True)
    
SourceData_train_independent= strat_train_set.drop(["Defect Percent"], axis=1)
SourceData_train_dependent=strat_train_set["Defect Percent"].copy()
SourceData_test_independent= strat_test_set.drop(["Defect Percent"], axis=1)
SourceData_test_dependent=strat_test_set["Defect Percent"].copy()
sc_X = StandardScaler()
X_train=sc_X.fit_transform(SourceData_train_independent.values)
y_train=SourceData_train_dependent
pickle.dump(sc_X, open("Scaler.sav", 'wb'))
X_test=sc_X.fit_transform(SourceData_test_independent.values)
y_test=SourceData_test_dependent
svm_reg = SVR(kernel="linear", C=1)
svm_reg.fit(X_train, y_train)
filename = 'SVR_TrainedModel.sav'
pickle.dump(svm_reg, open(filename, 'wb'),protocol=-1)
decision_predictions = svm_reg.predict(X_test)
Score = (svm_reg.score(X_test, y_test))  # It provides the R-Squared Value
print ( "The score of the Support  Vector model is", round(Score,2))
lin_mse = mean_squared_error(y_test, decision_predictions)
print("MSE  of  Vector  model is ", round(lin_mse,2))
lin_rmse = mean_squared_error(y_test, decision_predictions, squared=False)
print("RMSE of  Support  Vector  Learning model is ", round(lin_rmse,2))
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
filename = 'DecisionTree_TrainedModel.sav'
pickle.dump(tree_reg, open(filename, 'wb'),protocol=-1)
predictions = tree_reg.predict(X_test) 
Score = (tree_reg.score(X_test, y_test))  # It provides the R-Squared Value
print ( "The score of model Decision Tree model is ", round(Score,2))
lin_mse = mean_squared_error(y_test, predictions)
print("MSE of Decision Tree model is ", round(lin_mse,2))
lin_rmse = mean_squared_error(y_test, decision_predictions, squared=False)
print("RMSE of Decision Tree model is ", round(lin_rmse,2))
