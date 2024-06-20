import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor as XGBR
#
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

# Import Data
dataset = pd.read_csv('ML-SS.csv')
# Separate datasets
array = dataset.values
X = array[:, 0:10]
Y = array[:, 10]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.3, random_state=7)
# Evaluation means
num_folds = 5
scoring1 = 'neg_mean_squared_error'
scoring2 = 'r2'
scoring3 = 'mean_absolute_error'


#Algorithm optimization - Ridge
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {
    'alpha': [0.1, 1, 10, 100, 1000],
    'fit_intercept': [True, False],
    }
ridge = Ridge()
kfold = KFold(n_splits=num_folds, shuffle=True)
grid = GridSearchCV(estimator=ridge, param_grid=param_grid,scoring=scoring2, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=Y_train)
print('Best：%s Use%s' % (grid_result.best_score_, grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('%f (%f) with %r' % (mean, std, param))
#Best：0.27807752062720886 Use{'alpha': 10, 'fit_intercept': True}
#Ridge model
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
ridge = Ridge(alpha= 10, fit_intercept= True)
ridge.fit(rescaledX, Y_train)

# Evaluate the training set
rescaledX_train = scaler.transform(X_train)
pre_train = ridge.predict(rescaledX_train)
print('Training_data，MSE：%s' % (mean_squared_error(Y_train, pre_train)))
print('Training_data，MAE：%s' % (mean_absolute_error(Y_train, pre_train)))
print('Training_data，R2：%s' %  (r2_score(Y_train, pre_train)))

# Evaluate the testing set
rescaledX_test = scaler.transform(X_test)
pre_test = ridge.predict(rescaledX_test)
print('Testing_data，MSE：%s' % (mean_squared_error(Y_test, pre_test)))
print('Testing_data，MAE：%s' % (mean_absolute_error(Y_test, pre_test)))
print('Testing_data，R2：%s' % (r2_score(Y_test, pre_test)))
#Data saving-Ridge
#training set - true value
trainRidge = pd.DataFrame(data = Y_train)
trainRidge.to_csv('E:\MLrobust2\Ridge_train.csv')
#training set - Predictive value
pretrainRidge = pd.DataFrame(data = pre_train)
pretrainRidge.to_csv('E:\MLrobust2\Ridge_predictions_train.csv')

#testing set - true value
testRidge = pd.DataFrame(data = Y_test)
testRidge.to_csv('E:\MLrobust2\Ridge_test.csv')
#testing set - Predictive value
pretestRidge = pd.DataFrame(data = pre_test)
pretestRidge.to_csv('E:\MLrobust2\Ridge_predictions_test.csv')

# Algorithm optimization - KNN
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {
    'n_neighbors': range(1, 10),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': range(10, 50, 10),
    'p': [1, 2]
}
model1 = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, shuffle=True)
grid = GridSearchCV(estimator=model1, param_grid=param_grid, scoring=scoring2, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=Y_train)
print('Best：%s Use%s' % (grid_result.best_score_, grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('%f (%f) with %r' % (mean, std, param))
#Best：0.7836133997269881 Use{'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 3, 'p': 1, 'weights': 'distance'}
#KNN model
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
KNN = KNeighborsRegressor(algorithm='auto', leaf_size = 10, n_neighbors= 3, p= 1, weights= 'distance')
KNN.fit(rescaledX, Y_train)

# Evaluate the training set
rescaledX_train = scaler.transform(X_train)
pre_train = KNN.predict(rescaledX_train)
print('Training_data，MSE：%s' % (mean_squared_error(Y_train, pre_train)))
print('Training_data，MAE：%s' % (mean_absolute_error(Y_train, pre_train)))
print('Training_data，R2：%s' %  (r2_score(Y_train, pre_train)))

# Evaluate the testing set
rescaledX_test = scaler.transform(X_test)
pre_test = KNN.predict(rescaledX_test)
print('Testing_data，MSE：%s' % (mean_squared_error(Y_test, pre_test)))
print('Testing_data，MAE：%s' % (mean_absolute_error(Y_test, pre_test)))
print('Testing_data，R2：%s' % (r2_score(Y_test, pre_test)))
#Data saving-KNN
#training set - true value
trainKNN = pd.DataFrame(data = Y_train)
trainKNN.to_csv('E:\MLrobust2\KNN_train.csv')
#training set - Predictive value
pretrainKNN = pd.DataFrame(data = pre_train)
pretrainKNN.to_csv('E:\MLrobust2\KNN_predictions_train.csv')

#testing set - true value
testKNN = pd.DataFrame(data = Y_test)
testKNN.to_csv('E:\MLrobust2\KNN_test.csv')
#testing set - Predictive value
pretestKNN = pd.DataFrame(data = pre_test)
pretestKNN.to_csv('E:\MLrobust2\KNN_predictions_test.csv')

# Algorithm optimization - DT
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {
    'criterion': ['squared_error', 'friedman_mse'],
    'max_depth': range(1, 10),
    'min_samples_split': range(2, 10),
    'min_samples_leaf': range(1, 5),
    'max_features': ['sqrt', 'log2', None],
    'random_state': [42],
}
model2 = DecisionTreeRegressor()
kfold = KFold(n_splits=num_folds, shuffle=True)
grid = GridSearchCV(estimator=model2, param_grid=param_grid, scoring=scoring2, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=Y_train)
print('Best：%s Use%s' % (grid_result.best_score_, grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('%f (%f) with %r' % (mean, std, param))
#Best：0.6165252392672137 Use{'criterion': 'squared_error', 'max_depth': 9, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'random_state': 42}
#DT model
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
DT = DecisionTreeRegressor(criterion= 'squared_error', max_depth= 9, max_features= None, min_samples_leaf= 2, min_samples_split= 5, random_state= 42)
DT.fit(rescaledX, Y_train)

# Evaluate the training set
rescaledX_train = scaler.transform(X_train)
pre_train = DT.predict(rescaledX_train)
print('Training_data，MSE：%s' % (mean_squared_error(Y_train, pre_train)))
print('Training_data，MAE：%s' % (mean_absolute_error(Y_train, pre_train)))
print('Training_data，R2：%s' %  (r2_score(Y_train, pre_train)))

# Evaluate the testing set
rescaledX_test = scaler.transform(X_test)
pre_test = DT.predict(rescaledX_test)
print('Testing_data，MSE：%s' % (mean_squared_error(Y_test, pre_test)))
print('Testing_data，MAE：%s' % (mean_absolute_error(Y_test, pre_test)))
print('Testing_data，R2：%s' % (r2_score(Y_test, pre_test)))
#Data saving-DT
#training set - true value
trainDT = pd.DataFrame(data = Y_train)
trainDT.to_csv('E:\MLrobust2\DT_train.csv')
#training set - Predictive value
pretrainDT = pd.DataFrame(data = pre_train)
pretrainDT.to_csv('E:\MLrobust2\DT_predictions_train.csv')

#testing set - true value
testDT = pd.DataFrame(data = Y_test)
testDT.to_csv('E:\MLrobust2\DT_test.csv')
#testing set - Predictive value
pretestDT = pd.DataFrame(data = pre_test)
pretestDT.to_csv('E:\MLrobust2\DT_predictions_test.csv')

# Algorithm optimization - SVM
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {
    'C': [1, 10, 100, 1000],  # 误差项的惩罚参数
    'gamma': ['scale', 'auto', 0.01, 0.001, 0.0001],
    'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
}
model3 = SVR()
kfold = KFold(n_splits=num_folds, shuffle=True)
grid = GridSearchCV(estimator=model3, param_grid=param_grid, scoring=scoring2, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=Y_train)
print('Best：%s Use%s' % (grid_result.best_score_, grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('%f (%f) with %r' % (mean, std, param))
#Best：0.8480091610878515 Use{'C': 10, 'epsilon': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}
#svr model
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
svr = SVR(C= 10, epsilon= 0.1, gamma= 'scale', kernel= 'rbf')
svr.fit(rescaledX, Y_train)

# Evaluate the training set
rescaledX_train = scaler.transform(X_train)
pre_train = svr.predict(rescaledX_train)
print('Training_data，MSE：%s' % (mean_squared_error(Y_train, pre_train)))
print('Training_data，MAE：%s' % (mean_absolute_error(Y_train, pre_train)))
print('Training_data，R2：%s' %  (r2_score(Y_train, pre_train)))

# Evaluate the testing set
rescaledX_test = scaler.transform(X_test)
pre_test = svr.predict(rescaledX_test)
print('Testing_data，MSE：%s' % (mean_squared_error(Y_test, pre_test)))
print('Testing_data，MAE：%s' % (mean_absolute_error(Y_test, pre_test)))
print('Testing_data，R2：%s' % (r2_score(Y_test, pre_test)))
#Data saving-svr
#training set - true value
trainsvr = pd.DataFrame(data = Y_train)
trainsvr.to_csv('E:\MLrobust2\svr_train.csv')
#training set - Predictive value
pretrainsvr = pd.DataFrame(data = pre_train)
pretrainsvr.to_csv('E:\MLrobust2\svr_predictions_train.csv')

#testing set - true value
testsvr = pd.DataFrame(data = Y_test)
testsvr.to_csv('E:\MLrobust2\svr_test.csv')
#testing set - Predictive value
pretestsvr = pd.DataFrame(data = pre_test)
pretestsvr.to_csv('E:\MLrobust2\svr_predictions_test.csv')

# Algorithm optimization - MLP
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {
    'hidden_layer_sizes': [(50,), (100,)],
    'activation': ['relu', 'tanh', 'identity'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [100, 500, 1000],
    'random_state': [42],
}
mlp = MLPRegressor()
kfold = KFold(n_splits=num_folds, shuffle=True)
grid = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring=scoring2, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=Y_train)
print('Best：%s Use%s' % (grid_result.best_score_, grid_result.best_params_))
#Best：0.8703514267479889 Use{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100,), 'learning_rate': 'constant', 'learning_rate_init': 0.01, 'max_iter': 500, 'random_state': 42, 'solver': 'adam'}
#mlp model
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
mlp = MLPRegressor(activation= 'relu', alpha= 0.0001, hidden_layer_sizes= (100,), learning_rate= 'constant', learning_rate_init= 0.01, max_iter= 500, random_state= 42, solver= 'adam')
mlp.fit(rescaledX, Y_train)

# Evaluate the training set
rescaledX_train = scaler.transform(X_train)
pre_train = mlp.predict(rescaledX_train)
print('Training_data，MSE：%s' % (mean_squared_error(Y_train, pre_train)))
print('Training_data，MAE：%s' % (mean_absolute_error(Y_train, pre_train)))
print('Training_data，R2：%s' %  (r2_score(Y_train, pre_train)))

# Evaluate the testing set
rescaledX_test = scaler.transform(X_test)
pre_test = mlp.predict(rescaledX_test)
print('Testing_data，MSE：%s' % (mean_squared_error(Y_test, pre_test)))
print('Testing_data，MAE：%s' % (mean_absolute_error(Y_test, pre_test)))
print('Testing_data，R2：%s' % (r2_score(Y_test, pre_test)))
#Data saving-mlp
#training set - true value
trainmlp = pd.DataFrame(data = Y_train)
trainmlp.to_csv('E:\MLrobust2\mlp_train.csv')
#training set - Predictive value
pretrainmlp = pd.DataFrame(data = pre_train)
pretrainmlp.to_csv('E:\MLrobust2\mlp_predictions_train.csv')

#testing set - true value
testmlp = pd.DataFrame(data = Y_test)
testmlp.to_csv('E:\MLrobust2\mlp_test.csv')
#testing set - Predictive value
pretestmlp = pd.DataFrame(data = pre_test)
pretestmlp.to_csv('E:\MLrobust2\mlp_predictions_test.csv')

# Algorithm optimization - ET
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {
    'n_estimators': [10, 20, 50, 100, 150, 200, 250, 300, 400, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'random_state': [42]
}
etr = ExtraTreesRegressor()
kfold = KFold(n_splits=num_folds, shuffle=True)
grid = GridSearchCV(estimator=etr, param_grid=param_grid, scoring=scoring2, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=Y_train)
print('Best：%s Use%s' % (grid_result.best_score_, grid_result.best_params_))
#Best：0.8641915260075705 Use{'bootstrap': False, 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400, 'random_state': 42}
#ET model
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
ET = ExtraTreesRegressor(bootstrap= False, max_depth= 20, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 400, random_state= 42)
ET.fit(rescaledX, Y_train)

# Evaluate the training set
rescaledX_train = scaler.transform(X_train)
pre_train = ET.predict(rescaledX_train)
print('Training_data，MSE：%s' % (mean_squared_error(Y_train, pre_train)))
print('Training_data，MAE：%s' % (mean_absolute_error(Y_train, pre_train)))
print('Training_data，R2：%s' %  (r2_score(Y_train, pre_train)))

# Evaluate the testing set
rescaledX_test = scaler.transform(X_test)
pre_test = ET.predict(rescaledX_test)
print('Testing_data，MSE：%s' % (mean_squared_error(Y_test, pre_test)))
print('Testing_data，MAE：%s' % (mean_absolute_error(Y_test, pre_test)))
print('Testing_data，R2：%s' % (r2_score(Y_test, pre_test)))
#Data saving-ET
#training set - true value
trainET = pd.DataFrame(data = Y_train)
trainET.to_csv('E:\MLrobust2\ET_train.csv')
#training set - Predictive value
pretrainET = pd.DataFrame(data = pre_train)
pretrainET.to_csv('E:\MLrobust2\ET_predictions_train.csv')

#testing set - true value
testET = pd.DataFrame(data = Y_test)
testET.to_csv('E:\MLrobust2\ET_test.csv')
#testing set - Predictive value
pretestET = pd.DataFrame(data = pre_test)
pretestET.to_csv('E:\MLrobust2\ET_predictions_test.csv')

#Algorithm optimization - XGB
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'gamma': [0, 0.25, 1.0],
    'reg_alpha': [0, 1.0, 10.0],
    'reg_lambda': [1.0, 2.0, 3.0],
}
xgb = XGBR()
kfold = KFold(n_splits=num_folds, shuffle=True)
grid = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring=scoring2, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=Y_train)
print('Best：%s Use%s' % (grid_result.best_score_, grid_result.best_params_))
#Best：0.8402983636887376 Use{'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 7, 'n_estimators': 1000, 'reg_alpha': 1.0, 'reg_lambda': 2.0, 'subsample': 0.7}
#xgb model
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
xgb = XGBR(colsample_bytree= 1.0, gamma= 0, learning_rate= 0.05, max_depth= 7, n_estimators= 1000, reg_alpha= 1.0, reg_lambda= 2.0, subsample= 0.7)
xgb.fit(rescaledX, Y_train)

# Evaluate the training set
rescaledX_train = scaler.transform(X_train)
pre_train = xgb.predict(rescaledX_train)
print('Training_data，MSE：%s' % (mean_squared_error(Y_train, pre_train)))
print('Training_data，MAE：%s' % (mean_absolute_error(Y_train, pre_train)))
print('Training_data，R2：%s' %  (r2_score(Y_train, pre_train)))

# Evaluate the testing set
rescaledX_test = scaler.transform(X_test)
pre_test = xgb.predict(rescaledX_test)
print('Testing_data，MSE：%s' % (mean_squared_error(Y_test, pre_test)))
print('Testing_data，MAE：%s' % (mean_absolute_error(Y_test, pre_test)))
print('Testing_data，R2：%s' % (r2_score(Y_test, pre_test)))
#Data saving-xgb
#training set - true value
trainxgb = pd.DataFrame(data = Y_train)
trainxgb.to_csv('E:\MLrobust2\\xgb_train.csv')
#training set - Predictive value
pretrainxgb = pd.DataFrame(data = pre_train)
pretrainxgb.to_csv('E:\MLrobust2\\xgb_predictions_train.csv')

#testing set - true value
testxgb = pd.DataFrame(data = Y_test)
testxgb.to_csv('E:\MLrobust2\\xgb_test.csv')
#testing set - Predictive value
pretestxgb = pd.DataFrame(data = pre_test)
pretestxgb.to_csv('E:\MLrobust2\\xgb_predictions_test.csv')