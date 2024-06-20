#shapley-SVR
import shap
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colormaps
from sklearn.svm import SVR

# Import Data
dataset = pd.read_csv('ML-SS.csv')
# Separate datasets
array = dataset.values
X = array[:, 0:10]
y = array[:, 10]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=7)
scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)

# Training the SVR model
model = SVR(C= 10, epsilon= 0.1, gamma= 'scale', kernel= 'rbf')
model.fit(rescaledX_train, y_train)
# Define the svr_predict function
def svr_predict(X):
    return model.predict(X)
# Use the shap library to interpret the model
explainer = shap.KernelExplainer(svr_predict, rescaledX_train)
shapva = explainer(rescaledX_test)
shap_values =shapva.values
feature = ['Day', 'Temperature', 'pH', 'CN', 'Ammonia', 'Nitrate', 'TN', 'OM', 'Aeration rate', 'Additives']
rescaledX_test =pd.DataFrame(rescaledX_test,columns=feature)

# Set the global font for matplotlib
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# Create a figure and axes object
fig, ax = plt.subplots()
# Visualize Feature Importance
shap.summary_plot(shap_values, rescaledX_test,cmap = colormaps.get_cmap('viridis'))
ax.spines['top'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.show()
# Save graphics
plt.savefig('shap_SVR.pdf', format='pdf', bbox_inches='tight')
# Calculate the average absolute SHAP value for each feature
feature_importance = np.average(abs(shap_values),0)
# Output feature importance
for feature_index, importance in enumerate(feature_importance):
    print(f"Feature {feature_index}: Importance {importance}")


#GSA-SVR
import numpy as np
import pandas as pd
from SALib.sample import sobol
from SALib.analyze import sobol as si
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Import Data
dataset = pd.read_csv('ML-SS.csv')
# Separate datasets
array = dataset.values
X = array[:, 0:10]
y = array[:, 10]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=7)
scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)

bounds = pd.DataFrame(index=dataset.iloc[:, 0:10].columns,columns=['min','max'])
bounds['min'] = np.min(X,0)
bounds['max'] = np.max(X,0)
# Define the parameters of the problem
problem = {
    'num_vars': 10,
    'names': ['Day', 'Temperature', 'pH', 'CN', 'Ammonia', 'Nitrate', 'TN', 'OM', '曝气强度', '是否采用特殊添加剂'],
    'bounds': bounds.values.astype('float')
}
# Generate Sobol sequence samples
num_samples = 32768
param_values = sobol.sample(problem, num_samples)
scaledparam_values = scaler.transform(param_values)
# Get predictions for sample points
model = SVR(C= 10, epsilon= 0.1, gamma= 'scale', kernel= 'rbf')
model.fit(rescaledX_train, y_train)
y_pred = model.predict(scaledparam_values)

# Calculate the Sobol index
Si = si.analyze(problem, y_pred, print_to_console=True)

# Output results
for i, name in enumerate(problem['names']):
    print(f"Sobol index for {name}:")
    print(f"S1: {Si['S1'][i]}")
    print(f"ST: {Si['ST'][i]}")
    print("------------------------")


#PFI-SVR
import pandas as pd
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Import Data
dataset = pd.read_csv('ML-SS.csv')
# Separate datasets
array = dataset.values
X = array[:, 0:10]
y = array[:, 10]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=7)
scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)

# Initialize the SVR model
svr = SVR(C= 10, epsilon= 0.1, gamma= 'scale', kernel= 'rbf')

# Train the SVR model
svr.fit(rescaledX_train, y_train)

# Calculate the importance of replacement features
result = permutation_importance(svr, rescaledX_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
feature_importances = result.importances_mean


# Output the importance of each feature
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
for feature_name, importance in zip(feature_names, feature_importances):
    print(f'{feature_name}: {importance:.3f}')

#shapley-ET
import shap
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
from matplotlib import colormaps
# Import Data
dataset = pd.read_csv('ML-SS.csv')
# Separate datasets
array = dataset.values
X = array[:, 0:10]
y = array[:, 10]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=7)
scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)

model = ExtraTreesRegressor(bootstrap= False, max_depth= 20, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 400, random_state= 42)
model.fit(rescaledX_train, y_train)
def svr_predict(X):
    return model.predict(X)
explainer = shap.KernelExplainer(svr_predict, rescaledX_train)
shapva = explainer(rescaledX_test)
shap_values =shapva.values
feature = ['Day', 'Temperature', 'pH', 'CN', 'Ammonia', 'Nitrate', 'TN', 'OM', 'Aeration rate', 'Additives']
rescaledX_test =pd.DataFrame(rescaledX_test,columns=feature)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
fig, ax = plt.subplots()
shap.summary_plot(shap_values, rescaledX_test,cmap = colormaps.get_cmap('viridis'))
ax.spines['top'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.show()
plt.savefig('shap_ET.pdf', format='pdf', bbox_inches='tight')

feature_importance = np.average(abs(shap_values),0)
for feature_index, importance in enumerate(feature_importance):
    print(f"Feature {feature_index}: Importance {importance}")


#GSA-ET
import numpy as np
import pandas as pd
from SALib.sample import sobol
from SALib.analyze import sobol as si
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

# Import Data
dataset = pd.read_csv('ML-SS.csv')
# Separate datasets
array = dataset.values
X = array[:, 0:10]
y = array[:, 10]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=7)
scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)

bounds = pd.DataFrame(index=dataset.iloc[:, 0:10].columns,columns=['min','max'])
bounds['min'] = np.min(X,0)
bounds['max'] = np.max(X,0)

problem = {
    'num_vars': 10,
    'names': ['Day', 'Temperature', 'pH', 'CN', 'Ammonia', 'Nitrate', 'TN', 'OM', '曝气强度', '是否采用特殊添加剂'],
    'bounds': bounds.values.astype('float')
}

num_samples = 32768
param_values = sobol.sample(problem, num_samples)
scaledparam_values = scaler.transform(param_values)

model = ExtraTreesRegressor(bootstrap= False, max_depth= 20, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 400, random_state= 42)
model.fit(rescaledX_train, y_train)
y_pred = model.predict(scaledparam_values)

Si = si.analyze(problem, y_pred, print_to_console=True)

for i, name in enumerate(problem['names']):
    print(f"Sobol index for {name}:")
    print(f"S1: {Si['S1'][i]}")
    print(f"ST: {Si['ST'][i]}")
    print("------------------------")



#PFI-ET
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Import Data
dataset = pd.read_csv('ML-SS.csv')
# Separate datasets
array = dataset.values
X = array[:, 0:10]
y = array[:, 10]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=7)
scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)

et = ExtraTreesRegressor(bootstrap= False, max_depth= 20, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 400, random_state= 42)

et.fit(rescaledX_train, y_train)

result = permutation_importance(et, rescaledX_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

feature_importances = result.importances_mean

feature_names = [f'feature_{i}' for i in range(X.shape[1])]

for feature_name, importance in zip(feature_names, feature_importances):
    print(f'{feature_name}: {importance:.3f}')

#shapley-MLP
import shap
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from matplotlib import colormaps
# Import Data
dataset = pd.read_csv('ML-SS.csv')
# Separate datasets
array = dataset.values
X = array[:, 0:10]
y = array[:, 10]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=7)
scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)

model = MLPRegressor(activation= 'relu', alpha= 0.0001, hidden_layer_sizes= (100,), learning_rate= 'constant', learning_rate_init= 0.01, max_iter= 500, random_state= 42, solver= 'adam')
model.fit(rescaledX_train, y_train)

def svr_predict(X):
    return model.predict(X)

explainer = shap.KernelExplainer(svr_predict, rescaledX_train)
shapva = explainer(rescaledX_test)
shap_values =shapva.values
feature = ['Day', 'Temperature', 'pH', 'CN', 'Ammonia', 'Nitrate', 'TN', 'OM', 'Aeration rate', 'Additives']
rescaledX_test =pd.DataFrame(rescaledX_test,columns=feature)


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots()

shap.summary_plot(shap_values, rescaledX_test,cmap = colormaps.get_cmap('viridis'))
ax.spines['top'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.show()
plt.savefig('shap_MLP.pdf', format='pdf', bbox_inches='tight')

feature_importance = np.average(abs(shap_values),0)

for feature_index, importance in enumerate(feature_importance):
    print(f"Feature {feature_index}: Importance {importance}")


#GSA-MLP
import numpy as np
import pandas as pd
from SALib.sample import sobol
from SALib.analyze import sobol as si
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Import Data
dataset = pd.read_csv('ML-SS.csv')
# Separate datasets
array = dataset.values
X = array[:, 0:10]
y = array[:, 10]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=7)
scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)

bounds = pd.DataFrame(index=dataset.iloc[:, 0:10].columns,columns=['min','max'])
bounds['min'] = np.min(X,0)
bounds['max'] = np.max(X,0)

problem = {
    'num_vars': 10,
    'names': ['Day', 'Temperature', 'pH', 'CN', 'Ammonia', 'Nitrate', 'TN', 'OM', '曝气强度', '是否采用特殊添加剂'],
    'bounds': bounds.values.astype('float')
}

num_samples = 32768
param_values = sobol.sample(problem, num_samples)
scaledparam_values = scaler.transform(param_values)

model = MLPRegressor(activation= 'relu', alpha= 0.0001, hidden_layer_sizes= (100,), learning_rate= 'constant', learning_rate_init= 0.01, max_iter= 500, random_state= 42, solver= 'adam')
model.fit(rescaledX_train, y_train)
y_pred = model.predict(scaledparam_values)

Si = si.analyze(problem, y_pred, print_to_console=True)

for i, name in enumerate(problem['names']):
    print(f"Sobol index for {name}:")
    print(f"S1: {Si['S1'][i]}")
    print(f"ST: {Si['ST'][i]}")
    print("------------------------")

#PFI-MLP
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Import Data
dataset = pd.read_csv('ML-SS.csv')
# Separate datasets
array = dataset.values
X = array[:, 0:10]
y = array[:, 10]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=7)
scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)

mlp = MLPRegressor(activation= 'relu', alpha= 0.0001, hidden_layer_sizes= (100,), learning_rate= 'constant', learning_rate_init= 0.01, max_iter= 500, random_state= 42, solver= 'adam')

mlp.fit(rescaledX_train, y_train)

result = permutation_importance(mlp, rescaledX_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

feature_importances = result.importances_mean

feature_names = [f'feature_{i}' for i in range(X.shape[1])]

for feature_name, importance in zip(feature_names, feature_importances):
    print(f'{feature_name}: {importance:.3f}')
