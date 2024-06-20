#PDP
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn.inspection import partial_dependence


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
feature_name = ['Day', 'T', 'pH', 'C/N', 'NH4+', 'NO3-', 'TN', 'OM', 'AR', 'Additives']
features = list(range(10))
# Mapping
colors = ['#1f77b4','#ff7f0e','#2ca02c',  '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf']
plt.figure(figsize=(16, 12))
for i, feature in enumerate(features, 1):
    pd_results = partial_dependence(model, rescaledX_train, feature, grid_resolution=20)
    grid, pd_values =  pd_results['grid_values'],pd_results['average']
    center1, scale1 = scaler.center_[feature], scaler.scale_[feature]
    grid_x = grid[0] * scale1 + center1
    plt.subplot(4, 3, i)
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.plot(grid_x, pd_values.T, label='Partial Dependence',color=colors[i-1])
    plt.xlabel(feature_name[feature])
    plt.ylabel('Partial Dependence')
plt.tight_layout()
# Save Graphics
plt.savefig("PDP.pdf", format="pdf")
plt.show()