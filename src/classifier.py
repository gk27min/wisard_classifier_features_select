import wisardpkg as wsd
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def binarize(thermometer, X1, X2):
  num_features = X1.shape[1]
  thermometer_sizes = [thermometer] * num_features


  mins = np.min(X1, axis=0)
  maxs = np.max(X1, axis=0)

  dtherm = wsd.DynamicThermometer(thermometer_sizes, mins, maxs)
  binX_X1 = [dtherm.transform(X1[i]) for i in range(len(X1))]
  binX_X2 = [dtherm.transform(X2[i]) for i in range(len(X2))]
  return binX_X1, binX_X2

params = pd.read_csv('./optuna_study_wisard_results.cv')
X_train = pd.read_pickle('./X_train.pkl').to_numpy()
y_train = pd.read_pickle('./y_train.plk').to_numpy()
X_test = pd.read_pickle('./X_test.pkl').to_numpy()
y_test = pd.read_pickle('./y_train.pkl').to_numpy()
file_path_features = "./selected_features[{},{}].csv".format(params['min_perct_inclass'], params['max_perct_outclass'])
selected_features = np.loadtxt(file_path_features, delimiter=',')

BinX_train, BinX_test = binarize(params['thermometer'], X_train[:selected_features], X_test)

ds_train = wsd.DataSet(BinX_train, y_train)
ds_test = wsd.DataSet(BinX_test, y_test)

model = wsd.Wisard(params['ram'])
model.train(ds_train)
predicted = model.classify(ds_test)

score = accuracy_score(y_test, predicted)
print(score)