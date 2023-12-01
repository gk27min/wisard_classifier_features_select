import json
import numpy as np
import pandas as pd
import wisardpkg as wsd
from scipy.sparse import load_npz
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin

class WiSARDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, thermometer, ram):
        self.thermometer = thermometer
        self.ram = ram
        self.model = None

    def fit(self, X, y):
        # Treinamento do modelo WiSARD
        ds_train = wsd.DataSet(X, y)
        self.model = wsd.Wisard(self.ram)
        self.model.train(ds_train)
        del X, ds_train

    def predict(self, X):
        # Classificação com o modelo WiSARD
        ds_test = wsd.DataSet(X)
        result = self.model.classify(ds_test)
        del X, ds_test, self.model
        return result

class BinarizationTransformer(BaseEstimator):
    def __init__(self, thermometer):
        self.thermometer = thermometer

    def fit(self, X, y=None):
        # Não é necessário fazer nada no método fit
        return self

    def transform(self, X):
        # Binariza os dados usando o termômetro e a RAM fornecidos
        num_features = X.shape[1]
        thermometer_sizes = [self.thermometer] * num_features

        mins = X.min(axis=0).astype(float).toarray().tolist()[0]
        maxs = X.max(axis=0).astype(float).toarray().tolist()[0]

        dtherm = wsd.DynamicThermometer(thermometer_sizes, mins, maxs)
        del mins, maxs

        # Aplica a transformação em todo o conjunto de dados de uma vez
        binX = [dtherm.transform(X.getcol(i).toarray().flatten()) for i in range(X.shape[1])]
        print("fim da binarização!")
        return binX

class CustomWiSARDClassifier(BaseEstimator):
    def __init__(self, thermometer, ram):
        self.thermometer = thermometer
        self.ram = ram
        self.clf = WiSARDClassifier(thermometer=self.thermometer, ram=self.ram)

    def fit(self, X, y):
        # Não é necessário binarizar aqui, pois já fizemos isso em BinarizationTransformer
        self.clf.fit(X, y)

    def predict(self, X):
        # Não é necessário binarizar aqui, pois já fizemos isso em BinarizationTransformer
        return self.clf.predict(X)

def custom_scorer(estimator, X, y):
    predicted = estimator.predict(X)
    score = accuracy_score(y, predicted)
    print(f"params: {estimator.thermometer} {estimator.ram}, acuracy: {score: 0.2}" )
    return score

#modifica as configurações padrão considerando as configurações passadas pelo usuário
vocabulary_file = '/home/gssilva/datasets/atribuna-site/full/selections/vocabulary_0.60-0.20.json'
train_folder = '/home/gssilva/datasets/atribuna-site/full/train_test'
output_file_name = '/home/gssilva/datasets/atribuna-site/full/results/otimizacao_gridsearch.csv'
min_inclass = 0.6
max_outclass = 0.2
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_procs = 5
binarization__thermometer = 8


X_train = load_npz(f'{train_folder}/X_train.npz')
X_train = X_train.tocsc()
y_train = np.load(f'{train_folder}/y_train.npy', allow_pickle=True)

with open(vocabulary_file, 'r') as file:
    vocabulary = json.load(file)

with open(vocabulary_file, 'r') as file:
    vocabulary = json.load(file)
selected_features_indices = list(vocabulary.values())

transformed_train = X_train[:, selected_features_indices]

param_grid = {
    'classification__ram': list(range(binarization__thermometer, 64, binarization__thermometer))
}

print('Otimização começando')

# Criação do pipeline com binarização sob demanda
pipeline = Pipeline([
    ('binarization', BinarizationTransformer(thermometer=binarization__thermometer)),
    ('classification', CustomWiSARDClassifier(thermometer=4, ram=4))
])

# Criação do modelo WiSARD com GridSearchCV
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring=make_scorer(custom_scorer))
grid_search.fit(X_train, y_train)  # X_train e y_train são os dados completos de treinamento

# Converte os resultados para um DataFrame do pandas
results_df = pd.DataFrame(grid_search.cv_results_)

# Adiciona as informações adicionais que você deseja incluir
results_df['min_perct_inclass'] = min_inclass
results_df['max_perct_outclass'] = max_outclass

# Melhores parâmetros e resultados
print("Melhores parâmetros:", grid_search.best_params_)
print("Melhor score:", grid_search.best_score_)

# Salva os resultados em um arquivo CSV
results_df.to_csv(output_file_name, index=False)
