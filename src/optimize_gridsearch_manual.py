import json
import pandas as pd
import numpy as np
import scipy.sparse
import wisardpkg as wsd
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def binarize_vectorized(thermometer, X):

    num_features = X.shape[1]
    thermometer_sizes = [thermometer] * num_features

    # Converter para formato denso se X for uma matriz esparsa
    if isinstance(X, scipy.sparse.spmatrix):
        X = X.toarray()

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    dtherm = wsd.DynamicThermometer(thermometer_sizes, mins, maxs)

    # Aplica a transformação em todo o conjunto de dados de uma vez
    binX = [dtherm.transform(X[i]) for i in range(len(X))]
    return binX

def split_train_test(tfidf_matrix, labels, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, stratify=labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def divisors_and_multiples(n, max_value):
    divisors = [i for i in range(4, max_value + 1) if n % i == 0]
    multiples = [i for i in range(n, max_value + 1, n)]
    combined = list(set(divisors + multiples))  # Combina e remove duplicatas
    combined.sort()  # Ordena a lista
    return combined
#modifica as configurações padrão considerando as configurações passadas pelo usuário
vocabulary_file = '/home/gssilva/datasets/atribuna-elias/full/selection/vocabulary_0.60-0.20_not_centroid.json'
train_folder = '/home/gssilva/datasets/atribuna-elias/full/train-test'
output_file_name = '/home/gssilva/datasets/atribuna-elias/full/results/otimizacao_gridsearch_manual_full_128therm.csv'
min_inclass = 0.6
max_outclass = 0.2
n_procs = 5


X_train = load_npz(f'{train_folder}/X_train.npz')
X_train = X_train.tocsc()
y_train = np.load(f'{train_folder}/y_train.npy', allow_pickle=True)

with open(vocabulary_file, 'r') as file:
    vocabulary = json.load(file)

with open(vocabulary_file, 'r') as file:
    vocabulary = json.load(file)
selected_features_indices = list(vocabulary.values())

thermometers = list(range(4, 129))
for binarization_thermometer in thermometers:
    transformed_train = X_train[:, selected_features_indices]
    X_bin = binarize_vectorized(binarization_thermometer, transformed_train)
    print("done binarization!")

    rams = divisors_and_multiples(binarization_thermometer, 65)
    params = {'thermometer': [],'ram': [], 'min_inclass': [], 'max_outclass': [], 'acuracy': []}
    best_params = {'thermometer': 0,'ram': 0, 'min_inclass': 0, 'max_outclass': 0, 'acuracy': 0}

    for ram in rams:
        X_train_fold, X_validation, y_train_fold, y_validation = split_train_test(X_bin, y_train)

        ds_train = wsd.DataSet(X_train_fold, y_train_fold)
        ds_val = wsd.DataSet(X_validation, y_validation)

        model = wsd.ClusWisard(ram, 0.2, 1000, 10)
        model.train(ds_train)
        predicted = model.classify(ds_val)
        average_score = accuracy_score(y_validation, predicted)

        print(f'Parâmetros: therm:{binarization_thermometer} e ram: {ram}, Acurácia: {average_score}')
        params['thermometer'].append(binarization_thermometer)
        params['ram'].append(ram)
        params['min_inclass'].append(min_inclass)
        params['max_outclass'].append(max_outclass)
        params['acuracy'].append(average_score)

        if(average_score > best_params['acuracy']):
            best_params['thermometer'] = binarization_thermometer
            best_params['ram'] = ram
            best_params['min_inclass'] = min_inclass
            best_params['max_outclass'] = max_outclass
            best_params['acuracy'] = average_score

for key in params:
    params[key].append(best_params[key])

df_out = pd.DataFrame(params)
df_out.to_csv(output_file_name, index=False)





