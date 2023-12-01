import numpy as np
import wisardpkg as wsd
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna
import pandas as pd
import json
from scipy.sparse import load_npz

def split_train_test(tfidf_matrix, labels, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, stratify=labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

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

def objective_wisard(trial):
    thermometer = trial.suggest_int('thermometer', 4, 64)

    max_ram = 64 - (64 % thermometer)  # Ajusta o máximo para ser divisível pelo valor de 'thermometer'
    ram = trial.suggest_int('ram', thermometer, max_ram, step=thermometer)

    params = {
        'thermometer': thermometer,
        'ram': ram,
        'random_state': 42
    }

    # Binarização tanto para treino quanto para validação
    BinX = binarize_vectorized(params['thermometer'], transformed_train)

    # Divisão de treino e validação
    X_train, X_val, y_train, y_val = split_train_test(BinX, labels, 0.25, params['random_state'])

    # Preparação e treinamento do modelo WiSARD
    ds_train = wsd.DataSet(X_train, y_train)
    ds_val = wsd.DataSet(X_val, y_val)

    model = wsd.ClusWisard(params['ram'], 0.2, 100, 5)
    model.train(ds_train)
    predicted = model.classify(ds_val)

    # Cálculo da acurácia
    average_score = accuracy_score(y_val, predicted)

    # Log apenas da acurácia e dos parâmetros atuais
    print(f'Parâmetros: {params}, Acurácia: {average_score}')

    return average_score

OPTUNA_EARLY_STOPING = 250  # number of stagnation iterations required to raise an EarlyStoppingExceeded exception

class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    early_stop = OPTUNA_EARLY_STOPING
    early_stop_count = 0
    best_score = None

def early_stopping_opt(study, trial):
    if EarlyStoppingExceeded.best_score == None:
        EarlyStoppingExceeded.best_score = study.best_value

    if study.best_value > EarlyStoppingExceeded.best_score:
        EarlyStoppingExceeded.best_score = study.best_value
        EarlyStoppingExceeded.early_stop_count = 0
    else:
        if EarlyStoppingExceeded.early_stop_count > EarlyStoppingExceeded.early_stop:
            EarlyStoppingExceeded.early_stop_count = 0
            best_score = None
            raise EarlyStoppingExceeded()
        else:
            EarlyStoppingExceeded.early_stop_count = (
                EarlyStoppingExceeded.early_stop_count + 1
            )
    return


#modifica as configurações padrão considerando as configurações passadas pelo usuário
vocabulary_file = '/home/gssilva/datasets/atribuna-site/full/selections/vocabulary_0.60-0.20.json'
train_folder = '/home/gssilva/datasets/atribuna-site/full/train_test'
output_file_name = '/home/gssilva/datasets/atribuna-site/full/results/otimizacao_optuna.csv'
min_inclass = 0.60
max_outclass = 0.25
n_trials = 50
n_procs = 15

X_train = load_npz(f'{train_folder}/X_train.npz')
X_train = X_train.tocsc()
labels = np.load(f'{train_folder}/y_train.npy', allow_pickle=True)

with open(vocabulary_file, 'r') as file:
    vocabulary = json.load(file)
selected_features_indices = list(vocabulary.values())

# Selecionar características com base no vocabulário
transformed_train = X_train[:, selected_features_indices]

sampler = optuna.samplers.TPESampler(seed=42)
pruner = optuna.pruners.HyperbandPruner()

study_wisard = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
try:
    study_wisard.optimize(
        objective_wisard, n_trials=5000, callbacks=[early_stopping_opt], n_jobs=n_procs)
except EarlyStoppingExceeded:
    print(f"EarlyStopping Exceeded: No new best scores in {OPTUNA_EARLY_STOPING} iterations")


study_results = study_wisard.trials_dataframe()
study_results['min_perct_inclass'] = min_inclass
study_results['max_perct_outclass'] = max_outclass
study_results.to_csv(output_file_name, index=False)
