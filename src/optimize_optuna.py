import numpy as np
import wisardpkg as wsd
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna
from scipy.sparse import load_npz
from sklearn.utils.extmath import randomized_svd
import random

random.seed(42)

def split_train_test(tfidf_matrix, labels, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, stratify=labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def select_train_base(labels, matrix):
    indices_train = []
    indices_test = []
    labels_unique, counts = np.unique(labels, return_counts=True)

    for label, num_docs in zip(labels_unique, counts):
        # Define a porcentagem de treino baseada no número de documentos
        if num_docs < 300:
            percent_train = 0.95
        elif num_docs < 500:
            percent_train = 0.70
        elif num_docs < 1000:
            percent_train = 0.85
        elif num_docs < 2000:
            percent_train = 0.75
        else:
            percent_train = 0.60
        
        # Encontra os índices para a classe atual
        class_indices = np.where(labels == label)[0]

        # Divide os índices para treino e teste
        train_indices, test_indices = train_test_split(class_indices, train_size=percent_train)
        
        indices_train.extend(train_indices)
        indices_test.extend(test_indices)

    X_train = [matrix[i] for i in indices_train]
    X_test = [matrix[i] for i in indices_test]

    y_train = [labels[i] for i in indices_train]
    y_test = [labels[i] for i in indices_test]


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
    min_score = trial.suggest_float('min_score', 0, 1, step=0.1)
    threshold = trial.suggest_int('threshold', 100, 1000, 100)

    params = {
        'thermometer': thermometer,
        'ram': ram,
        'random_state': 42,
        'min_score': min_score,
        'threshold': threshold
    }

    # Binarização tanto para treino quanto para validação
    BinX = binarize_vectorized(params['thermometer'], transformed_train)

    # Divisão de treino e validação
    X_train, X_val, y_train, y_val = select_train_base(labels, BinX)

    # Preparação e treinamento do modelo WiSARD
    ds_train = wsd.DataSet(X_train, y_train)
    ds_val = wsd.DataSet(X_val, y_val)

    model = wsd.ClusWisard(params['ram'], params['min_score'], params['threshold'], 5)
    model.train(ds_train)
    predicted = model.classify(ds_val)

    # Cálculo da acurácia
    average_score = f1_score(y_val, predicted,average='macro')

    # Log apenas da acurácia e dos parâmetros atuais
    # print(f'Parâmetros: {params}, F1: {average_score}')

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

n_components = 100  # Ajuste conforme necessário

U, Sigma, VT = randomized_svd(X_train, n_components=n_components)
transformed_train = U * Sigma

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
