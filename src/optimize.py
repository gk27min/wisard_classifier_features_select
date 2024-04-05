import random
import optuna
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from scipy.sparse import load_npz
from utils import apply_svd
from classifiers_algoritms import WisardClassifier, SVMClassifier, KNNClassifier

# Constantes e Parâmetros
DEFAULT_RANDOM_STATE = 42
DEFAULT_LABELS_COLUMN = 'LABEL'
DEFAULT_DATA_FILE = '/home/gssilva/datasets/atribuna-elias/bin/bin_62.npz'
DEFAULT_LABELS_FILE = '/home/gssilva/datasets/atribuna-elias/preprocessed_aTribuna.csv'
DEFAULT_N_TRIALS = 100
DEFAULT_N_SPLITS = 5
DEFAULT_N_COMPONENTS = 100

def optimize_with_gridsearch(model, params, train_data, train_labels):
    grid_search = GridSearchCV(model, params, cv=DEFAULT_N_SPLITS, verbose=1, n_jobs=-1)
    grid_search.fit(train_data, train_labels)
    return grid_search.best_params_

def optimize_with_optuna(model, params, train_data, train_labels):
    def objective(trial):
        for param_name, param_range in params.items():
            if isinstance(param_range[0], float):
                setattr(model, param_name, trial.suggest_float(param_name, *param_range))
            elif isinstance(param_range[0], int):
                setattr(model, param_name, trial.suggest_int(param_name, *param_range))
            else:
                setattr(model, param_name, trial.suggest_categorical(param_name, param_range))

        model.fit(train_data, train_labels)
        return model.score(train_data, train_labels)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=DEFAULT_N_TRIALS)
    return study.best_params

def split_data_with_cross_validation(data, labels):
    kf = KFold(n_splits=DEFAULT_N_SPLITS, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    return [((data[train_index], labels[train_index]), (data[test_index], labels[test_index]))
            for train_index, test_index in kf.split(data)]

def main():
    data = load_npz(DEFAULT_DATA_FILE)
    df = pd.read_csv(DEFAULT_LABELS_FILE)
    labels = df[DEFAULT_LABELS_COLUMN].to_numpy()

    train_test_splits = split_data_with_cross_validation(data, labels)

    svm_params_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    svm_params_optuna = {'C': (0.1, 10), 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}

    knn_params_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}
    knn_params_optuna = {'n_neighbors': (3, 10), 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}

    wisard_params_grid = {'ram': [4, 8, 16, 24, 32, 40, 62, 64], 'min_score': [0.3, 0.5, 0.7], 'threshold': [0.05, 0.1, 0.2, 1000], 'discriminator_limit': [5, 10, 15]}
    wisard_params_optuna = {'ram': (4, 64), 'min_score': (0.3, 0.7), 'threshold': (0.05, 0.2), 'discriminator_limit': (5, 15)}

    models = {
        'SVM': SVMClassifier(),
        'KNN': KNNClassifier(),
        'WISARD': WisardClassifier()
    }

    best_params = {model_name: {'GridSearch': [], 'Optuna': []} for model_name in models.keys()}

    # for model_name, model in models.items():
    for (train_data, train_labels), (test_data, _) in train_test_splits:
        train_data_transformed = apply_svd(train_data, DEFAULT_N_COMPONENTS, _, gerete_img=False)
        best_params['SVM']['GridSearch'].append(optimize_with_gridsearch(SVMClassifier(), eval(f"{'SVM'.lower()}_params_grid"), train_data_transformed, train_labels))
        best_params['SVM']['Optuna'].append(optimize_with_optuna(SVMClassifier(), eval(f"{'SVM'.lower()}_params_optuna"), train_data_transformed, train_labels))

    # for model_name, params in best_params.items():
    #     print(f"Best {model_name} params:")
    #     print(f"GridSearch: {params['GridSearch']}")
    #     print(f"Optuna: {params['Optuna']}")
    #     print('-' * 50)
    print(f"Best {'SVM'} params:")
    print(f"GridSearch: {best_params['GridSearch']}")
    print(f"Optuna: {best_params['Optuna']}")
    print('-' * 50)

if __name__ == "__main__":
    main()
