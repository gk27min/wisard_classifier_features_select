from sklearn.model_selection import KFold, GridSearchCV
import optuna
from sklearn.metrics import f1_score
from scipy.sparse import load_npz
import random

random_state = 42
random.seed(random_state)
labels_column = 'LABEL'
data_file = '/home/gssilva/datasets/atribuna-elias/full/vectorized_aTribuna.npz'
labels_file = '/home/gssilva/datasets/atribuna-elias/full/preprocessed_aTribuna-Elias.csv'

data = load_npz(data_file)
df = pd.read_csv(labels_file)
labels = df[labels_column].to_numpy()

# Função de otimização com GridSearch
def optimize_with_gridsearch(model, params, train_data, train_labels):
    grid_search = GridSearchCV(model, params, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(train_data, train_labels)
    return grid_search.best_params_

# Função de otimização com Optuna
def optimize_with_optuna(model, params, train_data, train_labels, n_trials=100):
    def objective(trial):
        for param_name, param_range in params.items():
            if isinstance(param_range[0], float):
                setattr(model, param_name, trial.suggest_float(param_name, *param_range))
            elif isinstance(param_range[0], int):
                setattr(model, param_name, trial.suggest_int(param_name, *param_range))
            else:
                setattr(model, param_name, trial.suggest_categorical(param_name, param_range))

        model.fit(train_data, train_labels)
        score = model.score(train_data, train_labels)
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# Função para dividir os dados com validação cruzada
def split_data_with_cross_validation(data, labels, n_splits=5, random_state=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [((data[train_index], labels[train_index]), (data[test_index], labels[test_index]))
            for train_index, test_index in kf.split(data)]

# Divide os dados em conjuntos de treino e teste usando validação cruzada
train_test_splits = split_data_with_cross_validation(data, labels)

# Parâmetros para otimização com GridSearch e Optuna
svm_params_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
svm_params_optuna = {'C': (0.1, 10), 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}

knn_params_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}
knn_params_optuna = {'n_neighbors': (3, 10), 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}

wisard_params_grid = {'ram': [50, 100, 200], 'min_score': [0.3, 0.5, 0.7], 'threshold': [0.05, 0.1, 0.2], 'discriminator_limit': [5, 10, 15]}
wisard_params_optuna = {'ram': (50, 200), 'min_score': (0.3, 0.7), 'threshold': (0.05, 0.2), 'discriminator_limit': (5, 15)}

# Modelos
models = {
    'SVM': SVMClassifier(),
    'KNN': KNNClassifier(),
    'Wisard': WisardClassifier()
}

# Dicionário para armazenar os melhores parâmetros
best_params = {model_name: {'GridSearch': [], 'Optuna': []} for model_name in models.keys()}

# Otimização
for model_name, model in models.items():
    for (train_data, train_labels), (test_data, test_labels) in train_test_splits:
        # GridSearch
        best_params[model_name]['GridSearch'].append(optimize_with_gridsearch(model, eval(f"{model_name.lower()}_params_grid"), train_data, train_labels))
        
        # Optuna
        best_params[model_name]['Optuna'].append(optimize_with_optuna(model, eval(f"{model_name.lower()}_params_optuna"), train_data, train_labels))

# Exibindo os melhores parâmetros
for model_name, params in best_params.items():
    print(f"Best {model_name} params:")
    print(f"GridSearch: {params['GridSearch']}")
    print(f"Optuna: {params['Optuna']}")
    print('-' * 50)
