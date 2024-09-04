import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import apply_svd, generate_heatmap, evaluate_classification
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from binarizer import thermometer_binarize
from classifiers_algoritms import WisardClassifier, SVMClassifier, KNNClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from concurrent.futures import ProcessPoolExecutor
import os

# Constantes e Parâmetros
RANDOM_STATE = 42
LABELS_COLUMN = 'LABEL'
DATA_FILE = '/home/gssilva/datasets/atribuna-elias/vectorized_aTribuna.npz'
LABELS_FILE = '/home/gssilva/datasets/atribuna-elias/preprocessed_aTribuna.csv'
IMG_DISC = '/home/gssilva/outputs/results/images/fiscriminators.png'
IMG_SVD = '/home/gssilva/outputs/results/images/svd.png'
RESULTS_DIR = './results'
N_COMPONENTS = 100

random.seed(RANDOM_STATE)

data = load_npz(DATA_FILE)
labels = pd.read_csv(LABELS_FILE)[LABELS_COLUMN].to_numpy()
labels_unique = np.sort(np.unique(labels))
data_size = (data.getnnz()) - 1

print(f'Apply svd model on the data and save svd curve of \'Explained Variance Ratio of SVD\' ...')
data = apply_svd(data, N_COMPONENTS, IMG_SVD, True)
data = thermometer_binarize(62, data, data_size)

X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.25, random_state=RANDOM_STATE)
print('split train/test data...')

models = {
    'SVM': SVMClassifier(kernel='rbf', C=10, gamma='scale'),
    'KNN': KNNClassifier(n_neighbors=7, weights='uniform', algorithm='auto'),
    'Wisard': WisardClassifier(ram=62, min_score=0.5, threshold=1000, discriminator_limit=5)
}

# Avaliação padrão
evaluation_results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    if isinstance(model, WisardClassifier):
        discriminators = model.getMentalImages()
        generate_heatmap(discriminators, IMG_DISC)
        print('Discriminators Images done....')
    
    metrics = evaluate_classification(prediction, y_test)
    metrics['Model'] = model_name
    evaluation_results.append(metrics)
    print(f"\n{model_name} predictions:\n")
    print(metrics)

# Salvar resultados da avaliação padrão em CSV
evaluation_df = pd.DataFrame(evaluation_results)
os.makedirs(RESULTS_DIR, exist_ok=True)
evaluation_df.to_csv(os.path.join(RESULTS_DIR, 'evaluation_results.csv'), index=False)

# Função para avaliar modelos com diferentes tamanhos de conjunto de teste
def evaluate_models_with_varying_test_sizes(data, labels, models, initial_test_size, step_size, min_test_size):
    results = {model_name: [] for model_name in models.keys()}
    test_sizes = []

    current_test_size = initial_test_size
    while current_test_size >= min_test_size:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=current_test_size, random_state=RANDOM_STATE)
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            f1 = f1_score(y_test, predictions, average='macro')
            results[model_name].append(f1)
        test_sizes.append(current_test_size * len(labels))
        if current_test_size > 0.1:
            current_test_size -= step_size
        elif current_test_size > 0.01:
            current_test_size -= 0.01
        else:
            current_test_size -= 0.001

    return results, test_sizes

# Avaliar modelos com diferentes tamanhos de conjunto de teste
initial_test_size = 0.25  # 25% do conjunto de dados
step_size = 0.02  # Reduzir em 2% a cada iteração
min_test_size = 0.001  # 0.1% do conjunto de dados

results, test_sizes = evaluate_models_with_varying_test_sizes(data, labels, models, initial_test_size, step_size, min_test_size)

# Plotar os resultados
plt.figure(figsize=(10, 6))
for model_name, f1_scores in results.items():
    plt.plot(test_sizes, f1_scores, label=model_name)
plt.xlabel('Número de Documentos no Conjunto de Teste')
plt.ylabel('F1 Score (Macro)')
plt.title('Desempenho dos Modelos com Diferentes Tamanhos de Conjunto de Teste')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'map_f1_reduced_train.png'))
plt.show()
