
import json
import wisardpkg as wsd
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.sparse
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import load_npz, csr_matrix, save_npz
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, classification_report
import random


random_state = 42
labels_column = 'LABEL'
centroid_cal = True
most_dist = True
most_close = True
fixed_train = []

random.seed(random_state)

vectorized_file = '/home/gssilva/datasets/atribuna-elias/full/vectorized_aTribuna.npz'
vectorized_file_out = '/home/gssilva/datasets/atribuna-elias/full/vectorized_aTribuna_svd100.npz'
vocabulary_file = '/home/gssilva/datasets/atribuna-elias/full/selection/vocabulary_0.60-0.75_matrix_frequencia.json'
path_file_labels = '/home/gssilva/datasets/atribuna-elias/full/preprocessed_aTribuna-Elias.csv'
output_img = '/home/gssilva/datasets/atribuna-elias/full/results/imagens/full_discriminators_svd100.png'

vectorized = load_npz(vectorized_file)
df = pd.read_csv(path_file_labels)
labels = df[labels_column].to_numpy()  # Convertendo para NumPy array3
labels_uniq = df[labels_column].unique()
labels_uniq.sort()

def remove_documents_by_classes(class_list, csr_matrix, label_vector):
    # Crie um conjunto das classes a serem removidas para verificar a pertinência
    classes_to_remove = set(class_list)

    # Inicialize listas vazias para armazenar os índices e rótulos a serem mantidos
    indices_to_keep = []
    labels_to_keep = []

    # Itere pelos documentos e rótulos, mantendo apenas aqueles que não pertencem às classes a serem removidas
    for i, label in enumerate(label_vector):
        if label not in classes_to_remove:
            indices_to_keep.append(i)
            labels_to_keep.append(label)

    # Crie uma nova matriz CSR e vetor de rótulos com base nos índices mantidos
    csr_matrix_filtered = csr_matrix[indices_to_keep]
    label_vector_filtered = np.array(labels_to_keep)

    return csr_matrix_filtered, label_vector_filtered

def binarize_vectorized(thermometer, X):

    num_features = X.shape[1]
    thermometer_sizes = [thermometer] * num_features

    # Converter para formato denso se X for uma matriz esparsa
    if sp.issparse(X):
        X = X.toarray()

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    dtherm = wsd.DynamicThermometer(thermometer_sizes, mins, maxs)

    # Aplica a transformação em todo o conjunto de dados de uma vez
    binX = [dtherm.transform(X[i]) for i in range(len(X))]
    return binX

def binary_binarize(X):
    binarized_data = []
    for row in X:
        # Verifica se cada valor na linha é maior que zero e converte para int (0 ou 1)
        binarized_row = [int(value > 0) for value in row.data]
        binarized_data.append(binarized_row)
    return binarized_data

def split_train_test(bin_x, labels, test_size=0.25, fixed_train_indices=None, random_state=42):
    if fixed_train_indices is not None:
        total_size = len(bin_x)
        # Calcula o tamanho ajustado do conjunto de teste
        num_fixed = len(fixed_train_indices)
        adjusted_test_size = test_size / (1 - (num_fixed / total_size))

        # Seleciona os elementos correspondentes aos índices fixos para treinamento
        fixed_X_train = [bin_x[i] for i in fixed_train_indices]
        fixed_y_train = [labels[i] for i in fixed_train_indices]

        # Cria uma máscara para excluir os índices fixos do restante dos dados
        mask = np.ones(total_size, dtype=bool)
        mask[fixed_train_indices] = False

        # Divide o restante dos dados de forma estratificada
        X_train_rest, X_test, y_train_rest, y_test = train_test_split(
            [bin_x[i] for i in range(total_size) if mask[i]], 
            [labels[i] for i in range(len(labels)) if mask[i]], 
            stratify=[labels[i] for i in range(len(labels)) if mask[i]], 
            test_size=adjusted_test_size, random_state=random_state)

        # Juntar os conjuntos de treino fixos com os restantes
        X_train = fixed_X_train + X_train_rest
        y_train = fixed_y_train + y_train_rest
    else:
        # Dividir os dados normalmente se nenhum índice fixo for fornecido
        X_train, X_test, y_train, y_test = train_test_split(
            bin_x, labels, stratify=labels, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def evaluate_classification(predicted, expected):
    # Calculando as métricas
    accuracy = accuracy_score(expected, predicted)
    precision = precision_score(expected, predicted, average='macro')
    recall = recall_score(expected, predicted, average='macro')
    f1 = f1_score(expected, predicted, average='macro')
    h_loss = hamming_loss(expected, predicted)

    # Criando um dicionário para armazenar os resultados
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Hamming Loss': h_loss
    }

    return results

def generate_heatmap(data, file_name):
    # Calculando o vetor direcional médio para cada categoria
    mean_vectors = {category: np.mean(np.array(vectors), axis=0) for category, vectors in data.items()}
    
    # Determinar o número de elementos em cada range
    vector_lengths = [len(v) for v in mean_vectors.values()]
    max_length = max(vector_lengths)
    elements_per_range = max_length // 10 + (max_length % 10 > 0)
    
    # Agrupar os vetores em 10 ranges
    grouped_vectors = {
        category: np.array([
            np.mean(vector[i:i + elements_per_range])
            for i in range(0, len(vector), elements_per_range)
        ]) for category, vector in mean_vectors.items()
    }
    
    # Convertendo os vetores agrupados em uma matriz para o heatmap
    grouped_matrix = np.array(list(grouped_vectors.values()))
    
    # Normalizar os dados agrupados para melhorar a visualização
    normalized_matrix = (grouped_matrix - np.min(grouped_matrix)) / (np.max(grouped_matrix) - np.min(grouped_matrix))
    
    # Criando o heatmap com Seaborn
    plt.figure(figsize=(20, 10))
    sns.heatmap(normalized_matrix, cmap='Blues', annot=False, cbar_kws={'label': 'Average Activation Level'})

    # Adicionando rótulos para categorias, ajustando para que fiquem horizontais
    plt.yticks(ticks=np.arange(0.8, len(grouped_vectors)), labels=grouped_vectors.keys(), rotation=0)

    # Ajustando os rótulos do eixo x para mostrar os intervalos
    x_labels = [
        f"[{i}, {min(i + elements_per_range - 1, max_length)}]"
        for i in range(0, max_length, elements_per_range)
    ]
    plt.xticks(ticks=np.arange(0.8, len(x_labels)), labels=x_labels, rotation=0)

    plt.title('Heatmap of Average Directional Vector by Category')
    plt.xlabel('Window Position in Vector')
    plt.ylabel('Category')

    # Salvando o heatmap em um arquivo
    plt.tight_layout()  # Isso garante que os rótulos não sejam cortados
    plt.savefig(file_name)
    plt.close()


print('Getting all inicial files and selecting all features...')

params = {'thermometer': 62,'ram': 62}

# vectorized, labels = remove_documents_by_classes(['imo', 'mic', 'mul', 'tav'], vectorized, labels)
# labels_uniq = list(set(labels))
# labels_uniq.sort()

n_components = 100  # Ajuste conforme necessário

U, Sigma, VT = randomized_svd(vectorized, n_components=n_components)
vectorized_reduced = U * Sigma

save_npz(vectorized_file_out, csr_matrix(vectorized_reduced))

total_variance = np.sum(Sigma**2)
explained_variance_ratio = (Sigma**2) / total_variance

plt.plot(explained_variance_ratio)
plt.title('Explained Variance Ratio of SVD')
plt.xlabel('Component Number')
plt.ylabel('Explained Variance Ratio')
plt.tight_layout()
plt.savefig('/home/gssilva/datasets/atribuna-elias/full/results/imagens/svd100.png')
plt.close()
print('Aply svd model on the data and save svd curve of \'Explained Variance Ratio of SVD\' ...')

bin_x = binarize_vectorized(params['thermometer'], vectorized_reduced)
bin_train, bin_test, y_train, y_test = select_train_base(labels, bin_x, fixed_train)

print('binarized and split all data done...')

ds_train = wsd.DataSet(bin_train, y_train)
ds_test = wsd.DataSet(bin_test, y_test)
print('Dataset Wisard Concluide...')

model = wsd.ClusWisard(params['ram'], 0.5, 1000, 5)
model.train(ds_train)
print('Training model concluide...')

discriminators = model.getMentalImages()
generate_heatmap(discriminators, output_img)
print('Discriminators Imagens done....')

overfiting = model.classify(ds_train)
predicted = model.classify(ds_test)

print('Overfiting:\n')
print(classification_report(y_train, overfiting, target_names=labels_uniq))
print('Test Classification:\n')
print(classification_report(y_test, predicted, target_names=labels_uniq))
