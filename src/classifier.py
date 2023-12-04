
import json
import wisardpkg as wsd
import pandas as pd
import numpy as np
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
centroid_cal = False
most_dist = False
most_close = False
fixed_train = []

random.seed(random_state)

def select_train_base(labels, matrix):
    indices_train = []
    indices_test = []
    labels_unique, counts = np.unique(labels, return_counts=True)

    for label, num_docs in zip(labels_unique, counts):
        # Define a porcentagem de treino baseada no número de documentos
        if num_docs < 300:
            percent_train = 0.70
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

def centroid_calc():
    centroids = {}
    for label in set(labels_uniq):
        # Índices dos documentos que pertencem a esta classe
        indices = [i for i, l in enumerate(labels) if l == label]
        
        # Calculando a média dos vetores TF-IDF para esses documentos
        centroids[label] = np.mean(vectorized[indices].toarray(), axis=0)
        valores = list(centroids.values())
        matriz = np.array(valores)
    return csr_matrix(matriz), list(centroids.keys())

def most_distants_docs():
    most_distant_indices = []

    for label in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label]

        # Calcular as distâncias de cosseno entre os documentos
        distances = cosine_distances(vectorized[indices])

        # Binarizar as distâncias (manter apenas 0s e 1s)
        binary_distances = np.where(distances > 0, 1, 0)

        # Encontrar os índices dos documentos mais distantes (20%)
        num_docs_to_select = int(binary_distances.shape[1] * 0.5)
        selected_indices = np.argsort(binary_distances.sum(axis=1))[-num_docs_to_select:]

        # Mapear os índices selecionados de volta para os índices originais
        selected_indices = [indices[i] for i in selected_indices]

        # Adicionar os índices dos documentos mais distantes à lista
        most_distant_indices.extend(selected_indices)

    return most_distant_indices

def most_close_docs():
    most_close_indices = []

    for label in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label]

        # Calcular as distâncias de cosseno entre os documentos
        distances = cosine_distances(vectorized[indices])

        # Binarizar as distâncias (manter apenas 0s e 1s)
        binary_distances = np.where(distances > 0, 1, 0)

        # Encontrar os índices dos documentos mais próximos (20%)
        num_docs_to_select = int(binary_distances.shape[1] * 0.2)
        selected_indices = np.argsort(binary_distances.sum(axis=1))[:num_docs_to_select]

        # Mapear os índices selecionados de volta para os índices originais
        selected_indices = [indices[i] for i in selected_indices]

        # Adicionar os índices dos documentos mais próximos à lista
        most_close_indices.extend(selected_indices)

    return most_close_indices

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

print('Getting all inicial files and selecting all features...')

params = {'thermometer': 62,'ram': 62}

if centroid_cal:
    centroids, labels_centroids = centroid_calc()
    centroids_bin = binarize_vectorized(params['thermometer'], centroids)
    vectorized.extend(centroids_bin)
    labels.extend(labels_centroids)
    print('Centroids calculate')

if most_dist:
    indices = most_distants_docs()
    fixed_train.extend(indices)
    print('More distantes calculate')

if most_close:
    indices = most_close_docs()
    fixed_train.extend(indices)
    print('Centroid, most distants, most close documents calculate...')

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
bin_train, bin_test, y_train, y_test = select_train_base(labels, bin_x)
print('binarized and split all data done...')

ds_train = wsd.DataSet(bin_train, y_train)
ds_test = wsd.DataSet(bin_test, y_test)
print('Dataset Wisard Concluide...')

model = wsd.ClusWisard(params['ram'], 0.2, 100, 5)
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
