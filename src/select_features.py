import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import json
from scipy.sparse import load_npz, save_npz, csr_matrix, vstack
from sklearn.metrics.pairwise import cosine_distances

def select_features_parallel(matrix, classes_array, word_dict, min_perct_inclass, max_perct_outclass):
    if not isinstance(matrix, csr_matrix):
        raise ValueError("A matriz fornecida deve ser do tipo csr_matrix.")

    unique_classes = np.unique(classes_array)
    selected_features = []
    selected_words_dict = {}

    # A matriz de presença de features é calculada diretamente sobre a matriz esparsa
    feature_presence = matrix > 0

    for class_value in unique_classes:
        in_class_indices = classes_array == class_value
        out_class_indices = ~in_class_indices

        # As operações abaixo são otimizadas para matrizes esparsas
        percent_in_class = feature_presence[in_class_indices].mean(axis=0).A  # Converte para NumPy array
        percent_out_class = feature_presence[out_class_indices].mean(axis=0).A  # Converte para NumPy array

        # A operação de comparação é vetorizada
        mask = (percent_in_class >= min_perct_inclass) & (percent_out_class <= max_perct_outclass)
        mask = mask.ravel()  # Transforma em um array unidimensional

        for word, idx in word_dict.items():
            if mask[idx]:
                selected_features.append(idx)
                selected_words_dict[word] = idx

    # Utiliza indexação avançada 
    # para criar a matriz reduzida
    reduced_matrix = matrix[:, selected_features]
    return reduced_matrix, selected_features, selected_words_dict

def split_train_test(tfidf_matrix, labels, test_size=0.25): 
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, stratify=labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test 

def centroid_calc():
    centroids = {}
    for label in set(labels_uniq):
        # Índices dos documentos que pertencem a esta classe
        indices = [i for i, l in enumerate(labels) if l == label]
        
        # Calculando a média dos vetores TF-IDF para esses documentos
        centroids[label] = np.mean(vectorized[indices].toarray(), axis=0)
    return centroids

def most_distants_docs():
    most_distant_docs = []
    labels_output = []
    indices_output = []

    # Process each class
    for label in set(labels):
        # Indices of documents belonging to this class
        indices = [i for i, l in enumerate(labels) if l == label]

        # If the class has fewer than 2 documents, skip
        if len(indices) < 2:
            continue

        # Calculate cosine distances between all pairs of documents in this class
        distances = cosine_distances(vectorized[indices])

        # Find the pair of documents with the largest distance
        most_distant_pair_indices = np.unravel_index(np.argmax(distances), distances.shape)

        # Add the most distant documents to the list
        doc1_index, doc2_index = most_distant_pair_indices
        most_distant_docs.append(vectorized[indices[doc1_index]])
        most_distant_docs.append(vectorized[indices[doc2_index]])
        labels_output.extend([label] * 2)
        indices_output.extend([indices[doc1_index], indices[doc2_index]])

    return vstack(most_distant_docs), labels_output, indices_output

# Carregando dados e rótulos
vectorized_file = '/home/gssilva/datasets/atribuna-elias/full/vectorized_aTribuna.npz'
vocabulary_file = '/home/gssilva/datasets/atribuna-elias/full/vocabulary.json'
output_folder = '/home/gssilva/datasets/atribuna-elias/full/selection'
min_inclass = 0.60
max_inclass = 1.05
min_outclass = 0.20
max_outclass = 0.45
path_file_labels = '/home/gssilva/datasets/atribuna-elias/full/preprocessed_aTribuna-Elias.csv'
train_folder = '/home/gssilva/datasets/atribuna-elias/full/train-test'
labels_column = 'LABEL'
centroid_cal = False
distants = False

vectorized = load_npz(vectorized_file)
df = pd.read_csv(path_file_labels)
labels = df[labels_column].to_numpy()  # Convertendo para NumPy array3
labels_uniq = set(labels)

with open(vocabulary_file, 'r') as file:
    vocabulary = json.load(file)

# Divisão Treino-Teste
X_train, X_test, y_train, y_test = split_train_test(vectorized, labels)
train_indices = X_train.indices  # Obter índices dos documentos de treino

if centroid_cal:
    centroids = centroid_calc()
    centroid_matrix = np.array(list(centroids.values()))
    centroid_sparse_matrix = csr_matrix(centroid_matrix)
    X_train = vstack([X_train, centroid_sparse_matrix])
    centroid_labels = np.array(list(centroids.keys()))  # Convertendo para NumPy array
    y_train = np.concatenate([y_train, centroid_labels])  # Concatenação correta

if distants:
    most_distants, labels_distants, indices = most_distants_docs()
    for idx, doc in enumerate(most_distants):
        if indices[idx] not in train_indices:
            X_train = vstack([X_train, doc])
            y_train = np.append(y_train, labels_distants[idx])



# Salvando matrizes e rótulos divididos
save_npz(f'{train_folder}/X_train.npz', X_train)
save_npz(f'{train_folder}/X_test.npz', X_test)
np.save(f'{train_folder}/y_train.npy', y_train)
np.save(f'{train_folder}/y_test.npy', y_test)
print("Divisão Treino e Teste realizada!")

# Iterando sobre parâmetros para seleção de características
for i in np.arange(min_inclass, max_inclass, 0.05):  # Ajuste o passo conforme necessário
    for j in np.arange(min_outclass, max_outclass, 0.05):  # Ajuste o passo conforme necessário
        _, __, new_vocabulary = select_features_parallel(X_train, y_train, vocabulary, i, j)
        file_name = f'{output_folder}/vocabulary_{i:.2f}-{j:.2f}_not_centroid.json'.replace("'", "")
        with open(file_name, 'w') as file:
            json.dump(new_vocabulary, file, indent=4)
