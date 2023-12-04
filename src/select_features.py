import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import json
from scipy.sparse import load_npz, save_npz, csr_matrix, vstack
from sklearn.metrics.pairwise import cosine_distances

def calculate_intra_class_frequencies(matrix, classes_array):
    if not isinstance(matrix, csr_matrix):
        raise ValueError("A matriz fornecida deve ser do tipo csr_matrix.")

    unique_classes = np.unique(classes_array)
    num_features = matrix.shape[1]
    intra_class_frequencies = np.zeros((len(unique_classes), num_features))

    # A matriz de presença de features é calculada diretamente sobre a matriz esparsa
    feature_presence = matrix > 0

    for class_idx, class_value in enumerate(unique_classes):
        in_class_indices = classes_array == class_value
        
        # Calcula a frequência da feature na classe atual
        percent_in_class = feature_presence[in_class_indices].mean(axis=0).A1  # Converte para NumPy array unidimensional
        intra_class_frequencies[class_idx, :] = percent_in_class

    return intra_class_frequencies

def select_features(importance_matrix, word_dict, min_intra_class_freq, min_exclusivity):
    num_classes, num_features = importance_matrix.shape
    selected_features = []

    # Calcula a importância de cada feature para cada classe
    for feature_idx in range(num_features):
        feature_freqs = importance_matrix[:, feature_idx]

        # Calcula a frequência média da feature nas outras classes
        avg_freq_other_classes = (np.sum(feature_freqs) - feature_freqs) / (num_classes - 1)

        # Calcula a exclusividade da feature
        exclusivity = 1 - avg_freq_other_classes

        selected_words_dict = {}
        
        # Verifica se a feature é suficientemente frequente e exclusiva em alguma classe
        if np.any(feature_freqs >= min_intra_class_freq) and np.any(exclusivity >= min_exclusivity):
            selected_features.append(feature_idx)

    for word, idx in word_dict.items():
        if idx in selected_features:
            selected_words_dict[word] = idx

    return selected_features, selected_words_dict

def select_features_parallel(matrix, classes_array, word_dict, min_intra_class_freq, min_exclusivity):
    intra_class_frequencies = calculate_intra_class_frequencies(matrix, classes_array)

    # min_intra_class_freq = 0.6  # Frequência mínima dentro da classe
    # min_exclusivity = 0.7  # Medida mínima de exclusividade
    features_to_keep, selected_words_dict = select_features(intra_class_frequencies, word_dict, min_intra_class_freq, min_exclusivity)

    reduced_matrix = matrix[:, features_to_keep]
    return reduced_matrix, features_to_keep, selected_words_dict

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
vectorized_file = '/home/gssilva/datasets/atribuna-elias/full/vectorized_aTribuna_test.npz'
vocabulary_file = '/home/gssilva/datasets/atribuna-elias/full/vocabulary_test.json'
output_folder = '/home/gssilva/datasets/atribuna-elias/full/selection'
min_inclass = 0.50
max_inclass = 1.05
min_outclass = 0.45
max_outclass = 0.80
path_file_labels = '/home/gssilva/datasets/atribuna-elias/full/preprocessed_aTribuna-Elias.csv'
train_folder = '/home/gssilva/datasets/atribuna-elias/full/train-test'
labels_column = 'LABEL'
centroid_cal = False
distants = False

vectorized = load_npz(vectorized_file)
df = pd.read_csv(path_file_labels)

labels_to_keep = [label for label, count in df['LABEL'].value_counts().items() if count > 4000]
df = df[df['LABEL'].isin(labels_to_keep)]

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
        
        print(f'- in:{i};\tout:{j};\tlen:{len(new_vocabulary.keys())}')
        
        file_name = f'{output_folder}/vocabulary_{i:.2f}-{j:.2f}_matrix_frequencia_test.json'.replace("'", "")
        with open(file_name, 'w') as file:
            json.dump(new_vocabulary, file, indent=4)
