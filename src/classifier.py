
import json
import wisardpkg as wsd
import pandas as pd
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from scipy.sparse import load_npz, csr_matrix, vstack
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss

random_state = 42
labels_column = 'LABEL'
centroid_cal = True
most_dist = True
most_close = True
fixed_train = []

def centroid_calc():
    centroids = {}
    for label in set(labels_uniq):
        # Índices dos documentos que pertencem a esta classe
        indices = [i for i, l in enumerate(labels) if l == label]
        
        # Calculando a média dos vetores TF-IDF para esses documentos
        centroids[label] = np.mean(vectorized[indices].toarray(), axis=0)
    return csr_matrix(centroids.values()), centroids.keys()

def most_distants_docs():
    most_distant_docs = []
    labels_output = []
    indices_output = []

    # Process each class
    for label in set(labels):
        # Indices of documents belonging to this class
        indices = [i for i, l in enumerate(labels) if l == label]

        # Calculate cosine distances between all pairs of documents in this class
        distances = cosine_distances(vectorized[indices])

        # Mask out the diagonal (distance of a document with itself)
        np.fill_diagonal(distances, np.nan)

        if len(indices) >= 5:
            # Flatten the distance matrix and sort by distance, descending
            flat_distances = distances.flatten()
            sorted_indices = np.argsort(-flat_distances)

            # Calculate the number of documents to select (20%)
            num_docs_to_select = int(len(sorted_indices) * 0.2)

            # Select the top 20% distances
            selected_indices = sorted_indices[:num_docs_to_select]

            # Convert flat indices back to 2D indices
            selected_2d_indices = np.unravel_index(selected_indices, distances.shape)

            # Add the most distant documents to the list
            for idx_pair in zip(*selected_2d_indices):
                most_distant_docs.append(vectorized[indices[idx_pair[0]]])
                labels_output.append(label)
                indices_output.append(indices[idx_pair[0]])

        else:
            # Find the pair of documents with the largest distance if less than 5 documents
            if len(indices) >= 2:
                most_distant_pair_indices = np.unravel_index(np.nanargmax(distances), distances.shape)
                doc1_index, doc2_index = most_distant_pair_indices
                most_distant_docs.append(vectorized[indices[doc1_index]])
                most_distant_docs.append(vectorized[indices[doc2_index]])
                labels_output.extend([label] * 2)
                indices_output.extend([indices[doc1_index], indices[doc2_index]])

    return vstack(most_distant_docs), labels_output, indices_output

def most_close_docs():
    most_close_docs = []
    labels_output = []
    indices_output = []

    # Process each class
    for label in set(labels):
        # Indices of documents belonging to this class
        indices = [i for i, l in enumerate(labels) if l == label]

        # Calculate cosine distances between all pairs of documents in this class
        distances = cosine_distances(vectorized[indices])

        # Mask out the diagonal (distance of a document with itself)
        np.fill_diagonal(distances, np.nan)

        if len(indices) >= 5:
            # Flatten the distance matrix and sort by distance, ascending
            flat_distances = distances.flatten()
            sorted_indices = np.argsort(flat_distances)

            # Calculate the number of documents to select (20%)
            num_docs_to_select = int(len(sorted_indices) * 0.2)

            # Select the top 20% closest distances
            selected_indices = sorted_indices[:num_docs_to_select]

            # Convert flat indices back to 2D indices
            selected_2d_indices = np.unravel_index(selected_indices, distances.shape)

            # Add the most close documents to the list
            for idx_pair in zip(*selected_2d_indices):
                most_close_docs.append(vectorized[indices[idx_pair[0]]])
                labels_output.append(label)
                indices_output.append(indices[idx_pair[0]])

        else:
            # Find the pair of documents with the smallest distance if less than 5 documents
            if len(indices) >= 2:
                most_close_pair_indices = np.unravel_index(np.nanargmin(distances), distances.shape)
                doc1_index, doc2_index = most_close_pair_indices
                most_close_docs.append(vectorized[indices[doc1_index]])
                most_close_docs.append(vectorized[indices[doc2_index]])
                labels_output.extend([label] * 2)
                indices_output.extend([indices[doc1_index], indices[doc2_index]])

    return vstack(most_close_docs), labels_output, indices_output

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

def split_train_test(tfidf_matrix, labels, test_size=0.25, fixed_train_indices=None):
    if fixed_train_indices is not None:
        # Separar documentos e rótulos fixos
        fixed_X_train = tfidf_matrix[fixed_train_indices]
        fixed_y_train = [labels[i] for i in fixed_train_indices]

        # Remover documentos fixos do conjunto original
        mask = np.ones(len(labels), dtype=bool)
        mask[fixed_train_indices] = False
        remaining_X = tfidf_matrix[mask]
        remaining_y = np.array(labels)[mask]

        # Dividir os dados restantes
        X_train, X_test, y_train, y_test = train_test_split(
            remaining_X, remaining_y, stratify=remaining_y, test_size=test_size, random_state=42)

        # Juntar os documentos fixos ao conjunto de treino
        X_train = vstack([fixed_X_train, X_train])
        y_train = np.concatenate([fixed_y_train, y_train])
    else:
        # Dividir os dados normalmente se nenhum índice fixo for fornecido
        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_matrix, labels, stratify=labels, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

def evaluate_classification(predicted, expected):
    # Calculando as métricas
    accuracy = accuracy_score(expected, predicted)
    precision = precision_score(expected, predicted, average='binary')
    recall = recall_score(expected, predicted, average='binary')
    f1 = f1_score(expected, predicted, average='binary')
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
    """
    Generates a heatmap of the average directional vectors for each category
    and saves it to a file.

    :param data: Dictionary with categories as keys and lists of vectors as values.
    :param file_name: The name of the file to save the heatmap image.
    """
    # Calculating the average directional vector for each category
    mean_vectors = {category: np.mean(np.array(vectors), axis=0) for category, vectors in data.items()}

    # Converting the average vectors into a matrix for the heatmap
    mean_activation_matrix = np.array(list(mean_vectors.values()))

    # Creating the heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(mean_activation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Average Activation Level')

    # Adding labels for categories
    plt.yticks(ticks=range(len(mean_vectors)), labels=mean_vectors.keys())
    plt.xticks(ticks=range(len(next(iter(mean_vectors.values())))), labels=range(1, len(next(iter(mean_vectors.values()))) + 1))

    plt.title('Heatmap of Average Directional Vector by Category')
    plt.xlabel('Position in Vector')
    plt.ylabel('Category')

    # Saving the heatmap to a file
    plt.savefig(file_name)
    plt.close()

vectorized_file = '/home/gssilva/datasets/atribuna-elias/full/vectorized_aTribuna.npz'
vocabulary_file = '/home/gssilva/datasets/atribuna-elias/full/selection/vocabulary_0.60-0.20_not_centroid.json'
path_file_labels = '/home/gssilva/datasets/atribuna-elias/full/preprocessed_aTribuna-Elias.csv'
output_img = '/home/gssilva/datasets/atribuna-elias/full/results/imagens/full_discriminators.png'

vectorized = load_npz(vectorized_file)
df = pd.read_csv(path_file_labels)
labels = df[labels_column].to_numpy()  # Convertendo para NumPy array3
labels_uniq = set(labels)

with open(vocabulary_file, 'r') as file:
    vocabulary = json.load(file)
selected_features = list(vocabulary.values())

vectorized = vectorized[:, selected_features]

print('Getting all inicial files and selecting all features...')

params = {'thermometer': 62,'ram': 62}

if centroid_cal:
    centroids, labels_centroids = centroid_calc()
    centroids_bin = binarize_vectorized(params['thermometer'], centroids)

if most_dist:
    _, _, indices = most_distants_docs()
    fixed_train.extend(indices)

if most_close:
    _, _, indices = most_close_docs()
    fixed_train.extend(indices)

print('Centroid, most distants, most close documents calculate...')

bin_x = binarize_vectorized(params['thermometer'], vectorized)
bin_train, bin_test, y_train, y_test = split_train_test(bin_x, labels, 0.25, fixed_train)
print('binarized and split all data done...')

if centroid_cal:
    bin_train = vstack(bin_x, centroids_bin)
    y_train = np.concatenate([y_train, labels_centroids])

ds_train = wsd.DataSet(bin_train, y_train)
ds_test = wsd.DataSet(bin_test, y_test)

model = wsd.ClusWisard(params['ram'], 0.2, 100, 5)
model.train(ds_train)

discriminators = model.getMentalImages()
generate_heatmap(discriminators, output_img)

predicted = model.classify(ds_test)

score = evaluate_classification
print(score)