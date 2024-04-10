import random
import pandas as pd
import numpy as np
from utils import apply_svd, generate_heatmap, evaluate_classification
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from classifiers_algoritms import WisardClassifier, SVMClassifier, KNNClassifier

# Constantes e Parâmetros
RANDOM_STATE = 42
LABELS_COLUMN = 'LABEL'
DATA_FILE = '/home/gssilva/datasets/atribuna-elias/full/vect_aTribuna.npz'
LABELS_FILE = '/home/gssilva/datasets/atribuna-elias/full/preprocessed_aTribuna-Elias.csv'
IMG_DISC = '/home/gssilva/datasets/atribuna-elias/full/results/imagens/full_discriminators_svd100.png'
IMG_SVD = '/home/gssilva/datasets/atribuna-elias/full/results/imagens/svd100.png'
N_COMPONENTS = 100

random.seed(RANDOM_STATE)

data = load_npz(DATA_FILE)
labels = pd.read_csv(LABELS_FILE)[LABELS_COLUMN].to_numpy()
labels_unique = np.sort(np.unique(labels))

X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.25, random_state=RANDOM_STATE)

print('split train/test data...')
print(f'Aply svd model on the data and save svd curve of \'Explained Variance Ratio of SVD\' ...')

X_train = apply_svd(X_train, N_COMPONENTS, IMG_SVD, True)

models = {
    'SVM': SVMClassifier(kernel='rbf', C=10, gamma='scale'),
    'KNN': KNNClassifier(n_neighbors=7, weights='uniform', algorithm='auto'),
    'Wisard': WisardClassifier(ram=62, min_score=0.5, threshold=1000, discriminator_limit=5)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predition = model.predict(X_test)
    if isinstance(model, WisardClassifier):
        discriminators = model.getMentalImages()
        generate_heatmap(discriminators, IMG_DISC)
        print('Discriminators Images done....')
    print(f"{model_name} predictions:\n")
    print(evaluate_classification(y_test, predition))
