import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score
)

def evaluate_classification(predicted, expected):
    accuracy = accuracy_score(expected, predicted)
    
    precision_macro = precision_score(expected, predicted, average='macro')
    recall_macro = recall_score(expected, predicted, average='macro')
    f1_macro = f1_score(expected, predicted, average='macro')
    
    precision_micro = precision_score(expected, predicted, average='micro')
    recall_micro = recall_score(expected, predicted, average='micro')
    f1_micro = f1_score(expected, predicted, average='micro')
    
    precision_weighted = precision_score(expected, predicted, average='weighted')
    recall_weighted = recall_score(expected, predicted, average='weighted')
    f1_weighted = f1_score(expected, predicted, average='weighted')
    
    results = {
        'Accuracy': accuracy,
        
        'Precision Macro': precision_macro,
        'Recall Macro': recall_macro,
        'F1 Score Macro': f1_macro,
        
        'Precision Micro': precision_micro,
        'Recall Micro': recall_micro,
        'F1 Score Micro': f1_micro,
        
        'Precision Weighted': precision_weighted,
        'Recall Weighted': recall_weighted,
        'F1 Score Weighted': f1_weighted,
    }

    return results

def generate_heatmap(data, file_name):
    mean_vectors = {category: np.mean(np.array(vectors), axis=0) for category, vectors in data.items()}
    max_length = max(len(v) for v in mean_vectors.values())
    elements_per_range = max_length // 10 + (max_length % 10 > 0)
    
    grouped_vectors = {
        category: np.array([
            np.mean(vector[i:i + elements_per_range])
            for i in range(0, len(vector), elements_per_range)
        ]) for category, vector in mean_vectors.items()
    }
    
    grouped_matrix = np.array(list(grouped_vectors.values()))
    normalized_matrix = (grouped_matrix - np.min(grouped_matrix)) / (np.max(grouped_matrix) - np.min(grouped_matrix))
    
    plt.figure(figsize=(20, 10))
    sns.heatmap(normalized_matrix, cmap='Blues', annot=False, cbar_kws={'label': 'Average Activation Level'})
    plt.yticks(ticks=np.arange(0.8, len(grouped_vectors)), labels=grouped_vectors.keys(), rotation=0)
    x_labels = [f"[{i}, {min(i + elements_per_range - 1, max_length)}]" for i in range(0, max_length, elements_per_range)]
    plt.xticks(ticks=np.arange(0.8, len(x_labels)), labels=x_labels, rotation=0)
    plt.title('Heatmap of Average Directional Vector by Category')
    plt.xlabel('Window Position in Vector')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def apply_svd(data, n_compts: int, img: str, gerete_img : bool):
    U, Sigma, VT = randomized_svd(data, n_components=n_compts)
    data = U * Sigma

    if gerete_img == True:
        total_variance = np.sum(Sigma**2)
        explained_variance_ratio = (Sigma**2) / total_variance

        plt.plot(explained_variance_ratio)
        plt.title('Explained Variance Ratio of SVD')
        plt.xlabel('Component Number')
        plt.ylabel('Explained Variance Ratio')
        plt.tight_layout()
        plt.savefig(img)
        plt.close()
    return data
