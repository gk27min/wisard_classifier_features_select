import pandas as pd
import json
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

column_text = 'ABSTRACT'

def vectorize_tfidf(df):
    if column_text in df.columns:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df[column_text])
        return tfidf_vectorizer.vocabulary_, tfidf_matrix
    else:
        raise ValueError(f"A coluna {column_text} não está presente no DataFrame.")

# Localização dos arquivos
origin_csv_file = '/home/gssilva/datasets/atribuna-elias/full/preprocessed_aTribuna-Elias.csv'
output_file_name = '/home/gssilva/datasets/atribuna-elias/full/vectorized_aTribuna_test.npz'
vocabulary_file = '/home/gssilva/datasets/atribuna-elias/full/vocabulary_test.json'

# Leitura do arquivo CSV
df = pd.read_csv(origin_csv_file)
print('DataFrame obtido do CSV')

labels_to_keep = [label for label, count in df['LABEL'].value_counts().items() if count > 4000]

df = df[df['LABEL'].isin(labels_to_keep)]

# Vetorização TF-IDF
vocabulary, tfidf_matrix_sparse = vectorize_tfidf(df)
print("Vetorização TF-IDF concluída")

# Salvando o vocabulário
with open(vocabulary_file, 'w') as f:
    json.dump(vocabulary, f)
print("Vocabulário salvo")

# Salvando a matriz TF-IDF esparsa
save_npz(output_file_name, tfidf_matrix_sparse)
print("Matriz TF-IDF esparsa salva")