import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# Constantes
DEFAULT_COLUMN_TEXT = 'ABSTRACT'
DEFAULT_CSV_FILE = '/home/gssilva/datasets/atribuna-elias/preprocessed_aTribuna.csv'
DEFAULT_OUTPUT_FILE = '/home/gssilva/datasets/atribuna-elias/vectorized_aTribuna.npz'

def vectorize_tfidf(df, column):
    df[column].fillna("", inplace=True)  # Substitui NaN por string vazia
    if column in df.columns:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])
        return tfidf_vectorizer.vocabulary_, tfidf_matrix
    else:
        raise ValueError(f"A coluna {column} não está presente no DataFrame.")

def main(csv_file=DEFAULT_CSV_FILE, output_file=DEFAULT_OUTPUT_FILE, column_text=DEFAULT_COLUMN_TEXT):
    # Leitura do arquivo CSV
    df = pd.read_csv(csv_file)
    print('DataFrame obtido do CSV')

    # Vetorização TF-IDF
    vocabulary, tfidf_matrix = vectorize_tfidf(df, column_text)
    print("Vetorização TF-IDF concluída")

    # Salvando a matriz TF-IDF esparsa
    save_npz(output_file, tfidf_matrix)
    print("Matriz TF-IDF esparsa salva")

if __name__ == "__main__":
    main()
