import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

# Constantes
DEFAULT_COLUMN_TEXT = 'ABSTRACT'
DEFAULT_CSV_FILE = '/home/gssilva/datasets/atribuna-elias/full/preprocessed_aTribuna-Elias.csv'
DEFAULT_OUTPUT_FILE = '/home/gssilva/datasets/atribuna-elias/full/vectorized_aTribuna_test.npz'

def vectorize_tfidf(df, column):
    if column in df.columns:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])
        return tfidf_vectorizer.vocabulary_, tfidf_matrix
    else:
        raise ValueError(f"A coluna {column} não está presente no DataFrame.")

def main():
    # Argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Vetorização TF-IDF de um arquivo CSV.')
    parser.add_argument('--csv_file', type=str, default=DEFAULT_CSV_FILE, help='Caminho para o arquivo CSV de entrada.')
    parser.add_argument('--output_file', type=str, default=DEFAULT_OUTPUT_FILE, help='Caminho para o arquivo de saída da matriz TF-IDF.')
    parser.add_argument('--column_text', type=str, default=DEFAULT_COLUMN_TEXT, help='Nome da coluna que contém o texto a ser vetorizado.')
    args = parser.parse_args()

    # Leitura do arquivo CSV
    df = pd.read_csv(args.csv_file)
    print('DataFrame obtido do CSV')

    # Vetorização TF-IDF
    vocabulary, tfidf_matrix = vectorize_tfidf(df, args.column_text)
    print("Vetorização TF-IDF concluída")

    # Salvando a matriz TF-IDF esparsa
    save_npz(args.output_file, tfidf_matrix)
    print("Matriz TF-IDF esparsa salva")

if __name__ == "__main__":
    main()
