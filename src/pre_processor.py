import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from concurrent.futures import ProcessPoolExecutor
import argparse

# Download de stopwords
nltk.download('stopwords')
STOP_WORDS_PT = set(stopwords.words('portuguese'))

def clean_and_stem_text(text):
    if not isinstance(text, str):
        return ""
    
    stemmer = RSLPStemmer()
    
    # Remoção de padrões específicos
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\d{2}/\d{2}/\d{4}', '', text)  # datas
    text = re.sub(r'\d+', '', text)  # nums
    text = re.sub(r'http\S+', '', text)  # links
    text = re.sub(r'Editoria:.*', '', text)  # publicações notas
    text = re.sub(r'Data da Publicação:.*', '', text)
    
    # Remoção de pontuação
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Conversão para minúsculas e stemming
    text = ' '.join([stemmer.stem(word) for word in text.lower().split() if word not in STOP_WORDS_PT])
    
    return text

def clean_and_stem_text_parallel(texts, num_workers=4):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(clean_and_stem_text, texts))
    return results

def main():
    # Argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Preprocessamento de texto.')
    parser.add_argument('--csv_file', type=str, default='/home/gssilva/datasets/atribuna-elias/aTribuna-Elias.csv', help='Caminho para o arquivo CSV de entrada.')
    parser.add_argument('--output_file', type=str, default='/home/gssilva/datasets/atribuna-elias/full/preprocessed_aTribuna-Elias.csv', help='Caminho para o arquivo CSV de saída.')
    parser.add_argument('--n_procs', type=int, default=30, help='Número de processos para processamento paralelo.')
    parser.add_argument('--column_text', type=str, default='ABSTRACT', help='Nome da coluna que contém o texto a ser processado.')
    args = parser.parse_args()

    # Leitura do arquivo CSV
    df = pd.read_csv(args.csv_file, encoding="iso-8859-1")
    print("dataset was collected!")

    # Limpeza e stemming do texto
    df[args.column_text] = clean_and_stem_text_parallel(df[args.column_text], args.n_procs)

    # Salvando o DataFrame processado
    df.to_csv(args.output_file, index=False)
    print("Done preprocessing!")

if __name__ == "__main__":
    main()
