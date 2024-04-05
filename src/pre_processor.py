import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from concurrent.futures import ProcessPoolExecutor

# Download de stopwords e rslp
nltk.download('stopwords')
nltk.download('rslp')
STOP_WORDS_PT = set(stopwords.words('portuguese'))

# Constantes
CSV_FILE = '/home/gssilva/datasets/atribuna-elias/aTribuna.csv'
OUTPUT_FILE = '/home/gssilva/datasets/atribuna-elias/preprocessed_aTribuna.csv'
COLUMN_TEXT = 'ABSTRACT'
N_PROCS = 4

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
    try:
        df = pd.read_csv(CSV_FILE, encoding="iso-8859-1")
    except pd.errors.ParserError:
        print(f"Error reading CSV file {CSV_FILE}. Check the file for inconsistencies.")
        return

    print("dataset was collected!")

    # Limpeza e stemming do texto
    df[COLUMN_TEXT] = clean_and_stem_text_parallel(df[COLUMN_TEXT], N_PROCS)

    # Salvando o DataFrame processado
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done preprocessing!")

if __name__ == "__main__":
    main()
