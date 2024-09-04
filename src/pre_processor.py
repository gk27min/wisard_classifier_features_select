import pandas as pd
import nltk
import re
import spacy
import string
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from concurrent.futures import ProcessPoolExecutor

# Download de stopwords e rslp
nltk.download('stopwords')
nltk.download('rslp')
STOP_WORDS_PT = set(stopwords.words('portuguese'))
nlp = spacy.load("pt_core_news_sm")

# Constantes
CSV_FILE = '/home/gssilva/datasets/atribuna-elias/aTribuna.csv'
OUTPUT_FILE = '/home/gssilva/datasets/atribuna-elias/preprocessed_aTribuna.csv'
COLUMN_TEXT = 'ABSTRACT'
N_PROCS = 4

# Compilar expressões regulares fora da função
pattern_tags = re.compile(r'<[^>]+>')
pattern_dates = re.compile(r'\d{2}/\d{2}/\d{4}')
pattern_numbers = re.compile(r'\d+')
pattern_links = re.compile(r'http\S+')
pattern_editoria = re.compile(r'Editoria:.*')
pattern_data_publicacao = re.compile(r'Data da Publicação:.*')

def clean_and_lemmatize_text_TJSP(text):
    if not isinstance(text, str):
        return ""
    
    # Manter apenas o texto a partir de "Vistos"
    text = text.split('Vistos.', 1)[-1]
    
    # Remoção de padrões específicos
    text = pattern_tags.sub('', text)
    text = pattern_dates.sub('', text)  # datas
    text = pattern_numbers.sub('', text)  # números
    text = pattern_links.sub('', text)  # links
    text = pattern_editoria.sub('', text)  # publicações notas
    text = pattern_data_publicacao.sub('', text)
    
    # Remoção de pontuação
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Processar o texto com spaCy
    doc = nlp(text.lower())
    
    # Lematização, remoção de nomes próprios e stop words
    lemmatized_text = ' '.join([token.lemma_ for token in doc if token.ent_type_ != 'PER' and not token.is_stop])
    
    return lemmatized_text
    
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
