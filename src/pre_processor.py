import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import re
import string
from concurrent.futures import ProcessPoolExecutor
import argparse
nltk.download('stopwords')
stop_words_pt = set(stopwords.words('portuguese'))


def clean_and_stem_text(text):
    if not isinstance(text, str):
        return ""
    stemmer = RSLPStemmer()
    text = re.sub(r'<[^>]+>', '', text)

    text = re.sub(r'\d{2}/\d{2}/\d{4}', '', text)  # datas
    text = re.sub(r'\d+', '', text)  # nums
    text = re.sub(r'http\S+', '', text)  # links

    text = re.sub(r'Editoria:.*', '', text) #publications notes
    text = re.sub(r'Data da Publicação:.*', '', text)

    text = ''.join([char for char in text if char not in string.punctuation]) #punctuation

    text = text.lower() #lower

    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words if word not in stop_words_pt] #stopword and stemming

    return ' '.join(stemmed_words) if stemmed_words else text

def clean_and_stem_text_parallel(texts, num_workers=4):
    results = []
    for text in texts:
        results.append(clean_and_stem_text(text))
    return results

#modifica as configurações padrão considerando as configurações passadas pelo usuário
origin_csv_file = '/home/gssilva/datasets/atribuna-elias/aTribuna-Elias.csv'
output_file_name = '/home/gssilva/datasets/atribuna-elias/full/preprocessed_aTribuna-Elias.csv'
n_procs= 30
column_text = "ABSTRACT"

df = pd.read_csv(origin_csv_file, encoding="iso-8859-1")
print("dataset was colected!")

df[column_text] = clean_and_stem_text_parallel(df[column_text], n_procs)
df.to_csv(output_file_name,index=False)
print("Done preprocessing!")