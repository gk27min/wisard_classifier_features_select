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


# if __name__=="__main__":
 
#         #criação do parser para os argumentos em linha de comando
#     parser = argparse.ArgumentParser()

#     #adição dos argumentos ao parser
#     parser.add_argument("--csvIn", type=str, help="Spacy model pkg that will be used. i.e: 'en_core_web_lg'.")
#     parser.add_argument("--csvOut", type=str, required=False, help="Output File Name. 'Dataset' is the default name")
#     parser.add_argument("--nDocs",type=int, help="Number of documents from CSV file", )
#     parser.add_argument("--nProcs", type=int, help="Number of Processes",)
#     parser.set_defaults(debug=False)

#     #obtenção dos argumentos
#     args = parser.parse_args()

#modifica as configurações padrão considerando as configurações passadas pelo usuário
origin_csv_file = '/home/gssilva/datasets/atribuna-elias/aTribuna-Elias.csv'
output_file_name = '/home/gssilva/datasets/atribuna-elias/full/preprocessed_aTribuna-Elias.csv'
n_procs= 30
column_text = "ABSTRACT"

# with open(origin_csv_file, "r", encoding="iso-8859-1") as rdb:
#     data = [
#         [d.split(";")[0], d.split(";")[1], d.split(";")[2], ";".join(d.split(";")[3:])]
#         for d in rdb.read().split("<br><br>\n") if d != "" ]

# df = pd.DataFrame(data).rename({0:"index", 1: "class", 2: "filename", 3:"text"}, axis=1).set_index("index")

# proporc = n_texts_documents/len(df)
# df_reduced = df.groupby('coluna_estratificada').apply(lambda x: x.sample(frac=proporc)).reset_index(drop=True)

df = pd.read_csv(origin_csv_file, encoding="iso-8859-1")
print("dataset was colected!")

df[column_text] = clean_and_stem_text_parallel(df[column_text], n_procs)
df.to_csv(output_file_name,index=False)
print("Done preprocessing!")