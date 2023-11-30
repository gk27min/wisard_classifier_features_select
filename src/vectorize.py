import pandas as pd
import json
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_tfidf(df):
    if 'text' in df.columns:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
        return tfidf_vectorizer.vocabulary_, tfidf_matrix
    else:
        raise ValueError("A coluna 'text' não está presente no DataFrame.")

# Localização dos arquivos
origin_csv_file = '/home/gssilva/datasets/atribuna-site/full/processed_Atribuna.csv'
output_file_name = '/home/gssilva/datasets/atribuna-site/full/vectorized_aTribuna_full.npz'
vocabulary_file = '/home/gssilva/datasets/atribuna-site/full/vocabulary.json'

# Leitura do arquivo CSV
df = pd.read_csv(origin_csv_file)
print('DataFrame obtido do CSV')

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


# #modifica as configurações padrão considerando as configurações passadas pelo usuário
# origin_csv_file = '/home/gssilva/datasets/atribuna-site/full/vectorized_aTribuna_full.pkl'
# output_folder = '/home/gssilva/datasets/atribuna-site/full/selections/'
# min_inclass = 0.6
# max_inclass = 0.6
# min_outclass = 0.2
# max_outclass = 0.2
# path_file_labels = '/home/gssilva/datasets/atribuna-site/full/preprocessed_aTribuna_full.csv'
# train_folder = '/home/gssilva/datasets/atribuna-site/full/train_test/'
# labels_column = 'class'
# n_procs=10

# # tfidf_df = pd.read_pickle(origin_csv_file)
# # df = pd.read_csv(path_file_labels)
# labels = df[labels_column]
# # vectorized = tfidf_df.to_numpy(dtype='float32')

# X_train, X_test, y_train, y_test = split_train_test(vectorized, labels)

# # np.savetxt(f'{train_folder}/X_train.csv', X_train, delimiter=',')
# # np.savetxt(f'{train_folder}/X_test.csv', X_test, delimiter=',')
# # np.savetxt(f'{train_folder}/y_train.csv', y_train, delimiter=',')
# # np.savetxt(f'{train_folder}/y_test.csv', y_test, delimiter=',')
# print("Train and Test split!")

# def float_range(start, stop, step):
#     while start < stop:
#         yield round(start, 1)  # Arredonda para evitar problemas de precisão
#         start += step

# for i in float_range(min_inclass, max_inclass + 1, 0.5):
#     for j in float_range(min_outclass, max_outclass + 1, 0.5):
#         _, selected_features = select_features_parallel(X_train, y_train, min_inclass, max_outclass)
#         np.savetxt(f"{output_folder}/selected_features[{min_inclass},{max_outclass}].txt", selected_features, delimiter=",")
