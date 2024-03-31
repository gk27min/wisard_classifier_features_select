import argparse
import wisardpkg as wsd
import scipy.sparse as sp
import numpy as np
from scipy.sparse import load_npz, save_npz

def thermometer_binarize(thermometer, X):
    num_features = X.shape[1]
    thermometer_sizes = [thermometer] * num_features

    # Converter para formato denso se X for uma matriz esparsa
    if sp.issparse(X):
        X = X.toarray()

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    dtherm = wsd.DynamicThermometer(thermometer_sizes, mins, maxs)

    # Aplica a transformação em todo o conjunto de dados de uma vez
    binX = [dtherm.transform(X[i]) for i in range(len(X))]
    return binX

def binary_binarize(X):
    binarized_data = []
    for row in X:
        # Verifica se cada valor na linha é maior que zero e converte para int (0 ou 1)
        binarized_row = [int(value > 0) for value in row.data]
        binarized_data.append(binarized_row)
    return binarized_data

def main():
    # Argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Binarização de dados usando termômetros dinâmicos.')
    parser.add_argument('--therm', type=int, default=62, help='Tamanho do termômetro.')
    parser.add_argument('--data_file', type=str, default='/home/gssilva/datasets/atribuna-elias/full/vect_aTribuna.npz', help='Caminho para o arquivo de dados.')
    parser.add_argument('--bin_file', type=str, default='/home/gssilva/datasets/atribuna-elias/full/bin_aTribuna.npz', help='Caminho para o arquivo binarizado.')
    parser.add_argument('--method', type=str, choices=['thermometer', 'binary'], default='thermometer', help='Método de binarização a ser usado (thermometer ou binary).')
    args = parser.parse_args()

    # Carregar dados
    data = load_npz(args.data_file)

    # Binarizar dados
    if args.method == 'thermometer':
        binarized_data = thermometer_binarize(args.therm, data)
    else:
        binarized_data = binary_binarize(data)

    # Salvar dados binarizados
    save_npz(args.bin_file, binarized_data)

    print(f"Dados binarizados salvos em {args.bin_file}")

if __name__ == "__main__":
    main()
