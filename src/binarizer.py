import wisardpkg as wsd
import scipy.sparse as sp
import numpy as np
from scipy.sparse import lil_matrix, save_npz, load_npz

DEFAULT_THERM = 62
DEFAULT_DATA_FILE = '/home/gssilva/datasets/atribuna-elias/vectorized_aTribuna.npz'
DEFAULT_BIN_FILE = '/home/gssilva/datasets/atribuna-elias/bin/bin_62.npz'
DEFAULT_METHOD = 'thermometer'

def thermometer_binarize(thermometer, X):
    print("Binarizando os dados usando termômetros dinâmicos...")
    num_features = X.shape[1]
    thermometer_sizes = [thermometer] * num_features

    mins = np.min(X, axis=0).A.flatten()
    maxs = np.max(X, axis=0).A.flatten()

    bin_values = sp.lil_matrix(X.shape, dtype=np.int8)

    for i, row in enumerate(X):
        values = ((row.data - mins[row.indices]) / (maxs[row.indices] - mins[row.indices]) * thermometer).astype(np.int8)
        bin_values[i, row.indices] = values

    return bin_values


def binary_binarize(X):
    print("Binarizando os dados usando codificação binária...")
    binarized_data = []
    for row in X:
        binarized_row = [int(value > 0) for value in row.data]
        binarized_data.append(binarized_row)
    return binarized_data

def main(therm=DEFAULT_THERM, data_file=DEFAULT_DATA_FILE, bin_file=DEFAULT_BIN_FILE, method=DEFAULT_METHOD):
    print(f"Carregando dados de {data_file}...")
    # Carregar dados
    data = load_npz(data_file)

    # Binarizar dados
    if method == 'thermometer':
        binarized_data = thermometer_binarize(therm, data)
    else:
        binarized_data = binary_binarize(data)

    print(f"Salvando dados binarizados em {bin_file}...")

    binarized_data_csr = binarized_data.tocsr()
    save_npz(bin_file, binarized_data_csr)

    print(f"Dados binarizados salvos em {bin_file}")

if __name__ == "__main__":
    main()
