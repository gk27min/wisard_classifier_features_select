PATH = 'home/lcad/gess/WISARD_CLASSIFIER/'
PATH_MOSTER = '/home/gssilva/codes/wisard_codes'
VENV_PATH = /usr/bin/python3

PREPROCESSOR_160:
	$(VENV_PATH) src/pre_processor.py \
	--csvIn '/home/gssilva/datasets/atribuna-site/aTribuna.csv' \
	--csvOut '/home/gssilva/datasets/atribuna-site/full/preprocessed_aTribuna_full.csv' \
	--nDocs 45000 \
	--nProcs 50

VECTORIZE_160:
	$(VENV_PATH) src/vectorize.py \
	--csvIn '/home/gssilva/datasets/atribuna-site/full/processed_Atribuna.csv' \
	--csvOut '/home/gssilva/datasets/atribuna-site/full/vectorized_aTribuna_full.csv'

SELECTION_160:
	$(VENV_PATH) src/select_features \
	--csvIn '/home/gssilva/datasets/atribuna-site/full/vectorized_aTribuna_full.csv' \
	--folderOut '/home/gssilva/datasets/atribuna-site/full/selections/' \
	--labels '/home/gssilva/datasets/atribuna-site/full/preprocessed_aTribuna_full.csv' \
	--trainFolder '/home/gssilva/datasets/atribuna-site/full/train_test/' \
	--column 'class'
	--minInClass 0.60 \
	--maxInClass 0.60 \
	--minOutClass 0.20 \
	--maxOutClass 0.20 \
	--nProcs 5

OPTIMIZE_160:
	$(VENV_PATH) optimize_model.py \
	--features '/home/gssilva/datasets/atribuna-site/full/selections/selected_features[0.6, 0.2].txt' \
	--csvOut '/home/gssilva/datasets/atribuna-site/full/otimizacao/full_wisard.csv'
	--folderTrain '/home/gssilva/datasets/atribuna-site/full/train_test/'
	--minInClass 0.60 \
	--maxOutClass 0.20 \
	--nTrials 50 \
	--nProcs 20 \