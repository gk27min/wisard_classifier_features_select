import plotly.express as px
import pandas as pd

file_path = '/home/gssilva/datasets/atribuna-elias/aTribuna-Elias.csv'
df = pd.read_csv(file_path, delimiter=",")

img = px.histogram(df, x='LABEL')
img.write_image("/home/gssilva/datasets/atribuna-elias/full/results/imagens/distribution.png")

