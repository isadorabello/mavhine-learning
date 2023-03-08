import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)

# print(dados.head())

nome_pt = {
    "unfinished": "nao_finalizado",
    "expected_hours": "hrs_estimadas",
    "price": "preco"
}

dados = dados.rename(columns=nome_pt)
# print(dados.head())

# trocando os valores no CSV para facilitar a leitura: não finalizado -> finalizado, 0 -> 1, 1 -> 0

troca = {
    0:1,
    1:0
}

dados['finalizado'] = dados.nao_finalizado.map(troca)
# print(dados.head())
# print(dados.tail())

# PLOTAGEM
# separando os dados em uma gráfico
# sns.scatterplot(x='hrs_estimadas', y='preco', hue="finalizado", data = dados)
# sns.relplot(x='hrs_estimadas', y='preco', hue="finalizado",col="finalizado", data = dados)
# plotando
# plt.show()

x = dados[['hrs_estimadas', 'preco']]
y=dados['finalizado']

SEED = 20
train_x, test_x, train_y, test_y = train_test_split(x, y,  random_state = SEED, test_size = 0.25,stratify = y)

print(len(train_x), len(test_x))

modelo = LinearSVC()
modelo.fit(train_x,train_y)
previsoes = modelo.predict(test_x)

print("acuraria: %.2f%%" % (accuracy_score(test_y, previsoes) *100))

previsoes_base = np.ones(540)
print("acuraria baseline: %.2f%%" % (accuracy_score(test_y, previsoes_base) *100))

