import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

SEED = 5
train_x, test_x, train_y, test_y = train_test_split(x, y,  random_state = SEED, test_size = 0.25,stratify = y)

print(len(train_x), len(test_x))

modelo = LinearSVC()
modelo.fit(train_x,train_y)
previsoes = modelo.predict(test_x)

print("acuraria: %.2f%%" % (accuracy_score(test_y, previsoes) *100))

previsoes_base = np.ones(540)
print("acuraria baseline: %.2f%%" % (accuracy_score(test_y, previsoes_base) *100))

# PLOTAGEM
# separando os dados em uma gráfico
# sns.scatterplot(x='hrs_estimadas', y='preco', hue=test_y, data = test_x)
# sns.relplot(x='hrs_estimadas', y='preco', hue="finalizado",col="finalizado", data = dados)
# plotando
# plt.show()

x_min = test_x.hrs_estimadas.min()
x_max = test_x.hrs_estimadas.max()
y_min = test_x.preco.min()
y_max = test_x.preco.max()
print(x_min, x_max,y_min,y_max)

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max-x_min) / pixels)
print(eixo_x)

eixo_y = np.arange(y_min, y_max, (y_max-y_min) / pixels)
print(eixo_y)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]
print(pontos)

Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)
print(Z)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(test_x.hrs_estimadas, test_x.preco, c=test_y, s=1)

# plt.show()