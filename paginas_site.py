import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"

dados = pd.read_csv(uri)

# print(dados)
# print(dados.head())

mapa = {
    "home":"inicio",
    "how_it_works":"como_funciona",
    "contact":"contato",
    "bought":"comprou"
}
dados = dados.rename(columns=mapa)
x = dados[["inicio", "como_funciona", "contato"]]
y = dados["comprou"]

# print(x.head())
# print(y.head())

treino_x = x[:75]
treino_y = y[:75]
teste_x = x[75:]
teste_y = y[75:]

print("Treino: %d e Teste: %d" % (len(treino_x), (len(teste_x))))

model = LinearSVC()
model.fit(treino_x,treino_y)

previsoes = model.predict(teste_x)
acertos = (previsoes==teste_y).sum()
total = len(teste_x)

print("acuraria: %.2f%%" % (accuracy_score(teste_y, previsoes) *100))
