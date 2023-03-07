import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

print("acuraria 1: %.2f%%" % (accuracy_score(teste_y, previsoes) *100))

SEED = 20 # NUMERO INICIAL PARA A GERAÇÃO DE ELEMENTOS ALEATÓRIOS

#                                  RANDOMICO
train_x, test_x, train_y, test_y = train_test_split(x, y, 
                                                    random_state = SEED,  #faz com que não seja "tão" aleatório assim
                                                    test_size = 0.25,
                                                    stratify = y # manter os valores de y igualmente proporcionais nas porçoes de treino e teste
                                                    )

model = LinearSVC()
model.fit(train_x,train_y)

previsoes = model.predict(test_x)
acertos = (previsoes==test_y).sum()

print("acuraria 2: %.2f%%" % (accuracy_score(test_y, previsoes) *100))
