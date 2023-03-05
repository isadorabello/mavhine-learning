from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# features -> caracteristicas
# pelo curto, perna curta, latido

porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cao1 = [0, 1, 1]
cao2 = [1, 0, 1]
cao3 = [1, 1, 1]

# x-> dados
# y-> classes
treino_x = [porco1, porco2, porco3, cao1, cao2, cao3]
treino_y = [1, 1, 1, 0, 0, 0] # labels/etiquetas


model = LinearSVC()
model.fit(treino_x,treino_y)

animal1 = [1, 1, 1]
animal2 = [1, 1, 0]
animal3 = [0, 1, 1]
testes_x = [animal1, animal2, animal3]
testes_y = [0, 1, 1]


previsoes = model.predict(testes_x)

acertos = (previsoes==testes_y).sum()
total = len(testes_x)

print("taxa de acertos: %.2f" % (acertos/total * 100))

print("acuraria: %.2f" % (accuracy_score(testes_y, previsoes) *100))


