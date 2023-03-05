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

dados = [porco1, porco2, porco3, cao1, cao2, cao3]
classes = [1, 1, 1, 0, 0, 0]


model = LinearSVC()
model.fit(dados,classes)

animal = [1, 1, 1]
animal2 = [1, 1, 0]
animal3 = [0, 1, 1]
testes = [animal, animal2, animal3]

previsoes = model.predict(testes)
testes_classes = [0, 1, 1]

acertos = (previsoes==testes_classes).sum()
total = len(testes)

print("taxa de acertos: ", acertos/total * 100)

print("acuraria: ", accuracy_score(testes_classes, previsoes) *100)


