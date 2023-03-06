import pandas as pd

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