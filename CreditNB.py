# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 09:18:20 2024

# Projeto de Concessão de Crédito para Cartões de Crédito

@author: Luiz Paulo Rodrigues Almeida
"""

# %% Imports

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importar pré-processamento do sklearn
from sklearn import preprocessing

# Importar Split para Teste e Treino de modelo
from sklearn.model_selection import train_test_split

# Importar scaler
from sklearn.preprocessing import StandardScaler

# Importar métricas de acurácia
from sklearn.metrics import accuracy_score

# importar regressão logística
from sklearn.linear_model import LogisticRegression

# Importar GridSearch
from sklearn.model_selection import GridSearchCV

# Importar Regressão logística
from sklearn.neighbors import KNeighborsClassifier

# Importar Support Vector
from sklearn.svm import SVC

# Importar Arvorés de decisão
from sklearn.tree import DecisionTreeClassifier

# Importar Random Forest
from sklearn.ensemble import RandomForestClassifier

# Importar AdaBoost
from sklearn.ensemble import AdaBoostClassifier

# Importar XGBoost
from xgboost import XGBClassifier

# %% Ler arquivos .csv

# Arquivos .csv extraídos de: https://www.kaggle.com/datasets/rohitudageri/credit-card-details/data
# Tais arquivos estão classificados como domínio público

data = pd.read_csv("Credit_card.csv") # dados principais
label = pd.read_csv("Credit_card_label.csv") # se o crédito foi concedido ou não (1 rejeitado, 0 aprovado)

data = data.merge(label, on = "Ind_ID", how = "left")

data.head()

# %% Verificação de NAs e limpeza muito básica

data.info()
data.describe()
data.isna().sum()

# Na checagem tem uma coluna de nome "Type_Occupation" onde há muitos NAs
# então vamos remover essa coluna, nem toda renda vem de uma ocupação formal

data = data.drop(columns = ("Type_Occupation"))

# Checamos para ver se há muitos NAs ainda
data.isna().sum()

# Nesse caso há alguns poucos registros com NAs, então vamos remover
# Só esses registros para não influenciar o modelo
data = data.dropna()

# Vamos verificar se há dados duplicados
data.duplicated().sum()

# Não tem nenhum, mas se tivesse poderimos usar data.drop_duplicates()

# %% Adicionar algumas colunas calculadas
data["Age"] = data["Birthday_count"] / -365.25
data["Age"] = data["Age"].astype(int)

# %% Remover outliers da análise - Regra dos Desvios Padrão

# Calcular a média e o desvio padrão para os campos onde temos outliers
medAI = data["Annual_income"].mean()
desvpadAI = data["Annual_income"].std()

# Definir os limites inferior e superior para os outliers
li = medAI - 3 * desvpadAI
ls = medAI + 3 * desvpadAI

# Printar os limites inferiores e superiores
print("Limite inferior é ", li, " \n o limite superior é ", ls)

# Filtrar os outliers
data = data[(data["Annual_income"] >= li) & (data["Annual_income"] <= ls)]

data.head()

# %% Fazer algumas análises mais simples

# Média dos salários | 190775
avg_income = data["Annual_income"].mean()
avg_income

# Média dos salários por gênero | M = 222485 | F = 172391
gender_income = data.groupby("GENDER")["Annual_income"].mean()
gender_income 

sns.barplot(x= gender_income.index, y = gender_income.values)
plt.title("Renda anual por Gênero")
plt.show()

# Aprovação ou não aprovação de crédito
label_counts = data["label"].value_counts()
label_counts

# Plotando em gráfico de pizza
plt.pie(label_counts, labels =["Crédito Aprovado", "Crédito Não Aprovado"], autopct = "%1.1f%%")
plt.title("Status de aprovação de crédito")
plt.show()

data.groupby("Type_Income")["Annual_income"].mean()

data.groupby("Marital_status")["Employed_days"].mean()

data.groupby("Housing_type")["Family_Members"].sum()

round(data.groupby("GENDER")["Birthday_count"].mean() / 365.25)

data.groupby("Propert_Owner")["Annual_income"].mean()

data.groupby("EDUCATION")["Propert_Owner"]

data.groupby("CHILDREN")["Annual_income"].sum()

# Gráfico Estado Civil versus Renda Anual
sns.boxplot(x = "Marital_status", y = "Annual_income", data= data)
plt.title("Variação anual de renda por Estado Civil")
plt.xlabel("Estado civil")
plt.ylabel("Renda")
plt.show()

# Gráfico Violino Idade por Gênero
sns.violinplot(x = "GENDER", y = "Age", data = data)
plt.title("Distribuição idade por gênero")
plt.xlabel("Gênero")
plt.ylabel("idade")
plt.show()

# Gráfico Violino Renda por Idade
sns.violinplot(x = "Age", y = "Annual_income", data = data)
plt.title("Distribuição renda por idade")
plt.xlabel("Idade")
plt.ylabel("Renda")
plt.show()

# Gráfico Pizza Membros da Família por Estado Civil
data.groupby("Marital_status")["Family_Members"].sum().plot(kind = "pie", autopct = "%1.1f%%")
plt.title("Membros da família por Estado Civil")
plt.ylabel("")
plt.show()

#Gráfico Boxplot Renda por nível Educacional
data.boxplot(column = "Annual_income", by = "EDUCATION")
plt.title("Renda Anual por nível Educacional")
plt.show()

# %% Dados relevantes para o modelo

datam = data[["Car_Owner", "Propert_Owner", "Annual_income", "EDUCATION", "label"]]
datam.head()

# %% Uniformizar os dados
labelencoder = preprocessing.LabelEncoder()

datam["Car_Owner"] = labelencoder.fit_transform(datam["Car_Owner"])
datam["Propert_Owner"] = labelencoder.fit_transform(datam["Propert_Owner"])
datam["EDUCATION"] = labelencoder.fit_transform(datam["EDUCATION"])

X = datam.iloc[:, : -1]
y = datam.iloc[:, -1]
   
# %% Quebrar em modelo de teste e de treino e normalizar as valores

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)

# %% Testar modelo de regressão logistica

# Função para informar a performace do modelo
def modelperformance(predictions):
    print("Pontuação de acurácia do modelo é {}".format(accuracy_score(y_test, predictions)))

# Estabelecer modelo de regressão logistica
log_model = LogisticRegression()

# Treinar o modelo
log_model.fit(scaled_X_train, y_train)

# Com os dados de treino, comparar com os dados de teste
log_predictions = log_model.predict(scaled_X_test)

# Performance do modelo por reglog quanto mais perto de 1, melhor
modelperformance(log_predictions)
mreglog = accuracy_score(y_test, log_predictions)

# %% Testar modelo por Clusterização

# Parametros para Classificar por Kneighbors
param_grid = {"n_neighbors" : [3,5,7,9,11],
              "weights" : ["uniform", "distance"],
              "algorithm" : ["auto", "kd_tree"]}

# Usar o GridSearch para encontrar o melhor parametro entre os informados
gridkn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=2)

# Treinar o modelo
gridkn.fit(scaled_X_train, y_train)

# Verificar os melhores parametros
gridkn.best_params_

# Submeter o modelo ao teste
knpreds = gridkn.predict(scaled_X_test)

# Performance do modelo por Kneighbors
modelperformance(knpreds)
mknpreds = accuracy_score(y_test, knpreds)

# %% Testar Modelo com Vetor de classificação de suporte

# Atribuir o moddelo
svm = SVC()

# Parametros para Classificar por Support Vectors
param_grid_svr = {"C" : [0.01, 0.1, 0.5, 1],
                  "kernel" : ["linear", "rbf", "poly"]}

# Usar o GridSearch para encontrar o melhor parametro entre os informados
gridsvr = GridSearchCV(svm, param_grid_svr)

# Treinar o modelo
gridsvr.fit(scaled_X_train, y_train)

# Verificar os melhores parâmetros
gridsvr.best_params_

# Submeter o modelo ao teste
pred_svc = gridsvr.predict(scaled_X_test)

# Performance do modelo por SVC
modelperformance(pred_svc)
msvc = accuracy_score(y_test, pred_svc)

# %% Testar modelo de Arvorés de Decisão

# Parametros para Classificar por Arvorés de Decisão
param_grid = {
    "criterion"          : ["gini", "entropy"],
    "splitter"           : ["best", "random"],
    "max_depth"          : [None, 10,20, 30, 40, 50],
    "min_samples_split"  : [2,5,10],
    "min_samples_leaf"   : [1,2,4]
    }

# Usar o GridSearch para encontrar o melhor parametro entre os informados
grid_search_tree = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = param_grid)

# Treinar o modelo
grid_search_tree.fit(scaled_X_train, y_train)

# Verificar os melhores parâmetros
grid_search_tree.best_params_

# Submeter o modelo ao teste
treepred = grid_search_tree.predict(scaled_X_test)

# Performance do modelo por Arvores de Decisão
modelperformance(treepred)
mtree = accuracy_score(y_test, treepred)

# %% Testar modelo Random Forest

# Criar o modelo
rfc_model = RandomForestClassifier()

# Parametros para Classificar por Random Forest
n_estimators = [32, 64, 128, 256]
max_features = [2,3,4]
bootstrap = [True, False]
oob_score = [True, False]

param_grid_rfc = {
    "n_estimators" : n_estimators,
    "max_features" : max_features,
    "bootstrap" : bootstrap,
    "oob_score": oob_score
    }

# Usar o GridSearch para encontrar o melhor parametro entre os informados
grid_rfc = GridSearchCV(rfc_model, param_grid_rfc)

# Treinar o modelo
grid_rfc.fit(scaled_X_train, y_train)

# Verificar os melhores parâmetros
grid_rfc.best_params_

# Submeter o modelo ao teste
predsrfc = grid_rfc.predict(scaled_X_test)

# Performance do modelo por Arvores de Decisão
modelperformance(predsrfc)
mrfc = accuracy_score(y_test, predsrfc)

# %% Testar Ada Boost

# Criar o modelo
ada_classifier = AdaBoostClassifier()

# Parametros para Classificar por Ada Boost
ada_param_grid = {
    "n_estimators" : [50, 100, 200],
    "learning_rate" : [0.01, 0.1, 1, 10]
    }

# Usar o GridSearch para encontrar o melhor parametro entre os informados
ada_grid_search = GridSearchCV(ada_classifier, ada_param_grid)

# Treinar o modelo
ada_grid_search.fit(scaled_X_train, y_train)

# Verificar os melhores parâmetros
ada_grid_search.best_params_

# Submeter o modelo ao teste
adapred = ada_grid_search.predict(scaled_X_test)

# Performance do modelo por Ada Boost
modelperformance(adapred)
mada = accuracy_score(y_test, adapred)

# %% Testar model de Descida do Gradiente

# Criar o modelo
xgb_classifier = XGBClassifier()

# Parametros para Classificar por XGBoost
xgb_param_grid = {
    "n_estimators" : [60, 80, 100, 500, 1000],
    "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5],
    "max_depth" : [4,5,7]
    }

# Usar o GridSearch para encontrar o melhor parametro entre os informados
xgb_grid_search = GridSearchCV(xgb_classifier, xgb_param_grid, cv = 3)

# Treinar o modelo
xgb_grid_search.fit(scaled_X_train, y_train)

# Verificar os melhores parâmetros
xgb_grid_search.best_params_

# Submeter o modelo ao teste
xgbpred = xgb_grid_search.predict(scaled_X_test)

# Performance do modelo por XGBoost
modelperformance(xgbpred)
mxgb = accuracy_score(y_test, xgbpred)

# %% Comparação dos resultados dos modelos

comparativo = {
    'Metodo': ['Reg Logística', 'KNeighbors', 'SVC', 'Árvore', 'Random Forest', 'Ada Boost', 'XG Boost'],
    'Perfomance': [mreglog, mknpreds, msvc, mtree, mrfc, mada, mxgb]
}

comp_df = pd.DataFrame(comparativo)
comp_df
