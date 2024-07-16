# credit-approval
Análise de Crédito para Cartões de Crédito

Esse projeto tem como finalidade analisar dois pequenos datesets em formato .csv onde é possível extrair informações de clientes de uma compania de cartões de crédito, tais como Renda Anual, Formação Acadêmica, Estado Civil, Idade, se teve Credito Aprovado ou Não, entre outras informações.

A análise desses dois datasets podem permitir a criação de modelos de aprendizado de máquina para aprovar ou não a liberação de crédito para potenciais clientes.  

## Os datasets

Ambos os datasets, o "Credit_card.csv" onde contém as informações dos clientes e o "Credit_card_label.csv" que diz se foi aprovado ou não, foram baixados dê: https://www.kaggle.com/datasets/rohitudageri/credit-card-details/data sendo classificados como de domínio público.

* Uma observação importante quanto à coluna Birthday_count é que os valores estão negativos porque a contagem foi retroativa à data que o dataset foi gerado, por exemplo, hoje é 0, ontem foi -1 e assim por diante.
  
* A coluna Employed_Days segue a mesma lógica, mas no caso de valores positivos significa que são os dias que a pessoa está desempregada
  
*  Já a coluna label indica se aquele cliente teve crédito aprovado ou não, sendo 0 para Aprovado e 1 para Não-Aprovado

## Importação, análises iniciais e limpeza

Logo de início após importar os datasets, uni ambos em um dataframe pela coluna 'Ind_Id' que é comum a ambos, através de um left join.

Uma pequena análise mostrou que a coluna 'Type_Occupation' contava com 488 NAs, quase um terço das observações. 
Como o tipo de ocupação não é determinante para a análise, essa coluna foi excluída por completo.

Feito isso alguns poucos registros que também tinham NAs foram excluídos, nenhum dado duplicado foi encontrado.

Sem os NAs optei por remover os Outliers da coluna 'Annual_income' através da Regra dos Desvios Padrão, já que haviam valores alguns poucos valores muito discrepantes que poderiam atrapalhar o modelo.

## Gráficos para ajudar nos insights

Decidi plotar alguns gráficos que permitiram perceber algumas informações importantes como uma clara diferença de renda por gênero biológico, o percentual de crédito aprovado x não-aprovado e que há uma média salarial maior entre aqueles que possuem uma graduação.

Seguem alguns desses gráficos abaixo:

![Figure 2024-07-15 225738](https://github.com/user-attachments/assets/cac37889-f473-4850-8fe3-0453601755cb)

![Figure 2024-07-15 225811](https://github.com/user-attachments/assets/8a3070dc-4995-4bc9-8285-addda13bb7b1)

![Figure 2024-07-15 225816](https://github.com/user-attachments/assets/96556c37-e2f0-4751-b303-6e60936dac68)

![Figure 2024-07-15 225821](https://github.com/user-attachments/assets/e7dd5368-84b0-4d36-b7a5-7bee58998860)

![Figure 2024-07-15 225830](https://github.com/user-attachments/assets/1e83ce49-7969-4e31-9085-0bac9a23da99)

![Figure 2024-07-15 225833](https://github.com/user-attachments/assets/8eb729be-c208-45a5-a526-638cd55d96b3)

![Figure 2024-07-15 225837](https://github.com/user-attachments/assets/5f722386-b80b-44df-b10f-508f9b2cf24c)

## Modelos 

Com a ajuda desses insights, criei um dataframe menor, contendo apenas campos que seriam relevantes para aprovação de crédito para o cliente, como se o cliente tem carro, casa, o salário anual e o grau de instrução.

Também fiz a codificação dos campos 'Car_Owner', 'Propert_Owner' e 'EDUCATION' com Label Encoding, já que alguns modelos de Machine Learning exigem que os dados estejam numéricos.

Separei esses dados entre base de teste e base de treino através de um train_test_split, sendo a base de teste equivalente a 20% da base original.

Em sequência realizei a normalização de todos esses dados através do Z-Score.

Com isso pude partir para os modelos em si, testei 6 modelos:

### Regressão logistica

```py
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
```
### KNeighbors

```py 
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
```

### Support Vector

```py
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
```

### Árvores de Decisão

```py 
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
```

### Random Forest

```py
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
```

### Ada Boost

```py 
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
```

### XGBoost

```py
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
```

## Comparativo entre os modelos

Boa parte dos modelos teve exatamente a mesma performance, como pode ser analisado na tabela abaixo:
![image](https://github.com/user-attachments/assets/bfd2c992-cc1a-4d58-80e7-e5804eb7d0fc)

Com base nesses resultados métodos que consomem menos poder computacional tem preferência como a Regressão Logística, KNeighbors ou SVC.
Os resultados foram ligeramente piores (apesar de ainda serem bons) utilizando Árvore de Decisão e Random Forest... na análise eu rodei na base normalizada apesar de não ser necessário para esses métodos.
Ada Boost e XGBoost obtiveram o mesmo resultado, mas o processo desses métodos consome muito mais poder computacional.

*Note que se você for roda o código exatamente como está aqui, pode haver diferença nos valores, já que não usei uma semente específica para que o resultado seja sempre o mesmo.
