import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

print("--- INICIANDO PROCESSAMENTO EM PYTHON ---")

# Carregar exatamente as mesmas bases exportadas pelo R
treino = pd.read_csv('base_treino_churn.csv')
teste = pd.read_csv('base_teste_churn.csv')

# Separar as variáveis explicativas (X) da variável alvo (y)
X_treino = treino.drop('Churn', axis=1)
y_treino = treino['Churn']

X_teste = teste.drop('Churn', axis=1)
y_teste = teste['Churn']

# Transformar texto em números (Obrigatório no Python/Scikit-Learn)
X_treino_num = pd.get_dummies(X_treino, drop_first=True)
X_teste_num = pd.get_dummies(X_teste, drop_first=True)

# Garante que as colunas do teste fiquem idênticas às do treino
X_teste_num = X_teste_num.reindex(columns=X_treino_num.columns, fill_value=0)

# --- MODELO 1: NAIVE BAYES ---

# Treinar o modelo Naive Bayes
modelo_nb = GaussianNB()
modelo_nb.fit(X_treino_num, y_treino)

# Fazer as previsões
previsoes_nb = modelo_nb.predict(X_teste_num)

# Exibir as Métricas
print("\n--- RESULTADOS DO NAIVE BAYES ---")
print(f"Acurácia: {accuracy_score(y_teste, previsoes_nb):.4f}")
print(f"Revocação (Sensitivity): {recall_score(y_teste, previsoes_nb, pos_label='Yes'):.4f}")
print(f"Precisão: {precision_score(y_teste, previsoes_nb, pos_label='Yes'):.4f}")

print("\nMatriz de Confusão:")
# O labels garante a ordem correta para bater com o formato do R
print(confusion_matrix(y_teste, previsoes_nb, labels=['No', 'Yes']))

from sklearn.tree import DecisionTreeClassifier

# --- MODELO 2: ÁRVORE DE DECISÃO ---

# Treinar o modelo de Árvore de Decisão 
modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(X_treino_num, y_treino)

# Fazer previsões
previsoes_arvore = modelo_arvore.predict(X_teste_num)

# Exibir as Métricas
print("\n--- RESULTADOS DA ÁRVORE DE DECISÃO ---")
print(f"Acurácia: {accuracy_score(y_teste, previsoes_arvore):.4f}")
print(f"Revocação (Sensitivity): {recall_score(y_teste, previsoes_arvore, pos_label='Yes'):.4f}")
print(f"Precisão: {precision_score(y_teste, previsoes_arvore, pos_label='Yes'):.4f}")

print("\nMatriz de Confusão:")
print(confusion_matrix(y_teste, previsoes_arvore, labels=['No', 'Yes']))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# --- MODELO 3: RANDOM FOREST ---

# Treinar o modelo
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_treino_num, y_treino)

# Fazer previsões
previsoes_rf = modelo_rf.predict(X_teste_num)

# Exibir as Métricas
print("\n--- RESULTADOS DO RANDOM FOREST ---")
print(f"Acurácia: {accuracy_score(y_teste, previsoes_rf):.4f}")
print(f"Precisão: {precision_score(y_teste, previsoes_rf, pos_label='Yes'):.4f}")
print(f"Revocação (Sensitivity): {recall_score(y_teste, previsoes_rf, pos_label='Yes'):.4f}")
print(f"F1-score: {f1_score(y_teste, previsoes_rf, pos_label='Yes'):.4f}")

print("\nMatriz de Confusão:")
print(confusion_matrix(y_teste, previsoes_rf, labels=['No', 'Yes']))

from sklearn.model_selection import GridSearchCV

print("\n--- AJUSTE DE HIPERPARÂMETROS (ÁRVORE DE DECISÃO) ---")

# Definir os "configs" que o Python vai testar
# Vamos mandar ele testar árvores com profundidade máxima de 3, 5, 7 e 10 níveis.
parametros_para_testar = {
    'max_depth': [3, 5, 7, 10],
    'criterion': ['gini', 'entropy']
}

# Configurar o "Robô" que vai testar todas as combinações (GridSearchCV)
busca_parametros = GridSearchCV(DecisionTreeClassifier(random_state=42), parametros_para_testar, cv=5, scoring='accuracy')

# Rodar os testes na base de treino
busca_parametros.fit(X_treino_num, y_treino)

# Ver qual foi a configuração vencedora
print(f"Melhor configuração encontrada: {busca_parametros.best_params_}")

# Fazer previsões usando essa melhor árvore
arvore_tunada = busca_parametros.best_estimator_
previsoes_arvore_tunada = arvore_tunada.predict(X_teste_num)

# Exibir as novas métricas
print(f"Nova Acurácia (Tunada): {accuracy_score(y_teste, previsoes_arvore_tunada):.4f}")
print(f"Nova Precisão: {precision_score(y_teste, previsoes_arvore_tunada, pos_label='Yes'):.4f}")
print(f"Nova Revocação: {recall_score(y_teste, previsoes_arvore_tunada, pos_label='Yes'):.4f}")