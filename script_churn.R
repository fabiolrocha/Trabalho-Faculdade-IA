# --- CARREGANDO BASE DE DADOS ---

dados <- read.csv(file.choose(), stringsAsFactors = TRUE)

dados$customerID <- NULL

dados <- na.omit(dados)

# install.packages("caret")
library(caret)

set.seed(42)

indice_treino <- createDataPartition(dados$Churn, p = 0.8, list = FALSE)

treino <- dados[indice_treino, ]
teste <- dados[-indice_treino, ]

print("Tamanho da base de Treino (Linhas, Colunas):")
print(dim(treino))
print("Tamanho da base de Teste (Linhas, Colunas):")
print(dim(teste))

# --- MODELO 1: NAIVE BAYES ---

# install.packages("e1071")
library(e1071)

# Treinar o modelo de Naive Bayes
modelo_nb <- naiveBayes(Churn ~ ., data = treino)

previsoes_nb <- predict(modelo_nb, teste)

library(caret)

resultado_nb <- confusionMatrix(previsoes_nb, teste$Churn, positive = "Yes")

print(resultado_nb)


# --- MODELO 2: ÁRVORE DE DECISÃO ---

# install.packages("rpart")
# install.packages("rpart.plot") 

library(rpart)
library(rpart.plot)

# Treinar o modelo de Árvore de Decisão
modelo_arvore <- rpart(Churn ~ ., data = treino, method = "class")

previsoes_arvore <- predict(modelo_arvore, teste, type = "class")

library(caret)

resultado_arvore <- confusionMatrix(previsoes_arvore, teste$Churn, positive = "Yes")
print("--- RESULTADOS DA ÁRVORE DE DECISÃO ---")
print(resultado_arvore)

rpart.plot(modelo_arvore, main="Árvore de Decisão para Previsão de Churn", extra=104, box.palette="RdBu")


# --- MODELO 3: RANDOM FOREST ---

#install.packages("randomForest")
library(randomForest)
library(caret)

# Treinar o modelo de Random Fores
set.seed(42)
modelo_rf <- randomForest(Churn ~ ., data = treino, ntree = 100, importance = TRUE)

previsoes_rf <- predict(modelo_rf, teste)

resultado_rf <- confusionMatrix(previsoes_rf, teste$Churn, positive = "Yes")
print("--- RESULTADOS DO RANDOM FOREST ---")
print(resultado_rf)

varImpPlot(modelo_rf, main="Variáveis Mais Importantes (Random Forest)")

# --- EXPORTAÇÃO PARA PYTHON ---

write.csv(treino, "base_treino_churn.csv", row.names = FALSE)
write.csv(teste, "base_teste_churn.csv", row.names = FALSE)
