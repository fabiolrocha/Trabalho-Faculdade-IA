# --- CARREGANDO BASE DE DADOS ---

# O parâmetro stringsAsFactors = TRUE já converte colunas de texto em categorias (Fatores),
# o que é obrigatório para os algoritmos de classificação no R.
dados <- read.csv(file.choose(), stringsAsFactors = TRUE)

# Remover a coluna customerID
# O ID do cliente é único e não tem valor preditivo. Se deixarmos, o modelo pode "decorar" IDs.
dados$customerID <- NULL

# Tratar valores ausentes (NAs)
# A coluna TotalCharges tem 11 valores vazios. Como a base tem mais de 7000 linhas, 
# a remoção dessas 11 linhas é a abordagem mais simples e tecnicamente justificável.
dados <- na.omit(dados)

# Divisão em Treino e Teste
# Precisamos do pacote 'caret' para fazer uma divisão balanceada.
# install.packages("caret")
library(caret)

# Fixar a semente aleatória.
set.seed(42)

# Criar o índice para divisão (80% treino, 20% teste)
# A vantagem do createDataPartition é que ele mantém a proporção de "Yes" e "No" do Churn nas duas bases.
indice_treino <- createDataPartition(dados$Churn, p = 0.8, list = FALSE)

treino <- dados[indice_treino, ]
teste <- dados[-indice_treino, ]

# Verificando as dimensões para confirmar se deu tudo certo
print("Tamanho da base de Treino (Linhas, Colunas):")
print(dim(treino))
print("Tamanho da base de Teste (Linhas, Colunas):")
print(dim(teste))

# --- MODELO 1: NAIVE BAYES ---

# install.packages("e1071")
library(e1071)

# Treinar o modelo de Naive Bayes
# O til (~) com o ponto (.) significa: "Preveja a coluna Churn usando TODAS as outras colunas da base de treino"
modelo_nb <- naiveBayes(Churn ~ ., data = treino)

# Fazer as previsões (teste)
# Aqui pedimos para o modelo tentar adivinhar quem vai cancelar, olhando apenas as características da base de teste
previsoes_nb <- predict(modelo_nb, teste)

# Chama o pacote caret de volta.
library(caret)

# Avaliar o modelo
resultado_nb <- confusionMatrix(previsoes_nb, teste$Churn, positive = "Yes")

# Mostrar todos os resultados na tela
print(resultado_nb)



# --- MODELO 2: ÁRVORE DE DECISÃO ---

# install.packages("rpart")
# install.packages("rpart.plot") 

library(rpart)
library(rpart.plot)

# Treinar o modelo de Árvore de Decisão
# O method="class" indica que queremos classificar (Yes/No) e não prever um número contínuo
modelo_arvore <- rpart(Churn ~ ., data = treino, method = "class")

# Fazer as previsões na base de teste
previsoes_arvore <- predict(modelo_arvore, teste, type = "class")

# Chama o pacote caret de volta.
library(caret)

# Avaliar o modelo
resultado_arvore <- confusionMatrix(previsoes_arvore, teste$Churn, positive = "Yes")
print("--- RESULTADOS DA ÁRVORE DE DECISÃO ---")
print(resultado_arvore)

# Visual de ARVORE para melhor entendimento.
rpart.plot(modelo_arvore, main="Árvore de Decisão para Previsão de Churn", extra=104, box.palette="RdBu")



# --- MODELO 3: RANDOM FOREST ---

#install.packages("randomForest")
library(randomForest)
library(caret)

# Treinar o modelo de Random Forest
# O ntree = 100 define que vamos plantar 100 árvores de decisão. 
# O importance = TRUE vai nos permitir ver quais colunas foram mais importantes para a decisão.
set.seed(42)
modelo_rf <- randomForest(Churn ~ ., data = treino, ntree = 100, importance = TRUE)

# Fazer as previsões na base de teste
previsoes_rf <- predict(modelo_rf, teste)

# Avaliar o modelo
resultado_rf <- confusionMatrix(previsoes_rf, teste$Churn, positive = "Yes")
print("--- RESULTADOS DO RANDOM FOREST ---")
print(resultado_rf)

# Visual de GRAFICO para melhor entendimento.
varImpPlot(modelo_rf, main="Variáveis Mais Importantes (Random Forest)")

# --- EXPORTAÇÃO PARA PYTHON ---

# Exportando as bases divididas para garantir 100% de igualdade no Python
write.csv(treino, "base_treino_churn.csv", row.names = FALSE)
write.csv(teste, "base_teste_churn.csv", row.names = FALSE)
