# Projeto 1
# Definição do problema de negócio:
# Detecção de Fraudes no Tráfego de Cliques em Propagandas de Aplicações Mobile
# O Objetivo é prever se o clique é fraudulento ou não

library(lubridate) # Trabalhar com Datas
library(tidyverse) # Manipular os dados
library(data.table) # Carregar os dados
library(caret) # ConfusionMatrix (análise da acurácia do modelo)
library(ROSE) # Balanciamento dos dados
library(caTools) # Dividir os dados (Treino e teste)
library(randomForest) # Modelo de machine learning
library(rpart) # Modelo de machine learning

# Carregando o dataset
df = as.data.frame(fread('train_sample.csv'))

# Informações básicas do dataset
View(df) # Visualizando o conjunto de dados
str(df) # Informações dos dados
summary(df)# Informações dos dados
head(df) # Visualizando as primeiras linhas dos dados

# Nossa variavél preditora é 'is_attibuted', vamos checar a quantidade de fraudes que temos.
prop.table(table(as.numeric(df$is_attributed))) # 99,87% dos clqiues sao fraudulentos

# Com o gráfico podemos confirmar isso, o número 0 significa cliques fradulentos
# Número 1 não aparece no grafico, pois é muito pequeno
ggplot(df)+
  geom_bar(aes(is_attributed)) 


# Limpeza e transformação
df$ip =NULL #Removendo o IP
df$attributed_time = NULL #Removendo a hora que foi feitoa a compra
df$click_time = NULL #Removendo a hora que foi feito o clique

# Transformando a variavel preditora em fator, para o modelo de machine learning ter uma performance melhor
df$is_attributed = as.factor(df$is_attributed) 

# Checando a distribuição dos dados nas variaveis independentes
boxplot(df$app) # outliers
boxplot(x= df$device) # outliers
boxplot(x= df$os) # outliers
boxplot(df$channel)

# Removendo os outliers das 3 variaveis: os, device, app
df = df%>%
  filter(os <= 30, device <= 2, app <= 30 )


# Transformando device em fator
df$device = as.factor(df$device)

# Visualizando e transformando a coluna channel dividida em 20 grupos.
hist(table(cut(x = df$channel, breaks = 20)))
df$channel = cut(df$channel, 20)

# Transformando nossa variavel preditora em fator
df$is_attributed = as.factor(df$is_attributed)


# Explorando os dados
boxplot(df$app) 
boxplot(x= df$device) 
boxplot(x= df$os) 
boxplot(df$channel)

# Nesse gráfico vemos que a maioria dos channels se encontra entre 102 e 299
ggplot(data=df)+
  geom_bar(aes(channel))

# Frequencia de fraudes por app
ggplot(df)+
  geom_freqpoly(aes(x = app, color = is_attributed))

# Aqui temos um insight, que mostra qual combinação das variaveis app, os, device e channels 
# que tiveram mais fraudes
risco_fraude = df %>%
  filter(is_attributed == 0)%>%
  group_by(app,os,device, channel)%>%
  summarise(c =  sum(as.numeric(is_attributed)))%>%
  arrange(desc(c))
View(risco_fraude)

# Qual app teve mais fraudes
df%>%
  filter(is_attributed == 0)%>%
  group_by(app)%>%
  summarise(total = sum(as.numeric(is_attributed)))%>%
  arrange(desc(total))

# Qual os teve mais fraudes
df%>%
  filter(is_attributed == 0)%>%
  group_by(os)%>%
  summarise(total = sum(as.numeric(is_attributed)))%>%
  arrange(desc(total))

# Qual device teve mais fraudes
df%>%
  filter(is_attributed == 0)%>%
  group_by(device)%>%
  summarise(total = sum(as.numeric(is_attributed)))%>%
  arrange(desc(total))

# Qual channel teve mais fraudes
df%>%
  filter(is_attributed == 0)%>%
  group_by(channel)%>%
  summarise(total = sum(as.numeric(is_attributed)))%>%
  arrange(desc(total))

# Valores minimos e maximos das variaveis
range(df$app)
range(df$os)

# Mais um resumo dos dados
summary(df$device)
str(df)

# Balanceando os dados, pois temos muitos mais dados de fraudes do que não fraudes, isso é um problema

# Vamos gerar dados sintéticos, para equilibrar o dataset
dados <- ROSE(is_attributed ~ channel + os + app + device, data = df)$data
View(dados)

# Vendo a proporção, está quase 50/50, já está otimo.
prop.table(table(dados$is_attributed))


# Dividindo os dados para treino e teste
split = sample.split(Y = dados, SplitRatio = 0.7)

dados_treino = subset(dados, split == T)
dados_teste = subset(dados, split == F)


# Removendo valores negativos que foram gerados sinteticamente
View(dados_treino)
dados_treino = dados_treino%>%
  filter(os >= 0)

# Removendo valores negativos que foram gerados sinteticamente
View(dados_teste)
dados_teste = dados_teste %>%
  filter(os >=0)

# Treinando o modelo 
modelo1 = rpart(is_attributed ~., data = dados_treino, method = 'class')

# Previsao do modelo
previsao1 = predict(modelo1, dados_teste, type = 'class')

# Medindo a Accuracia 

# Acuracia de 83,56% na ConfusionMatrix
confusionMatrix(dados_teste$is_attributed, previsao1, positive = '1')

# Acuracia de 83% na curva ROC
roc.curve(dados_teste$is_attributed, previsao1, plotit = T, col = 'red')


# Testando outro modelo 
modelo2 <- randomForest(is_attributed ~.,data = dados_treino)

# Previsao
previsao2 <- predict(modelo2, dados_teste, type = 'class')

# Medindo Acuracia

# Acurácia de 89,53% na ConfusionMatrix
confusionMatrix(dados_teste$is_attributed, previsao2, positive = '1')

# Acurácia de 89,3% na curva ROC
roc.curve(dados_teste$is_attributed, previsao2)


## CONCLUSÃO ##

# Após limpar, analisar e balanciar o nosso dataset, testamos 2 modelos de machine learning, 
# os dois com uma boa performance, porém o segundo teve um aumento significativo da acurácia,
# fazendo com que escolhamos a segunda opção, do modelo Random Forest, com acurácia de 89%







