# -*- coding: utf-8 -*-
"""
Created on Thu May  2 20:47:30 2024

@author: Matheus
"""

#pip install git+https://github.com/tensorflow/addons.git


import pandas as pd
import tensorflow as tf
import keras
from keras.metrics import *
import tensorflow_addons as tfa
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

#Declarando as métricas

print(f'Versão da biblioteca TensorFlow: {tf.__version__}')

METRICS = [BinaryAccuracy(name = 'accuracy'),
           TruePositives(thresholds = 0.5, name = 'tp'),
           TrueNegatives(thresholds = 0.5, name = 'tn'),
           FalsePositives(thresholds = 0.5, name = 'fp'),
           FalseNegatives(thresholds = 0.5, name = 'fn'),
           PrecisionAtRecall(recall = 0.5, name = 'precision'),
           SensitivityAtSpecificity(0.5, name = 'sensitivity'),
           SpecificityAtSensitivity(sensitivity = 0.5,
                                                  name = 'specificity'),
           Recall(name='recall')]

#Carregando o dataset
data = datasets.load_breast_cancer()

#Mostrando as descrições das colunas do dateset
print(data.DESCR)

#Atribuindo a X todo o conteúdo da base de datas
X = pd.DataFrame(data = data.data,
                 columns = data.feature_names)  # feature_names guarda as informações do cabeçalho que esta na documentação
                                                # do banco de dados
#Retorna os primeiros registros do banco
print(X.head())

#Retorna informações sobre o banco de dados
print(X.info())

#Retorna as colunas do banco de dados
print(X.columns)

#Informa as saídas de acordo com as entradas (Um aprendizado de máquina supervisionado pois temos entradas e saídas)
y = data.target

print(y)

# Imprime o nome de cada coluna (propriedade)
print(data.feature_names)

# Imprime os possíveis resultados da rede neural
print(data.target_names)

# Imprime a dimensão do banco de dados (linhas,colunas)
print(X.shape)

###############################################################################

#                               Próximo passo

###############################################################################

#Definir os valores de treino e de teste

X_treino, X_teste, y_treino, y_teste = train_test_split(X, # Banco de dados
                                                        y, # Saídas
                                                        test_size = 0.2,  # 20% Para a base de teste e 80% p treino
                                                        random_state = 0, # Define a randomização
                                                        stratify = y)     # Configura para q o banco não veja os dados de teste
                                                                          
#Retorna a quantidade de dados e colunas de treino e teste
print(X_treino.shape)
print(X_teste.shape)



clf = KNeighborsClassifier()
clf.fit(X_treino, y_treino)
prediction = clf.predict(X_teste)

print(clf.score(X_treino, y_treino))
print(clf.score(X_teste, y_teste))

print(classification_report(y_teste, prediction))

