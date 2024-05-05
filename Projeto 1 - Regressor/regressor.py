# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:47:08 2024

@author: Matheus
"""

# -*- coding: utf-8 -*-

# Importação das bibliotecas, módulos e pacotes
import tensorflow as tf
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model


# Carregamento da base de dados
data = tf.keras.datasets.boston_housing

# Divisão em base para treino e para teste
(x_train, y_train), (x_test, y_test) = data.load_data()

# https://keras.io/api/datasets/boston_housing/

# Tipo de dado
print(type(x_train))

# Formato dos dados de treino
print(x_train.shape)

# Formato dos dados de teste
print(x_test.shape)

# Primeira amostra da base de treino
print(x_train[0])

# Primeira amostra da base de teste
print(y_test[0])

# Normalização dos dados através da média e desvio padrão
media = x_train.mean(axis = 0)
desvio = x_train.std(axis = 0)

x_train = (x_train - media) / desvio
x_test = (x_test - media) / desvio

# https://numpy.org/doc/stable/reference/generated/numpy.mean.html
# https://numpy.org/doc/stable/reference/generated/numpy.std.html

# Definição do modelo
model = Sequential([
  Dense(units = 64,
        activation = 'relu',
        input_shape = [13]),
  Dense(units = 64,
        activation = 'relu'),
  Dense(units = 1)
])

# https://keras.io/api/models/sequential/#sequential-class
# https://keras.io/api/layers/core_layers/dense/
# https://keras.io/api/layers/activations/

#Mostra o número de parâmetros, os que são treináveis e os que não são
model.summary()

#Plotando o esqueleto do modelo com entradas e saídas e número de neurônios de cada camada
plot_model(model,
           to_file = 'model.png',
           show_shapes = True,
           show_layer_names = False)

#Compilando o modelo
model.compile(optimizer = 'adam',
              loss = 'mse', # erro quadrático médio
              metrics = ['mae']) # erro absoluto médio

#O erro quadrático médio eleva a penalização do erro ao quadrado para que ele não erre novamente

# https://keras.io/api/optimizers/
# https://keras.io/api/losses/
# https://keras.io/api/metrics/


#Treinando o modelo
history = model.fit(x_train,
                    y_train,
                    epochs = 100,
                    validation_split = 0.2)

print(history.history.keys())
print(history.history)  


loss, mae = model.evaluate(x_test,
                           y_test)


#Plotando a taxa de perda do modelo
plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Taxa de Perda',
            'Taxa de Perda (Validacao)'],
           loc = 'upper right', fontsize = 'x-large')
plt.xlabel('Epocas de processamento', fontsize=16)
plt.ylabel('Valor', fontsize=16)
plt.title('Taxa de Perda', fontsize=18)
plt.show()

#Plotando o erro absoluto do modelo
plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.plot(history.history['mae'])
plt.legend(['Erro Absoluto'],
           loc = 'upper right', fontsize = 'x-large')
plt.xlabel('Epocas de processamento', fontsize=16)
plt.ylabel('Valor', fontsize=16)
plt.title('Erro Absoluto Médio', fontsize=18)
plt.show()

#Testando a entrada número 10 dos testes
x_new = x_test[:10]
y_pred = model.predict(x_new)
print(y_pred[0])

#Salvando o modelo
model.save('/Curso Redes Neurais/regressor.h5')
model.save_weights('/Curso Redes Neurais/regressor.weights.h5')

#Carregando o modelo para aplicar testes
model = Sequential()
model = load_model('/Curso Redes Neurais/regressor.h5')
model.load_weights('/Curso Redes Neurais/regressor.weights.h5')
#nova_amostra = '/content/drive/MyDriv0e/arquivo.ext'
#resultado_teste = model.predict(nova_amostra)

#print(resultado_teste)