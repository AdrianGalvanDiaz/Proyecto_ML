import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Cargamos nuestro archivo
df = pd.read_csv('df_final.csv')
x = df.drop('class', axis=1).values
y = df['class'].values

#Hiper-parámetros
w = np.zeros(x.shape[1])
b = 0

#Parámetros
learning_rate = 0.01
epochs = 3000

#Función de gradiente descendiente
def GD(x, y, w, b, learning_rate, n):
    #Iniciamos b y w con zeros 
    loss_w = np.zeros(x.shape[1])
    loss_b = 0.0
    for i in range(n):
        #Cálculo de las predicciones
        hyp = np.dot(x[i], w) + b
        error = y[i] - hyp
        #Derivada parcial de b y de w
        loss_w += -2 * x[i] * (error)
        loss_b += -2 * error

    #Actualizamos los parámetros
    w = w - learning_rate * (1/n) * loss_w
    b = b - learning_rate * (1/n) * loss_b

    return w, b 

#Función que calcula el loss de cada época
def loss(x, y, w, b, n): 
    total_error = 0
    for i in range(n):
        total_error += (y[i] - (np.dot(x[i], w) + b))**2
    return total_error / n

""" #Función main que entrena el modelo y manda a llamar a las funciones para poder hacerlo
def train(x, y, w, b, learning_rate, epochs):
    for e in range(epochs):
        n = len(y)
        w, b = GD(x, y, w, b, learning_rate, n)
        current_loss = loss(x, y, w, b, n)
        print(f'Epoch {e}, Loss {current_loss}, Bias {b}')

w, b = train(x, y, w, b, learning_rate, epochs) """

#Función main que entrena el modelo, manda a llamar a las funciones para poder hacerlo e imprime un plt del entrenamiento del modelo
def train(x, y, w, b, learning_rate, epochs):
    loss_values = []  #Almacenamos los valores de loss
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training')

    for e in range(epochs):
        n = len(y)
        w, b = GD(x, y, w, b, learning_rate, n)
        current_loss = loss(x, y, w, b, n)
        loss_values.append(current_loss)
        ax.plot(loss_values)
        plt.draw()
        plt.pause(0.1) 
        print(f'Epoch {e}, Loss {current_loss}, Bias {b}')
    
    plt.show()

w, b = train(x, y, w, b, learning_rate, epochs)

