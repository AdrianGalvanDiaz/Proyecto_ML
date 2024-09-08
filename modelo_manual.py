import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Cargamos nuestro archivo
df = pd.read_csv('df_final.csv')
x = df.drop('class', axis=1).values
y = df['class'].values

# Dividimos el dataset en conjunto de entrenamiento + validación (80%) y conjunto de prueba (20%)
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Luego dividimos el conjunto de entrenamiento + validación en entrenamiento (80%) y validación (20%)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42)
# Nota: 0.25 * 0.8 = 0.2, por lo que x_val e y_val tendrán el 20% de los datos originales

# Hiper-parámetros
w = np.zeros(x_train.shape[1])
b = 0

# Parámetros
learning_rate = 0.01  # Ajustado en base a la discusión anterior
epochs = 6000

# Función Sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función de gradiente descendiente para Regresión Logística
def GD(x, y, w, b, learning_rate, n):
    for i in range(n):
        # Cálculo de las predicciones usando la función sigmoide
        z = np.dot(x[i], w) + b
        hyp = sigmoid(z)
        
        # Derivada parcial de w y b
        error = y[i] - hyp
        loss_w = -x[i] * error
        loss_b = -error
        
        # Actualizamos los parámetros
        w = w - learning_rate * (1/n) * loss_w
        b = b - learning_rate * (1/n) * loss_b
    
    return w, b  

# Función que calcula la pérdida de cada época (Entropía Cruzada)
def loss(x, y, w, b, n):
    total_error = 0
    for i in range(n):
        z = np.dot(x[i], w) + b
        hyp = sigmoid(z)
        total_error += -y[i]*np.log(hyp) - (1-y[i])*np.log(1-hyp)
    return total_error / n

# Listas para almacenar las pérdidas de entrenamiento y validación
losses_train = []
losses_val = []

# Entrenamiento del modelo
for epoch in range(epochs):
    w, b = GD(x_train, y_train, w, b, learning_rate, len(y_train))
    current_loss_train = loss(x_train, y_train, w, b, len(y_train))
    current_loss_val = loss(x_val, y_val, w, b, len(y_val))
    losses_train.append(current_loss_train)
    losses_val.append(current_loss_val)
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Training Loss: {current_loss_train}, Validation Loss: {current_loss_val}')

# Gráfica de Loss vs. Épocas para entrenamiento y validación
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), losses_train, label='Training Loss')
plt.plot(range(epochs), losses_val, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.show()

# Predicción
def predict(x, w, b):
    z = np.dot(x, w) + b
    return sigmoid(z) >= 0.5

# Evaluación del modelo en el conjunto de prueba
predicciones_test = predict(x_test, w, b)
test_accuracy = np.mean(predicciones_test == y_test)
test_loss = loss(x_test, y_test, w, b, len(y_test))
print(f'Precisión en el conjunto de prueba: {test_accuracy * 100:.2f}%')

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, predicciones_test)

# Graficar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Evaluación del modelo en el conjunto de entrenamiento y validación
train_accuracy = np.mean(predict(x_train, w, b) == y_train)
val_accuracy = np.mean(predict(x_val, w, b) == y_val)

# Resumen final para el diagnóstico del modelo
final_training_loss = losses_train[-1]
final_validation_loss = losses_val[-1]

print(f"\nFinal Training Loss: {final_training_loss}")
print(f"Final Validation Loss: {final_validation_loss}")
print(f"Final Test Loss: {test_loss}")

print(f"Precisión en el conjunto de entrenamiento: {train_accuracy * 100:.2f}%")
print(f"Precisión en el conjunto de validación: {val_accuracy * 100:.2f}%")

