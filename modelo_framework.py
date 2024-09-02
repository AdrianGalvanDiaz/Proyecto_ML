import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('df_final.csv')

X = df.drop('class', axis=1)  
y = df['class']

# Dividir los datos en conjunto de entrenamiento, validación y prueba
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Parámetros de RandomForest (max_depth, n_estimators)
# parameters = np.array([[2,1],[2,10],[2,100],[5,1],[5,10],[5,100],[15,1],[15,10],[15,100]])
parameters = np.array([[1,1],[1,5],[1,10],[3,1],[3,5],[3,10],[5,1],[5,5],[5,10]])

# Inicializar listas para almacenar los modelos y las precisiones
models = []
train_scores = []
val_scores = []

# Entrenar los modelos para cada combinación de parámetros
for params in parameters:    
    # Entrenar el modelo
    forest = RandomForestClassifier(max_depth=params[0], n_estimators=params[1], random_state=42)
    model = forest.fit(X_train, y_train)
    
    # Calcular y almacenar los scores
    train_score = accuracy_score(y_train, model.predict(X_train))
    val_score = accuracy_score(y_val, model.predict(X_val))
    models.append(model)
    train_scores.append(train_score)
    val_scores.append(val_score)
    
    # Imprimir los scores
    print(f"max_depth={params[0]} \t n_estimators={params[1]} \t train_score={train_score:.4f} \t val_score={val_score:.4f}")

# Reshape de los scores para visualización
max_depth = np.unique(parameters[:,0])
n_estimators  = np.unique(parameters[:,1])
train_scores_2d = np.around(np.array(train_scores).reshape(3,3), decimals=4)
val_scores_2d =  np.around(np.array(val_scores).reshape(3,3), decimals=4)

# Función para graficar el heatmap
def plot_heap_map(X, x_params, y_params, title, ax):
    im = ax.imshow(X)
    ax.set_xticks(np.arange(len(x_params)))
    ax.set_yticks(np.arange(len(y_params)))
    ax.set_xticklabels(x_params)
    ax.set_yticklabels(y_params)
    for i in range(len(x_params)):
        for j in range(len(y_params)):
            text = ax.text(j, i, X[i, j], ha="center", va="center", color="r")
    ax.set_title(title)

# Graficar los heatmaps para train y validation scores
fig, axes = plt.subplots(1, 2, figsize=(15, 15))
plot_heap_map(train_scores_2d, n_estimators, max_depth, 'Train Scores', axes[0])
plot_heap_map(val_scores_2d, n_estimators, max_depth, 'Validation Scores', axes[1])
plt.show()

# Calcular y mostrar el accuracy del conjunto de prueba
best_model = models[np.argmax(val_scores)]
y_pred_test = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f'Accuracy en el conjunto de prueba: {test_accuracy * 100:.2f}%')
