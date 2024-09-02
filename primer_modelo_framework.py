import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('df_final.csv')

X = df.drop('class', axis=1)  
y = df['class']

# Dividir los datos en conjunto de entrenamiento, validación y prueba
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Crear el modelo de Random Forest
# rf_model = RandomForestClassifier(n_estimators=1, random_state=42)
# rf_model = RandomForestClassifier()
rf_model = RandomForestClassifier(
    n_estimators=20,
    max_depth=10,
    # max_features='sqrt',
    # bootstrap=False,
    random_state=42,
)


# Entrenar el modelo
rf_model.fit(X_train, y_train)


# Evaluar el modelo en los conjuntos de validación y prueba
y_pred_val = rf_model.predict(X_val)
y_pred_test = rf_model.predict(X_test)


# Calcular y mostrar la precisión en los conjuntos de validación y prueba
val_accuracy = accuracy_score(y_val, y_pred_val)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f'Precisión en el conjunto de validación: {val_accuracy * 100:.2f}%')
print(f'Precisión en el conjunto de prueba: {test_accuracy * 100:.2f}%')


# Mostrar el reporte de clasificación y la matriz de confusión
print("\nReporte de clasificación en el conjunto de prueba:")
print(classification_report(y_test, y_pred_test))


# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Random Forest')
plt.show()


# Importancia de las características
# features = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['Importance'])
# print(features.head(20))


# Generar la curva de aprendizaje
train_sizes, train_scores, val_scores = learning_curve(rf_model, X_train, y_train, cv=5,
                                                      scoring='accuracy', n_jobs=-1,
                                                      train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)


# Calcular medias y desviaciones estándar
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)


train_std = np.std(train_scores, axis=1)
val_std = np.std(val_scores, axis=1)


# Graficar la curva de aprendizaje
plt.plot(train_sizes, train_mean, label='Training Accuracy')
plt.plot(train_sizes, val_mean, label='Validation Accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Training Data Size')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Learning Curve')
plt.show()