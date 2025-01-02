# Script para analizar el dataset y aplicar el algoritmo KNN

# Paso 1: Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Paso 2: Cargar el dataset (asegúrate de subir el archivo a Colab antes de ejecutar este paso)
from google.colab import files
uploaded = files.upload()  # Subir el archivo "loan_data.csv"

# Leer el archivo CSV
data = pd.read_csv("loan_data.csv")

# Paso 3: Explorar los datos
print("Vista previa del dataset:")
print(data.head())
print("\nInformación del dataset:")
print(data.info())

# Paso 4: Preprocesamiento de datos
# Verificar valores nulos
data = data.dropna()

# Codificar variables categóricas (si existen)
data = pd.get_dummies(data, drop_first=True)

# Dividir las variables independientes y dependientes
X = data.drop("loan_status", axis=1)  # Ajustar según la columna objetivo
y = data["loan_status"]

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar las variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Paso 5: Aplicar el algoritmo KNN
# Inicializar el modelo con k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Realizar predicciones
y_pred = knn.predict(X_test)

# Paso 6: Evaluar el modelo
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("\nPrecisión del modelo:")
print(f"{accuracy_score(y_test, y_pred):.2f}")

# Paso 7: Gráfico de validación para elegir el mejor valor de k
accuracies = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), accuracies, marker='o')
plt.xlabel("Valores de k")
plt.ylabel("Precisión")
plt.title("Selección del Mejor Valor de k")
plt.show()
