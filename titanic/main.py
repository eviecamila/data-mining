import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Carga del dataset
data = pd.read_csv("titanic.csv")

# Preprocesamiento de los datos
data = data[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Survived']]
data = data.dropna()
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Variables independientes y dependientes
X = data[['Pclass', 'Sex', 'Age', 'Parents/Children Aboard']]
y = data['Survived']

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluar el modelo
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))

print("Efectividad en entrenamiento:", train_accuracy)
print("Efectividad en prueba:", test_accuracy)

# Guardar el modelo entrenado
with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)

