import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from pandas.plotting import scatter_matrix

def main():
    print('COSO RARO HECHO POR LA EVIE, UN SALUDO A MINMIN')
    print('https://github.com/eviecamila/data-mining')
    # Cargar el dataset Iris
    print('Cargando el dataset Iris...')
    print('credits to: https://github.com/mwaskom/seaborn-data/blob/master/iris.csv')
    ds = pd.read_csv('iris.csv')
    
    # Mostrar los primeros registros
    print('Mostrando los primeros registros del dataset:')
    print(ds.head())
    
    # Mostrar el resumen del dataset
    print('\nResumen del dataset:')
    print(ds.describe())
    
    # Información del dataset
    print('\nInformación del dataset:')
    ds.info()
    
    # Agrupación por especie
    print('\nAgrupación de clases por especie:')
    print(ds.groupby('species').size())
    
    # Gráfico de caja
    print('\nGenerando gráfico de caja para las características del dataset...')
    ds.plot(kind='box', sharex=False, sharey=False)
    plt.show()
    
    # Histograma
    print('\nGenerando histograma de las características del dataset...')
    ds.hist(edgecolor='red', linewidth=1.2)
    plt.show()
    
    # Gráfico de violín para la longitud de los pétalos por especie
    print('\nGenerando gráfico de violín (Seaborn) para la longitud de los pétalos por especie...')
    sbn.violinplot(data=ds, x='species', y='petal_length')
    plt.show()
    
    # Matriz de diagramas de dispersión
    print('\nGenerando matriz de diagramas de dispersión...')
    scatter_matrix(ds, figsize=(15, 15))
    plt.show()
    
    # Gráfico de par con correlación de variables
    print('\nGenerando gráfico de pares (Seaborn)...')
    sbn.pairplot(ds, hue='species')
    plt.show()
    
    # Separación de datos para entrenamiento y prueba
    print('\nSeparando los datos en variables de entrada y de salida...')
    X = ds.iloc[:, :-1].values
    y = ds.iloc[:, -1].values
    
    print('\nDividiendo los datos en conjunto de entrenamiento y prueba...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Entrenamiento del modelo de Árbol de Decisión
    print('\nEntrenando el modelo de Árbol de Decisión...')
    cls_tree = DecisionTreeClassifier()
    cls_tree.fit(X_train, y_train)
    
    # Predicciones y evaluación del modelo
    print('\nHaciendo predicciones con el modelo entrenado...')
    y_pred = cls_tree.predict(X_test)
    
    print('\nMatriz de confusión:')
    print(confusion_matrix(y_test, y_pred))
    
    print('\nReporte de clasificación:')
    print(classification_report(y_test, y_pred))
    
    # Gráfico del árbol de decisión
    print('\nGenerando gráfico del Árbol de Decisión...')
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_tree(cls_tree, filled=True, feature_names=ds.columns[:-1])
    plt.show()

