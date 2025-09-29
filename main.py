# importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree


# leer archivo csv "/content/drive/MyDrive/Colab Notebooks/DataSet_Titanic.csv"
df = pd.read_csv("data\\DataSet_Titanic.csv")


# visualizar las primeras 5 filas
print(df.head())


# guardar en variable X los atributos predictores (todas las etiquetas excepto "Sobreviviente")
X = df.drop("Sobreviviente", axis=1)


# guardar en y la etiqueta a predecir ("Sobreviviente")
y = df["Sobreviviente"]


# visualizar x
print(X.head())


# visualizar y
print(y.head())


# Creamos un objeto arbol
arbol = DecisionTreeClassifier(max_depth=4, random_state=42)


# entrenamos a la máquina
arbol.fit(X, y)


# Predecimos sobre nuestro set
pred_y = arbol.predict(X)
# Comaparamos con las etiquetas reales
print(f'Presición: {accuracy_score(pred_y, y)}')

# creamos una matriz de confusión
cm = confusion_matrix(y, pred_y)
print(cm)


# creamos un gráfico para la matriz de confusión
# plot_confusion_matrix(arbol, X, y, , cmap=plt.cm.Blues, values_format='.0f')

# La función plot_confusion_matrix está DEPRECATED, se ha sustituido por otra
disp = ConfusionMatrixDisplay.from_estimator(arbol, X, y, cmap=plt.cm.Blues, values_format='.0f')
plt.show()


# creamos un gráfico para la matriz de confusión normalizada
# plot_confusion_matrix(arbol, X, y, , cmap=plt.cm.Blues, values_format='.2f', normalize=True)

disp = ConfusionMatrixDisplay.from_predictions(y, pred_y, cmap=plt.cm.Blues, values_format='.2g', normalize='true')
plt.show()


# mostramos un árbol gráficamente
plt.figure(figsize=(15,10))
tree.plot_tree(arbol, filled=True, feature_names=X.columns)
plt.show()


# graficamos las importancias en un gráfico de barras
# creamos las variables x (importancias) e y (columnas)
x = arbol.feature_importances_
y = X.columns

# creamos el gráfico
sns.barplot(x=y, y=x)
plt.title('Importancia de las variables')
plt.ylabel('Importancia')
plt.xlabel('Variables')
plt.show()