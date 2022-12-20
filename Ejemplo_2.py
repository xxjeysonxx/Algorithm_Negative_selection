# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Cargar los datos
data = pd.read_csv("data.csv")
X = data[["feature1", "feature2", "feature3"]]
y = data["target"]

# Dividir los datos en conjunto de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Inicializar un modelo de regresión logística
model = LogisticRegression()

# Entrenar el modelo usando el conjunto de entrenamiento
model.fit(X_train, y_train)

# Calcular la precisión del modelo en el conjunto de prueba
accuracy = model.score(X_test, y_test)
print("Precisión:", accuracy)

# Identificar los ejemplos del conjunto de entrenamiento que fueron clasificados incorrectamente
y_pred = model.predict(X_train)
wrong_examples = [i for i in range(len(y_pred)) if y_pred[i] != y_train[i]]

# Eliminar los ejemplos identificados del conjunto de entrenamiento
X_train = X_train.drop(wrong_examples)
y_train = y_train.drop(wrong_examples)

# Entrenar el modelo nuevamente con el conjunto de entrenamiento actualizado
model.fit(X_train, y_train)

# Calcular la precisión del modelo en el conjunto de prueba
accuracy = model.score(X_test, y_test)
print("Precisión actualizada:", accuracy)

#* En este ejemplo, se cargan los datos desde un archivo CSV y se utilizan para entrenar
#  un modelo de regresión logística. Luego, se calcula la precisión del modelo en el conjunto
#  de prueba y se identifican los ejemplos del conjunto de entrenamiento que fueron clasificados
#  incorrectamente. Finalmente, se eliminan estos ejemplos del conjunto de entrenamiento y se vuelve
#  a entrenar el modelo para calcular la precisión actualizada.#*