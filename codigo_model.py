import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("wine.csv", delimiter=",")

df.head()

df.size

#Nos fijamos que si hay datos duplicados
df[df.duplicated() == True]

#Eliminamos los duplicados
df = df.drop_duplicates()
df.size

#Chequear si hay nulos
nulls = df.isna().sum()
df_null = pd.DataFrame(nulls)
df_null.transpose()

numerical_columns = df.drop('quality', axis=1)

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()

#No se aprecia mucha correlacion entre variables

# Obtener las columnas del DataFrame excepto la variable de salida

for columna in numerical_columns:
    plt.figure()
    df.boxplot(column=columna, color='blue')  # Puedes cambiar el color aquí
    plt.title(f'Boxplot de {columna}')
    plt.show()

#Eliminamos los valores atipicos

# Definir un factor de multiplicación para determinar los límites de los bigotes
outlier_factor = 1.5

# Crear una copia del DataFrame original
df_filtered = df.copy()

# Iterar sobre las columnas categóricas
for columna in numerical_columns:
    # Calcular el rango intercuartílico (IQR)
    q1 = np.percentile(df[columna], 25)
    q3 = np.percentile(df[columna], 75)
    iqr = q3 - q1

    # Calcular los límites de los bigotes
    lower_bound = q1 - outlier_factor * iqr
    upper_bound = q3 + outlier_factor * iqr

    # Filtrar los valores que estén dentro de los límites de los bigotes
    df_filtered = df_filtered[(df_filtered[columna] >= lower_bound) & (df_filtered[columna] <= upper_bound)]

# Mostrar el DataFrame filtrado sin valores atípicos
df_filtered

#Procedemos a configuarar el Pipeline

#Separamos las labels de la base (separamos los x de la y)
X = df_filtered.drop('quality', axis=1)
y = df_filtered['quality']

#dividimos los datos para entrenar y para testear
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

# Definir el pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier())
])

# Definir los hiperparámetros a probar
param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 5, 10],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

# Realizar la búsqueda de hiperparámetros utilizando GridSearchCV
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Obtener los mejores parámetros y el mejor puntaje
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Mejores parámetros:", best_params)
print("Mejor puntaje:", best_score)

#Ahora mostramos una matriz de consufion para ver que tan bien predice el modelo

# Obtener las predicciones del modelo en los datos de prueba
y_pred = grid_search.predict(X_test)

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión como un mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.show()

import pickle

# Entrenar y ajustar el modelo con los mejores parámetros
best_model = pipeline.set_params(**best_params)
best_model.fit(X_train, y_train)


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Cargar el modelo entrenado
#model = joblib.load('modelo.pkl')

# Definir una función para realizar la predicción
def predict(observation):
    # Crear un DataFrame con la observación
    df = pd.DataFrame([observation], columns=X.columns)
    
    # Crear un DataFrame con los nombres de las características
    feature_names = ['alcohol', 'fixed acidity', 'volatile acidity', 'citric acid',
       'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide', 'density', 'pH', 'sulphates']
    df_features = pd.DataFrame(columns=feature_names)
    
    # Añadir una muestra al DataFrame con los nombres de las características
    sample = [0] * len(feature_names)  # Colocar ceros para todas las características
    df_features.loc[0] = sample
    
    # Escalar las características de la observación
    scaler = StandardScaler()
    scaler.fit(df_features)
    df_scaled = scaler.transform(df)
    
    # Realizar la predicción
    prediction = best_model.predict(df_scaled)
    
    print("Predicción:", prediction)

    return prediction

# Observación de ejemplo
#observation = [9.8, 11.2, 0.28, 0.56, 1.9, 0.075, 17.0, 60.0, 0.9980, 3.16, 0.58]
#observation2 = [9.4,	7.4,	0.700,	0.00,	1.9,	0.076,	11.0,	34.0,	0.99780,	3.51,	0.56]

# Realizar la predicción
#prediction = predict(observation)
#prediction = predict(observation2)
#print("Predicción:", prediction)


# Guardar el modelo y el pipeline en un archivo pkl
with open('modelo.pkl', 'wb') as file:
    pickle.dump(best_model, file)


with open('modelo.pkl', 'rb') as file:
    loaded_model = pickle.load(file)






















