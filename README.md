# pd_pec2

Primero hay que descargar la imagen del DockerHub, para eso se ejecuta el siguiente comando.

   docker pull facurod388/pd_pec2:pec2facundorodriguez

Luego hay que generar el contenedor de docker utilizando la imagen descargada
 
   docker build -t facurod388/pd_pec2:pec2facundorodriguez .

Luego ejecutamos el contenedor en este caso en el puerto 5000.

   docker run -p 5000:5000 facurod388/pd_pec2:pec2facundorodriguez

Luego de ejecutar esa linea, debemos realizar una consulta de tipo POST a la url http://localhost:5000/predict, Para esto podemos utilizar postman o algun programa similar.

![image](https://github.com/FacundoRR/pd_pec2/assets/69487504/534fa5f7-bf44-4640-b90a-270972550285)

En este caso se envia como parametro la observacion ["9.8", "11.2", "0.28", "0.56", "1.9", "0.075", "17.0", "60.0", "0.9980", "3.16", "0.58"]
 
En la respuesta se puede observar el resultado dentro de "perdiction", en este caso fue "good".


Tambien se puede realizar desde la terminal utilizando por ejemplo Invoke-WebRequest.
Para esto debemos abrir una nueva terminal y ejecutar la siguiente linea:

Invoke-WebRequest -Method POST -Uri "http://localhost:5000/predict" -Body '["9.8", "11.2", "0.28", "0.56", "1.9", "0.075", "17.0", "60.0", "0.9980", "3.16", "0.58"]' -ContentType "application/json"


Breve descripción del codigo del modelo de ML

Utilizamos para el ejercicio la base de datos wine para predecir la calidad del vino. 
Después de chequear la base, procedemos a borrar registros duplicados y registros outliers. Una vez hecho estos cambios procedemos a separar los labels del df original y separar en dos para entrenar y testear el modelo. 
Definimos a su vez un pipeline escalando los los regresores numericos y utilizamos un modelo random forest para entrenarlo. También realizmaos la búsqueda de los mejores hyperparametros y terminamos mostrando el scrore y una matriz de confusión para ver qué tan bueno fue nuestro modelo. 

Al final se guarda el modelo como pkl para luego proceder a hacer los demás pasos. 
