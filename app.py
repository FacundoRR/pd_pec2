from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Cargar el modelo desde el archivo pkl
with open('modelo.pkl', 'rb') as file:
    modelo = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos de la solicitud
    data = request.json

    print(data)
    print(data)
    # Realizar la predicción utilizando el modelo cargado
    prediction = modelo.predict([data])

    # Devolver la respuesta como JSON
    response = {'prediction': prediction.tolist()}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
