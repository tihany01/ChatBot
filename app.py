from flask import Flask, request, jsonify
# Import the main function from the LLM module. 
from LLM import main  
from flask_cors import CORS
from waitress import serve

app = Flask(__name__)
CORS(app)

@app.route('/api/generar_respuesta', methods=['POST'])
def generar_respuesta():
    try:
        # Extract the 'consulta' field from the incoming JSON request.
        consulta = request.json['consulta']
         # Call the main function (imported from LLM) with consulta as an argument and store the result in respuesta.
        respuesta = main(consulta)
        # Return the 'respuesta' wrapped in JSON format.
        return jsonify({'respuesta': respuesta})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000)
