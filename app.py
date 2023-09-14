from flask import Flask, request, jsonify
# Suponiendo que la función main está definida en un módulo llamado LLM
from LLM import main  
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/generar_respuesta', methods=['POST'])
def generar_respuesta():
    try:
        consulta = request.json['consulta']
        respuesta = main(consulta)
        return jsonify({'respuesta': respuesta})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)

