from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
import numpy as np
app = Flask(__name__)
CORS(app)

model = joblib.load("random_forest_model_compressed.pkl")
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('one_hot_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.json
        konut_type = data['konut_type']
        oda_sayisi = data['oda_sayisi']
        metre_kare = data['metre_kare']
        bina_yasi = data['bina_yasi']

        
        input_data = pd.DataFrame({
            'oda_sayisi': [oda_sayisi],
            'metre_kare': [metre_kare],
            'bina_yasi': [bina_yasi]
        })

        
        konut_encoded = encoder.transform([[konut_type]])
        konut_df = pd.DataFrame(konut_encoded, columns=encoder.get_feature_names_out(['konut']))

        
        input_final = pd.concat([input_data, konut_df], axis=1)

        
        input_scaled = scaler.transform(input_final)

        
        prediction = np.expm1(model.predict(input_scaled))


    
        return jsonify({'predicted_price': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
