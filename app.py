from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Load your trained model, scaler, and encoder
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('one_hot_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the frontend
        data = request.json
        konut_type = data['konut_type']
        oda_sayisi = data['oda_sayisi']
        metre_kare = data['metre_kare']
        bina_yasi = data['bina_yasi']

        # Prepare the input data
        input_data = pd.DataFrame({
            'oda_sayisi': [oda_sayisi],
            'metre_kare': [metre_kare],
            'bina_yasi': [bina_yasi]
        })

        # Encode the 'konut_type'
        konut_encoded = encoder.transform([[konut_type]])
        konut_df = pd.DataFrame(konut_encoded, columns=encoder.get_feature_names_out(['konut']))

        # Combine input data with the encoded data
        input_final = pd.concat([input_data, konut_df], axis=1)

        # Scale the input data
        input_scaled = scaler.transform(input_final)

        # Make a prediction
        prediction = model.predict(input_scaled)

        # Return the predicted price as JSON
        return jsonify({'predicted_price': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
