from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = "spam_pipeline.joblib"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

model = load_model()

@app.route('/')
def home():
    return "Spam Detection API is running!"

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get prediction and probability
        prediction = model.predict([message])[0]
        probability = model.predict_proba([message])[0]
        
        result = {
            'message': message,
            'is_spam': bool(prediction == 1),
            'spam_probability': float(probability[1]),
            'ham_probability': float(probability[0]),
            'confidence': float(max(probability) * 100)
        }
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not messages or not isinstance(messages, list):
            return jsonify({'error': 'Messages must be a non-empty list'}), 400
        
        # Filter out empty messages
        messages = [m.strip() for m in messages if m.strip()]
        
        if not messages:
            return jsonify({'error': 'No valid messages provided'}), 400
        
        predictions = model.predict(messages)
        probabilities = model.predict_proba(messages)
        
        results = []
        for msg, pred, proba in zip(messages, predictions, probabilities):
            results.append({
                'message': msg,
                'is_spam': bool(pred == 1),
                'spam_probability': float(proba[1]),
                'ham_probability': float(proba[0]),
                'confidence': float(max(proba) * 100)
            })
        
        return jsonify({'results': results}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

