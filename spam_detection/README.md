# 🛡️ Spam Detection AI

A professional real-time SMS/Message spam detection system using Machine Learning with a beautiful web interface.

## Features

✅ **Real-time Spam Detection** - Instantly check if a message is spam  
✅ **Professional UI** - Clean, modern, and responsive design  
✅ **ML-Powered** - Using TF-IDF + Multinomial Naive Bayes classifier  
✅ **Probability Scores** - See detailed confidence levels  
✅ **Great Accuracy** - Trained on SMS spam dataset  
✅ **REST API** - Can be integrated with other applications  

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify the Model

The trained model should already exist as `spam_pipeline.joblib`. If you need to retrain:

```bash
python Spam_dectation.py --data sms.tsv --train
```

## How to Use

### Option 1: Web Interface (Recommended)

1. **Start the Flask Backend Server:**
```bash
python app.py
```

The server will start on `http://localhost:5000`

2. **Open the Web Interface:**

Open the `index.html` file in your web browser:
- You can directly open the file in your browser (drag and drop into browser)
- Or serve it locally using Python:
```bash
python -m http.server 8000
```
Then visit: `http://localhost:8000/index.html`

3. **Use the Interface:**
- Type or paste a message in the text area
- Click "Check Message" button (or press Ctrl+Enter)
- View the result with confidence percentage
- Try multiple messages - the app stays running!

### Option 2: Command Line

For single message prediction:
```bash
python Spam_dectation.py --predict --text "Your message here"
```

For batch prediction from file:
```bash
python Spam_dectation.py --predict --file messages.txt
```

### Option 3: API (for developers)

The Flask backend provides REST API endpoints:

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Your message here"}'
```

Response:
```json
{
  "message": "Your message here",
  "is_spam": false,
  "spam_probability": 0.15,
  "ham_probability": 0.85,
  "confidence": 85.0
}
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:5000/api/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"messages": ["message1", "message2", "message3"]}'
```

## Understanding Results

- **Green (HAM)**: Legitimate message ✅
- **Red (SPAM)**: Spam/Phishing message ⚠️
- **Confidence**: How certain the AI model is (0-100%)
- **Probability Scores**: Detailed breakdown of spam vs. legitimate probability

## Files

- `app.py` - Flask backend API server
- `index.html` - Web interface (open in browser)
- `Spam_dectation.py` - Training and prediction script
- `spam_pipeline.joblib` - Pre-trained ML model
- `sms.tsv` - Training dataset
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.7+
- Flask & Flask-CORS
- scikit-learn
- pandas
- joblib

## Troubleshooting

**"Connection refused" error in web interface:**
- Make sure Flask server is running: `python app.py`
- Server should be on http://localhost:5000
- Check if port 5000 is not blocked

**"Model file not found" error:**
- Ensure `spam_pipeline.joblib` exists in the project directory
- If missing, retrain: `python Spam_dectation.py --data sms.tsv --train`

**Web interface not showing results:**
- Check browser console (F12) for JavaScript errors
- Ensure Flask server is running
- Check if CORS is enabled (it is in app.py)

## Performance

- **Model Accuracy**: ~97% on test dataset
- **Response Time**: <100ms per prediction
- **Memory Usage**: ~5MB (model size)

## Future Enhancements

- [ ] Deep Learning model (LSTM)
- [ ] Multiple language support
- [ ] Email spam detection
- [ ] Database logging of predictions
- [ ] Admin dashboard
- [ ] Docker containerization

## License

This project is open for educational and professional use.

---

Developed with ❤️ for AI/ML learning
