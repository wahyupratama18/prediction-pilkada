from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pprint import pprint
from flask import jsonify

app = Flask(__name__)

def predict_sentiment(text):
    # Load the model and tokenizer
    model_path = 'model'  # Path to the model folder
    tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(model_path)

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def result():
    # Check if 'text' key exists in the form data or query parameters
    text = request.form.get('text') or request.json.get('text') if request.method == 'POST' else request.args.get('text')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    prediction = predict_sentiment(text)
    sentiment = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}.get(prediction, 'Unknown')

    # Debug statement to print the prediction and sentiment
    print("Prediction:", prediction)
    print("Sentiment:", sentiment)

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)