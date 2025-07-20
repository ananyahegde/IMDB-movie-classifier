import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from definitions import ROOT_DIR
os.chdir(ROOT_DIR)

from src.inputs.preprocess import Preprocess
from src.features.tokens import Tokenize

app = Flask(__name__)
CORS(app) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(0.4)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embed_dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# Load model and vocab at startup
model = None
vocab = None
vocab_to_idx = None
preprocess = None
tokenize = None

def load_model():
    """Load model and preprocessing components"""
    global model, vocab, vocab_to_idx, preprocess, tokenize
    
    # Load vocabulary
    with open('data/interim/vocab.pkl', 'rb') as f:
        vocab, vocab_to_idx = pickle.load(f)
    
    # Initialize preprocessing components
    preprocess = Preprocess()
    tokenize = Tokenize()
    
    # Load model
    model = BiLSTM(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden_dim=64,
        output_dim=2,
        pad_idx=vocab_to_idx['<pad>']
    )
    
    model.load_state_dict(torch.load('models/BiLSTM_SK5Fold_best.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")

def predict_sentiment(text):
    """Make prediction using the loaded model"""
    global model, vocab, vocab_to_idx, preprocess, tokenize
    
    if model is None:
        raise Exception("Model not loaded")
    
    # Preprocess the input text
    test_samples = preprocess.process_data([text])  # Pass as list
    
    # Tokenize the processed text
    padded = tokenize.tokenize_data(test_samples, vocab, vocab_to_idx, train=False)
    
    # Convert to tensor and move to device
    padded = torch.tensor(padded).to(device)
    
    with torch.no_grad():
        output = model(padded)
        # Apply softmax to get probabilities
        probabilities = torch.softmax(output, dim=1)
        # Get probability of positive class (index 1)
        probability = probabilities[0][1].item()
        
        # Convert probability to 1-10 scale for movie reviews
        scaled_score = 1 + (probability * 9)
        
        # Initialize all sentiment flags
        is_positive = False
        is_negative = False
        is_neutral = False
        
        # Classify based on your rules: ≤4 = negative, 4-7 = neutral, ≥7 = positive
        if scaled_score <= 4:
            sentiment = 'negative'
            is_negative = True
        elif scaled_score >= 7:
            sentiment = 'positive'
            is_positive = True
        else:
            sentiment = 'neutral'
            is_neutral = True
            
        return {
            'sentiment': sentiment,
            'score': min(10, max(1, scaled_score)),  # Ensure 1-10 range
            'raw_probability': probability,
            'is_positive': is_positive,
            'is_negative': is_negative,
            'is_neutral': is_neutral,
            'confidence': abs(probability - 0.5) * 2  # Convert to 0-1 confidence
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
        else:
            text = request.form.get('text', '')
        
        if not text or not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Make prediction
        result = predict_sentiment(text.strip())
        
        return jsonify({
            'success': True,
            'sentiment': result['sentiment'],
            'score': round(result['score'], 1),
            'raw_probability': result['raw_probability'],
            'is_positive': result['is_positive'],
            'is_negative': result['is_negative'],
            'is_neutral': result['is_neutral'],
            'confidence': round(result['confidence'], 3),
            'text_length': len(text)
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device
    })

if __name__ == '__main__':
    # Load model and vectorizer on startup
    print("Loading model...")
    load_model()
    print("Server starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)