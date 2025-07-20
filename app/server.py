import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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

# Model components
bilstm_model = None
naive_bayes_model = None
vectorizer = None
vocab = None
vocab_to_idx = None
preprocess = None
tokenize = None

def load_models():
    """Load both BiLSTM and Naive Bayes models"""
    global bilstm_model, naive_bayes_model, vectorizer, vocab, vocab_to_idx, preprocess, tokenize
    
    print("Loading models...")
    
    # Initialize preprocessing components
    preprocess = Preprocess()
    tokenize = Tokenize()
    
    # Load BiLSTM model
    try:
        # Load vocabulary for BiLSTM
        with open('data/interim/vocab.pkl', 'rb') as f:
            vocab, vocab_to_idx = pickle.load(f)
        
        # Load BiLSTM model
        bilstm_model = BiLSTM(
            vocab_size=len(vocab),
            embed_dim=100,
            hidden_dim=64,
            output_dim=2,
            pad_idx=vocab_to_idx['<pad>']
        )
        
        bilstm_model.load_state_dict(torch.load('models/BiLSTM_SK5Fold_best.pth', map_location=device))
        bilstm_model = bilstm_model.to(device)
        bilstm_model.eval()
        print("BiLSTM model loaded successfully!")
        
    except Exception as e:
        print(f"Failed to load BiLSTM model: {e}")
        bilstm_model = None
    
    # Load Naive Bayes model
    try:
        # Load Naive Bayes model and vectorizer
        with open('models/multinomial_naive_bayes.pkl', 'rb') as f:
            naive_bayes_model = pickle.load(f)
        
        with open('models/best_count_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        print("Naive Bayes model loaded successfully!")
        
    except Exception as e:
        print(f"Failed to load Naive Bayes model: {e}")
        naive_bayes_model = None
        vectorizer = None

def predict_sentiment_bilstm(text):
    """Make prediction using BiLSTM model"""
    global bilstm_model, vocab, vocab_to_idx, preprocess, tokenize
    
    if bilstm_model is None:
        raise Exception("BiLSTM model not loaded")
    
    # Preprocess the input text
    test_samples = preprocess.process_data([text])  # Pass as list
    
    # Tokenize the processed text
    padded = tokenize.tokenize_data(test_samples, vocab, vocab_to_idx, train=False)
    
    # Convert to tensor and move to device
    padded = torch.tensor(padded).to(device)
    
    with torch.no_grad():
        output = bilstm_model(padded)
        # Apply softmax to get probabilities
        probabilities = torch.softmax(output, dim=1)
        # Get probability of positive class (index 1)
        probability = probabilities[0][1].item()
        
        return classify_sentiment(probability)

def predict_sentiment_naive_bayes(text):
    """Make prediction using Naive Bayes model"""
    global naive_bayes_model, vectorizer, preprocess
    
    if naive_bayes_model is None or vectorizer is None:
        raise Exception("Naive Bayes model not loaded")
    
    # Preprocess the input text
    processed_text = preprocess.process_data([text])[0]  # Get single processed text
    
    # Vectorize the text
    text_vector = vectorizer.transform([processed_text])
    
    # Get prediction probabilities
    probabilities = naive_bayes_model.predict_proba(text_vector)[0]
    
    # Assuming the model outputs [negative_prob, positive_prob]
    # Get probability of positive class
    probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
    
    return classify_sentiment(probability)

def classify_sentiment(probability):
    """Common function to classify sentiment based on probability"""
    # Convert probability to 1-10 scale for movie reviews
    scaled_score = 1 + (probability * 9)
    
    # Initialize all sentiment flags
    is_positive = False
    is_negative = False
    is_neutral = False
    
    # Classify based on rules: ≤4 = negative, 4-7 = neutral, ≥7 = positive
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
            model_type = data.get('model', 'bilstm').lower()  # Default to bilstm
        else:
            text = request.form.get('text', '')
            model_type = request.form.get('model', 'bilstm').lower()
        
        if not text or not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Validate model type
        if model_type not in ['bilstm', 'naive_bayes']:
            return jsonify({'error': 'Invalid model type. Use "bilstm" or "naive_bayes"'}), 400
        
        # Make prediction based on selected model
        if model_type == 'bilstm':
            if bilstm_model is None:
                return jsonify({'error': 'BiLSTM model not available'}), 500
            result = predict_sentiment_bilstm(text.strip())
        else:  # naive_bayes
            if naive_bayes_model is None or vectorizer is None:
                return jsonify({'error': 'Naive Bayes model not available'}), 500
            result = predict_sentiment_naive_bayes(text.strip())
        
        return jsonify({
            'success': True,
            'model_used': model_type,
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

@app.route('/models', methods=['GET'])
def get_available_models():
    """Get list of available models"""
    available_models = []
    
    if bilstm_model is not None:
        available_models.append({
            'name': 'bilstm',
            'display_name': 'BiLSTM Neural Network',
            'description': 'Bidirectional LSTM with attention mechanism'
        })
    
    if naive_bayes_model is not None and vectorizer is not None:
        available_models.append({
            'name': 'naive_bayes',
            'display_name': 'Naive Bayes',
            'description': 'Traditional machine learning with TF-IDF features'
        })
    
    return jsonify({
        'available_models': available_models,
        'default_model': 'bilstm' if bilstm_model is not None else 'naive_bayes'
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'bilstm_loaded': bilstm_model is not None,
        'naive_bayes_loaded': naive_bayes_model is not None and vectorizer is not None,
        'device': device
    })

if __name__ == '__main__':
    # Load both models on startup
    print("Loading models...")
    load_models()
    print("Server starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)