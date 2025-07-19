import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

pkl_vectorizer = pickle.load(open('best_vectorizer.pkl','rb'))
pkl_model = pickle.load(open('nb_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
 
        df_sample = pd.DataFrame([[text]], columns=['text'])
        X_sample = pkl_vectorizer.transform(df_sample['text'])
        output = pkl_model.predict(X_sample)
        return jsonify({'prediction': str(output[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
