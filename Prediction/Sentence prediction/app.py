from flask import Flask, render_template, request, jsonify
from model import predictive_model  # Import the predictive model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['inputText']
    prediction = predictive_model.predict(user_input)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
