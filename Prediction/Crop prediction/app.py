from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

app = Flask(__name__)

# Load the machine learning model for crop recommendation
model = RandomForestClassifier(random_state=42)
csv_file_path = Path(__file__).resolve().parent.joinpath('Crop_recommendation.csv')
df = pd.read_csv(csv_file_path)

# Preparing the features and target variable
X = df.drop(columns=["label"])
y = df["label"]
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    soil_data = request.json  # Assuming the frontend sends JSON data
    # Create a DataFrame from the input data
    new_data = pd.DataFrame([[
        soil_data['N'], 
        soil_data['P'], 
        soil_data['K'], 
        soil_data['temperature'], 
        soil_data['humidity'], 
        soil_data['ph'], 
        soil_data['rainfall']
    ]], columns=X.columns)  # Use the original feature names

    predicted_crop = model.predict(new_data).tolist()  # Convert ndarray to Python list
    return jsonify({"predicted_crop": predicted_crop})

if __name__ == '__main__':
    app.run(debug=True)
