from flask import Flask, render_template, request, redirect, url_for
from models.models import CVDModel
import numpy as np

app = Flask(__name__)
model = CVDModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = float(request.form['age'])
        cholesterol = float(request.form['cholesterol'])
        blood_pressure = float(request.form['blood_pressure'])
        heart_rate = float(request.form['heart_rate'])
        st_depression = float(request.form['st_depression'])
        chest_pain = int(request.form['chest_pain'])
        
        # Make prediction
        input_data = [age, cholesterol, blood_pressure, heart_rate, st_depression, chest_pain]
        result = model.predict(input_data)
        
        # Prepare result message
        if result['prediction'] == 1:
            diagnosis = "High Risk of Cardiovascular Disease"
            recommendation = "Please consult with a cardiologist immediately."
        else:
            diagnosis = "Low Risk of Cardiovascular Disease"
            recommendation = "Maintain a healthy lifestyle and regular checkups."
        
        return render_template('result.html', 
                             diagnosis=diagnosis,
                             probability=f"{result['probability']*100:.2f}%",
                             recommendation=recommendation)
    
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)