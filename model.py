import joblib
import numpy as np

class CVDModel:
    def __init__(self):
        self.model = joblib.load('models/model.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
    
    def predict(self, data):
        # Convert input data to numpy array and scale it
        data_array = np.array(data).reshape(1, -1)
        scaled_data = self.scaler.transform(data_array)
        
        # Make prediction
        prediction = self.model.predict(scaled_data)
        probability = self.model.predict_proba(scaled_data)
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1])  # Probability of having CVD
        }