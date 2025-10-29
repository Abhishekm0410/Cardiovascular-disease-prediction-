Cardiovascular Disease Prediction Model

This project uses a trained Machine Learning model to predict the likelihood of Cardiovascular Disease (CVD) based on user input data such as health indicators and medical parameters.

Project Structure

Cardiovascular/
│
├── models/
│ ├── model.pkl → Trained ML model (e.g., Logistic Regression, Random Forest)
│ └── scaler.pkl → Fitted Scaler for input normalization
│
├── cvd_model.py → Main class for prediction (CVDModel)
└── README.md

Model Overview

The CVDModel class handles:

Loading the trained model and scaler

Preprocessing user data using the loaded scaler

Predicting the likelihood of CVD

Returning both class prediction and probability

How It Works

Import and Initialize the Model:

from cvd_model import CVDModel
cvd = CVDModel()


Prepare Input Data:
Provide the input as a list or array (numeric values only).
Example:

input_data = [45, 1, 120, 80, 1, 0, 1, 24.5]


Get Prediction:

result = cvd.predict(input_data)
print(result)


Example Output:

{
  'prediction': 1,           # 1 = CVD likely, 0 = No CVD
  'probability': 0.83        # Model confidence (83%)
}

Dependencies

Install the required libraries:

pip install numpy joblib scikit-learn

File Details
File	Description
cvd_model.py	Main script that loads model & scaler, performs prediction
models/model.pkl	Pre-trained machine learning model
models/scaler.pkl	StandardScaler/MinMaxScaler used during model training
Example Use Case

This module can be integrated into:

Flask or FastAPI backends

Streamlit or React frontends

Healthcare data dashboards

Author

Abhishek Maheshwari
VIT Vellore
GitHub: https://github.com/Abhishekm0410

License

This project is for educational and research purposes.
Feel free to use or modify it with proper attribution.
