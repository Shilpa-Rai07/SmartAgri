from django.shortcuts import render
import joblib
import os
import numpy as np

# Project root: D:\SMARTAGRI
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# ML model path: D:\SMARTAGRI\ml\crop_model.pkl
MODEL_PATH = os.path.join(PROJECT_ROOT, 'ml', 'crop_model.pkl')

# Load model ONCE using joblib (Python 3.13 safe)
model = joblib.load(MODEL_PATH)


def predict_crop(request):
    prediction = None

    if request.method == 'POST':
        n = float(request.POST.get('nitrogen'))
        p = float(request.POST.get('phosphorus'))
        k = float(request.POST.get('potassium'))
        temp = float(request.POST.get('temperature'))
        humidity = float(request.POST.get('humidity'))
        ph = float(request.POST.get('ph'))
        rainfall = float(request.POST.get('rainfall'))

        features = np.array([[n, p, k, temp, humidity, ph, rainfall]])
        prediction = model.predict(features)[0]

    return render(request, 'agri/predict.html', {'prediction': prediction})
