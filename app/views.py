from django.shortcuts import render
import numpy as np
import pickle

with open('app/svm_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('app/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def home(request):
    result = None
    if request.method == 'POST':
        pregnancies = int(request.POST['pregnancies'])
        glucose = int(request.POST['glucose'])
        blood_pressure = int(request.POST['blood_pressure'])
        skin_thickness = int(request.POST['skin_thickness'])
        insulin = int(request.POST['insulin'])
        bmi = float(request.POST['bmi'])
        dpf = float(request.POST['dpf'])
        age = int(request.POST['age'])

        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        standardized_data = scaler.transform(input_data)
        prediction = classifier.predict(standardized_data)
        
        if prediction[0] == 0:
            result = "The person is not diabetic"
        else:
            result = "The person is diabetic"

    return render(request, 'home.html', {'result': result})
