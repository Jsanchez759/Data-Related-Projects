from flask import Flask, request, jsonify
import joblib, json, os
import numpy as np

app = Flask(__name__)
first_call = True

def predict_model(model, input):
    return 'Launch' if model.predict([input])[0] == 1 else 'No Launch'

def prepare_input(PayloadMass, Flights, GridFins, Reused, Legs, ReusedCount, Orbit, LaunchSite, columns_model, scaler_model):
    input = np.zeros(len(list(columns_model['columns'])))
    loc_index_orbit= np.where(list(columns_model['columns']) == Orbit)[0]
    loc_index_launchsite = np.where(list(columns_model['columns']) == LaunchSite)[0]
    input[0] = scaler_model.transform([[PayloadMass]])[0][0]
    input[1] = Flights
    input[2] = GridFins
    input[3] = Reused
    input[4] = Legs
    input[5] = ReusedCount
    if loc_index_orbit >= 0:
        input[loc_index_orbit] = 1
    if loc_index_launchsite >= 0:
        input[loc_index_launchsite] = 1
    return input

def get_model():
    return joblib.load(open('models/model.pkl', 'rb')), joblib.load(open('models/scaler_model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Hello, World!'})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global first_call
    if first_call:
        first_call = False
        #os.chdir(os.getcwd() + '\Machine_Learning\Classification\spaceX_classification_project')
    columns_model = json.load(open('models/columns_model.json'))
    PayloadMass = float(request.json['PayloadMass'])
    Flights = int(request.json['Flights'])
    GridFins = int(request.json['GridFins'])
    Reused = int(request.json['Reused'])
    Legs = int(request.json['Legs'])
    ReusedCount = int(request.json['ReusedCount'])
    Orbit = request.json['Orbit']
    LaunchSite = request.json['LaunchSite']

    model, scaler_model = get_model()
    input = prepare_input(PayloadMass, Flights, GridFins, Reused, Legs, ReusedCount, Orbit, LaunchSite, columns_model, scaler_model)
    prediction = predict_model(model, input)
    return jsonify({'Launch Rocket': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0') # To direct to my personal IP address and accept all