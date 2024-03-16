import requests

def call_app():
    url = "http://127.0.0.1:5000/predict"
    data = {"PayloadMass": 3000, "Flights": 4, "GridFins": 1, "Reused": 1, "Legs": 1, "ReusedCount": 0, 
            "Orbit": 'Orbit_VLEO', "LaunchSite": 'LaunchSite_CCSFS SLC 40'}
    response = requests.post(url, json=data)
    print(response.json()['Launch Rocket'])


if __name__ == "__main__":
    call_app()      