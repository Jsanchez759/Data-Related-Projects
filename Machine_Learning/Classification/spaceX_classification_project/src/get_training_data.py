import requests, warnings, sys, os
import pandas as pd
import numpy as np
sys.path.append('..')

def getBoosterVersion(data):
    BoosterVersion = []
    for x in data['rocket']:
        response = requests.get("https://api.spacexdata.com/v4/rockets/"+str(x)).json()
        BoosterVersion.append(response['name'])
    return BoosterVersion

def getLaunchSite(data):
    Longitude, Latitude, LaunchSite = [], [], []
    for x in data['launchpad']:
        response = requests.get("https://api.spacexdata.com/v4/launchpads/"+str(x)).json()
        Longitude.append(response['longitude'])
        Latitude.append(response['latitude'])
        LaunchSite.append(response['name'])
    return Longitude, Latitude, LaunchSite

def getPayloadData(data):
    PayloadMass, Orbit = [], []
    for load in data['payloads']:
        response = requests.get("https://api.spacexdata.com/v4/payloads/"+load).json()
        PayloadMass.append(response['mass_kg'])
        Orbit.append(response['orbit'])
    return PayloadMass, Orbit

def getCoreData(data):
    Block, ReusedCount, Serial, Outcome, Flights, GridFins, Reused, Legs, LandingPad = [], [], [], [], [], [], [], [], []
    for core in data['cores']:
        if core['core'] != None:
            response = requests.get("https://api.spacexdata.com/v4/cores/"+core['core']).json()
            Block.append(response['block'])
            ReusedCount.append(response['reuse_count'])
            Serial.append(response['serial'])
        else:
            Block.append(None)
            ReusedCount.append(None)
            Serial.append(None)
        Outcome.append(str(core['landing_success'])+' '+str(core['landing_type']))
        Flights.append(core['flight'])
        GridFins.append(core['gridfins'])
        Reused.append(core['reused'])
        Legs.append(core['legs'])
        LandingPad.append(core['landpad'])
    return Block, ReusedCount, Serial, Outcome, Flights, GridFins, Reused, Legs, LandingPad

def get_raw_data(url):
    response = requests.get(url)
    data = pd.json_normalize(response.json())
    data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]

    data = data[data['cores'].map(len) == 1]
    data = data[data['payloads'].map(len) == 1]

    data['cores'] = data['cores'].map(lambda x : x[0])
    data['payloads'] = data['payloads'].map(lambda x : x[0])

    data['date'] = pd.to_datetime(data['date_utc']).dt.date
    BoosterVersion = getBoosterVersion(data)
    Longitude, Latitude, LaunchSite = getLaunchSite(data)
    PayloadMass, Orbit = getPayloadData(data)
    Block, ReusedCount, Serial, Outcome, Flights, GridFins, Reused, Legs, LandingPad = getCoreData(data)
    launch_dict = {'FlightNumber': list(data['flight_number']),
               'Date': list(data['date']),
               'BoosterVersion':BoosterVersion,
               'PayloadMass':PayloadMass,
               'Orbit':Orbit,
               'LaunchSite':LaunchSite,
               'Outcome':Outcome,
               'Flights':Flights,
               'GridFins':GridFins,
               'Reused':Reused,
               'Legs':Legs,
               'LandingPad':LandingPad,
               'Block':Block,
               'ReusedCount':ReusedCount,
               'Serial':Serial,
               'Longitude': Longitude,
               'Latitude': Latitude}
    df = pd.DataFrame.from_dict(launch_dict)
    df = df[df['BoosterVersion'] != 'Falcon 1']
    df.loc[:,'FlightNumber'] = list(range(1, df.shape[0]+1))
    return df

def data_cleaning(df):
    payload_mean = df['PayloadMass'].mean()
    df['PayloadMass'] = df['PayloadMass'].replace(np.nan, payload_mean)
    bad_outcomes = {'False ASDS', 'False Ocean', 'False RTLS', 'None ASDS', 'None None'}
    df['Class'] = df['Outcome'].map(lambda x: 0 if x in bad_outcomes else 1)
    df = pd.get_dummies(df, columns=['Orbit', 'LaunchSite'], drop_first=False)
    df['GridFins'] = df['GridFins'].map(lambda x: 0 if x is False else 1)
    df['Reused'] = df['Reused'].map(lambda x: 0 if x is False else 1)
    df['Legs'] = df['Legs'].map(lambda x: 0 if x is False else 1)
    df['Serial'] = df['Serial'].map(lambda x: int(x[1:]))
    dummie_colums = list(df.columns[16:])
    for i in dummie_colums:
        df[i] = df[i].map(lambda x: 0 if x is False else 1)
    return df

def save_data(df):
    df.to_csv('data/clean_data.csv', index=False)


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/past"
    os.chdir(os.getcwd() + '\Machine_Learning\Classification\spaceX_classification_project')
    df = get_raw_data(url)
    df = data_cleaning(df)
    save_data(df)
    print('Data Cleaned and Saved')