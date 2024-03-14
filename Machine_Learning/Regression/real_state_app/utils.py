import pickle, json
import numpy as np

model = pickle.load(open('model.pickle', 'rb'))
columns_model = json.load(open('columns_model.json'))

def price_prediction(size, sqft, baths, location):
    loc_index = np.where(list(columns_model['columns_models'])==location)[0] # Here, we check if the location exist in the columns of the resulting data set
    
    x = np.zeros(len(list(columns_model['columns_models']))) # We create the array to insert in the model
    x[0] = size
    x[1] = sqft
    x[2] = baths
    if loc_index >= 0: # If the location exist, we put 1 in their corresponding column
        x[loc_index] = 1
    
    return model.predict([x])[0]