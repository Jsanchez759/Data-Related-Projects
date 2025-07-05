from flask import Flask, request, jsonify, render_template
import utils

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict_home_prices():
    total_sqft = float(request.form['total_sqft'])
    total_size = float(request.form['total_size'])
    baths = float(request.form['baths'])
    location = request.form['location']

    return jsonify({'estimated_price': utils.price_prediction(total_size, total_sqft, baths, location)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)