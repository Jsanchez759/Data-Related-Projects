from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder

filepath = 'model.h5'
model_load = tf.keras.models.load_model(
    filepath, custom_objects=None, compile=True
)

def predict(model, images):
    img_predict = tf.keras.utils.load_img(images, target_size=(256, 256)) # To convert the images into RGB scale tensor
    img_array = tf.keras.utils.img_to_array(img_predict)
    img_array = tf.expand_dims(img_array, 0) # To create a batch

    predictions = model.predict(img_array)
    confidence = tf.nn.softmax(predictions[0])
    return class_names[np.argmax(confidence)], 100 * np.max(confidence)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        prediction, confidence = predict(model_load, img)
        print(prediction, confidence)
        return render_template('index.html', img=img, prediction=prediction, confidence=round(confidence,2))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8001)