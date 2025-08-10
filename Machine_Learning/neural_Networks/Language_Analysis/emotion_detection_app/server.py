"""
Emotion Detection Server

This script, server.py, is the main entry point for our application. 
It sets up a server using the Flask framework, a lightweight and flexible 
micro web framework for Python.

The server is designed to perform emotion detection on user-provided text. 
"""

from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("emotionDetector", static_folder='static_code', static_url_path='/static_code')

@app.route("/")
def index():
    """
    This function initiates the rendering of the main application
    page over the Flask channel
    """
    return render_template('index.html')

@app.route("/emotionDetector")
def sent_emotion():
    """
    Analyze the user-provided text for emotions and return the result.
    """
    text_to_analyse = request.args.get("textToAnalyze")
    response = emotion_detector(text_to_analyse)
    if response['negative'] is not None:
        string_response = f"""
        for the given statement, the following emotions were detected:
        \n negative with {response['negative']*100:.2f}% of confidence, 
        \n neutral with {response['neutral']*100:.2f}%  of confidence
        \n and positive with {response['positive']*100:.2f}%  of confidence"""
    else:
        string_response = 'Invalid text! Please try again!.'
    return string_response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug = True)
