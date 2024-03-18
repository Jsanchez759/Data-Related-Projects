import gradio as gr
import requests

server = "http://127.0.0.1:5000/"
model_name = "microsoft/DialoGPT-medium"

css = """
.container {
    height: 90vh;
}
"""

def make_request(prompt:str): 

    data = {"text": prompt}
    answer = requests.post(server, json=data)

    return answer.json()['text'].strip()


def chatbot_response(message, history):
    response = make_request(message)
    return response

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_classes=["container"]):
        gr.Markdown(f" # Interact with {model_name}")
        gr.ChatInterface(chatbot_response, fill_height=True)

demo.queue().launch()
