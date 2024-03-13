import gradio as gr
import requests

server = 'https://o2r0iu1g13robz-80.proxy.runpod.net'
model_name_wizard = 'WizardLM-33B-V1.0-Uncensored'

default_prompt = "You are a helpful AI assistant."

def generate_prompt(prompt):
    return  f"""
You are a helpful AI assistant.

USER: {prompt}
ASSISTANT:

""".strip()

def make_request(prompt:str, temperature: int, max_new_tokens: int): 
    data = {
        "inputs": prompt,
        "parameters": {"best_of": 1, "temperature":temperature, "max_new_tokens": max_new_tokens},
    }
    headers = {
        "Content-Type": "application/json",
    }
    answer = requests.post(f"{server}/generate", json=data, headers=headers)
    return answer.json()['generated_text'].strip()


def chatbot_response(message, history, temperature, max_new_tokens, system_prompt):
    prompt = generate_prompt(message)
    response = make_request(prompt, temperature, max_new_tokens)
    return response

with gr.Blocks() as demo:
    gr.Markdown(f" # Interact with {model_name_wizard}")
    system_prompt = gr.Textbox(value= default_prompt, label="Default System Prompt: You can manipulate the system prompt")
    gr.Markdown("You can manipulate the temperature of the model")
    temperature = gr.Slider(0, 1, value=0.1, step = 0.1)
    gr.Markdown("You can manipulate the max tokens of the output of the model")
    max_new_tokens = gr.Slider(512, 4096, value=512, step = 10)
  
    gr.ChatInterface(chatbot_response, additional_inputs=[temperature, max_new_tokens, system_prompt])

demo.queue().launch(share=True)
