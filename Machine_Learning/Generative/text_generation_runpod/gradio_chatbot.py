import gradio as gr
import requests

wizard_server, mistral_server = 'https://o2r0iu1g13robz-80.proxy.runpod.net','https://2beghhdjwoqzpm-80.proxy.runpod.net'
model_name_wizard, model_name_mistral = 'WizardLM-33B-V1.0-Uncensored', 'Mistral-7B-v0.1'

def generate_prompt(prompt: str, model_name: str) -> str:
    if model_name == model_name_wizard:
        prompt = f"""
        You are a helpful AI assistant.

        USER: {prompt}
        ASSISTANT:
        """.strip()
    else:
        prompt = prompt
    return prompt

def make_request(prompt:str, temperature: int, max_new_tokens: int, model_name: str): 
    data = {
        "inputs": prompt,
        "parameters": {"best_of": 1, "temperature":temperature, "max_new_tokens": max_new_tokens},
    }
    headers = {
        "Content-Type": "application/json",
    }
    if model_name == model_name_wizard:
        server = wizard_server
    else:
        server = mistral_server
    answer = requests.post(f"{server}/generate", json=data, headers=headers)
    return model_name + '\n' + answer.json()['generated_text'].strip()


def chatbot_response(message, history, model_name, temperature, max_new_tokens):
    if model_name == 1:
        model_name = model_name_wizard
    else:
        model_name = model_name_mistral
    prompt = generate_prompt(message, model_name)
    response = make_request(prompt, temperature, max_new_tokens, model_name)
    return response

with gr.Blocks() as demo:
    gr.Markdown(f" # Please select the model \n \
                - 1 for {model_name_wizard} \n \
                - 0 for {model_name_mistral}")
    model_name = gr.Slider(0, 1, value=0, step = 1)
    gr.Markdown("You can manipulate the temperature of the model")
    temperature = gr.Slider(0, 1, value=0.1, step = 0.1)
    gr.Markdown("You can manipulate the max tokens of the output of the model")
    max_new_tokens = gr.Slider(512, 4096, value=512, step = 10)
  
    gr.ChatInterface(chatbot_response, additional_inputs=[model_name, temperature, max_new_tokens])

demo.queue().launch(share=True)
