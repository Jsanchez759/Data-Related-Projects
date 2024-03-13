import gradio as gr
import config, os, runpod, time
os.environ['RUNPOD_API_KEY'] = config.RUNPOD_API_KEY

runpod.api_key = os.getenv("RUNPOD_API_KEY")
endpoint = runpod.Endpoint("lrlpopd6tvfmfc")

system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, \
while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal \
content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, \
or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, \
please don't share false information."

cost_server = 0.00044

def generate_prompt(prompt, default_prompt = system_prompt):
    return f"""
    [INST] <<SYS>>
    {default_prompt}
    <</SYS>>
{prompt}[/INST]

    """.strip()

def make_request(prompt:str, temperature: int, max_new_tokens: int): 

    run_request = endpoint.run(
        {"prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature}
    )
    init_request = time.time()
    i = 0
    status = run_request.status()
    while status != 'COMPLETED':
        status = run_request.status()
        i += 1
        if i > 50:
            break
    end_request = time.time()
    time_total = round(end_request-init_request,2)
    money = '${:,.8f}'.format(round(cost_server*time_total,7))
    cost_request = f'This request/answer cost around {money}: '
    return f'{cost_request} {run_request.output()}'


def chatbot_response(message, history, temperature, max_new_tokens, new_system_prompt):
    prompt = generate_prompt(message, new_system_prompt)
    response = make_request(prompt, temperature, max_new_tokens)
    return response

with gr.Blocks() as demo:
    gr.Markdown(f" # Interact with Llama2-13B")
    new_system_prompt = gr.Textbox(value=system_prompt, label="Default System Prompt: You can manipulate it")
    gr.Markdown("You can manipulate the temperature of the model")
    temperature = gr.Slider(0, 1, value=0.1, step = 0.1)
    gr.Markdown("You can manipulate the max tokens of the output of the model")
    max_new_tokens = gr.Slider(512, 4096, value=512, step = 10)
    gr.ChatInterface(chatbot_response, additional_inputs=[temperature, max_new_tokens, new_system_prompt])

demo.queue().launch(share=True)
