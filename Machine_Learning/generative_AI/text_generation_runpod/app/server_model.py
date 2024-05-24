from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)
first_call = True
chat_history_ids = []
step_chat = 0


def model_definition():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def get_answer(prompt, model, tokenizer):
    global first_call, chat_history_ids, step_chat

    prompt_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    if not first_call:
        prompt_ids = torch.cat([chat_history_ids[step_chat], prompt_ids], dim=-1)
        step_chat += 1

    new_chat_history_ids = model.generate(
        prompt_ids,
        max_length=1000,
        do_sample=True,
        top_p=0.95,
        top_k=0,
        temperature=0.75,
        pad_token_id=tokenizer.eos_token_id
    )

    chat_history_ids.append(new_chat_history_ids)

    output = tokenizer.decode(new_chat_history_ids[:, prompt_ids.shape[-1]:][0], skip_special_tokens=True)
    if first_call:
        first_call = False

    return output

@app.route('/', methods=['GET', 'POST'])
def get_response():
    
    model, tokenizer = model_definition()

    prompt = request.json['text']

    output = get_answer(prompt, model, tokenizer)

    return jsonify({'text': output})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0") # To direct to my personal IP address and accept all


