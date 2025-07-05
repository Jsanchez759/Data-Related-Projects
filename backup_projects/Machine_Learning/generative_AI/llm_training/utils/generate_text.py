import torch

class GenerateText():
    def __init__(self, model, tokenizer, max_length=50):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, prompt):
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to('cpu')

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=self.max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
