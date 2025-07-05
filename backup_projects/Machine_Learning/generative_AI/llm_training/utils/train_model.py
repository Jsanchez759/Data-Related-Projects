from transformers import AdamW, get_linear_schedule_with_warmup
import json
import torch

class TrainModel():
    def __init__(self, train_loader, epochs, model) -> None:
        self.train_loader = train_loader
        self.epochs = epochs
        self.model = model
        self.total_steps = len(self.train_loader) * self.epochs
        self.results = {}

    def train(self):
        print("Iniciando el entrenamiento")
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=self.total_steps
        )
        model = self.model.train()
        for epoch in range(self.epochs):
            for step, batch in enumerate(self.train_loader):
                batch = batch.to('cpu')
                outputs = self.model(batch, labels=batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                self.results[f"training_loss_{epoch}"] = loss.item()

        return model

    def save_models(self):
        model_name = "gpt2_based_model"
        torch.save(
            self.model.state_dict(),
            f"Generative_models_2.1/artifacts/models/{model_name}.pth",
        )
        with open(
            f"Generative_models_2.1/artifacts/models/{model_name}_results.json", "w"
        ) as f:
            json.dump(self.results, f)

    def __len__(self):
        """
        Return the total number of steps for the training process.
        """
        return self.total_steps
