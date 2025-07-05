import math
import torch

class EvaluateModel:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def compute_perplexity(self, model, data_loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to('cpu')
                outputs = model(batch, labels=batch)
                loss = outputs.loss
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        perplexity = math.exp(avg_loss)
        return perplexity

    def evaluate(self):
        train_perplexity = self.compute_perplexity(self.model, self.train_loader)
        val_perplexity = self.compute_perplexity(self.model, self.val_loader)

        print(f"Perplejidad de Entrenamiento: {train_perplexity}")
        print(f"Perplejidad de Validación: {val_perplexity}")
