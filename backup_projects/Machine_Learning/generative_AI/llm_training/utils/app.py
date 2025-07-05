from data_prepation import TextDataset
from generate_text import GenerateText
from train_model import TrainModel
from evaluate_model import EvaluateModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import random_split, DataLoader


if __name__ == "__main__":
    # Cargar el modelo pre-entrenado y su tokenizador
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    DATASET_PATH = "Generative_models_2.1/artifacts/data.txt"
    dataset = TextDataset(DATASET_PATH, tokenizer)

    if len(dataset) == 0:
        raise ValueError("El conjunto de datos está vacío después de la tokenización.")

    # Dividir los datos en validacion y entrenamiento
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Preparar el dataset de entrenamiento y validacion usando la clase DataLoader de Pytorch
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # Entrenar el modelo a partir de la clase creada
    train_model = TrainModel(train_loader, epochs= 10, model = model)
    trained_model = train_model.train()
    train_model.save_models()

    # Generar el texto
    generate_text = GenerateText(trained_model, tokenizer)
    prompt = "Machine Learning is"
    generated_text = generate_text(prompt)

    # Evaluar el modelo
    eval_model = EvaluateModel(trained_model, train_loader, val_loader)
    eval_model.evaluate()
    print(f"Generación de texto del modelo a '{prompt}': " + generated_text)
