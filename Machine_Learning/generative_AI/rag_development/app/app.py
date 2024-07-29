from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModel,
)
import warnings
import time
from dotenv import load_dotenv
from datasets import load_dataset
import torch
from pinecone import Pinecone
from pinecone import ServerlessSpec
import os, re
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def create_pineco_index(api_key):
    pc = Pinecone(api_key=api_key)
    spec = ServerlessSpec(cloud="aws", region="us-east-1")

    index_name = "chatbot-rag"
    if index_name not in [index_info["name"] for index_info in pc.list_indexes()]:
        pc.create_index(name=index_name, dimension=384, metric="cosine", spec=spec)

    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

    return pc.Index(index_name)


def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def index_data(data, batch_size, index):
    for i in range(0, len(data), batch_size):
        i_end = min(len(data), i + batch_size)
        batch = data.iloc[i:i_end]
        ids = [f"{x['id']}" for i, x in batch.iterrows()]
        embeds = [
            embed_text("Question: " + x["question"] + ", Answer: " + x["answer"])
            for _, x in batch.iterrows()
        ]
        metadata = [
            {"question": x["question"], "answer": x["answer"]} for i, x in batch.iterrows()
        ]
        index.upsert(vectors=zip(ids, embeds, metadata))
    return index


def get_responses(question, index, k=3):
    # Model sin RAG respuesta
    input_ids = tokenizer_flat(question, return_tensors="pt").input_ids
    outputs = model_flat.generate(input_ids)
    response_without_rag = tokenizer_flat.decode(outputs[0])

    # RAG respuesta
    query_embedding = embed_text(question)
    results = index.query(vector=query_embedding.tolist(), top_k=k, include_metadata=True)
    retrieved_docs = [
        "Question: " + match["metadata"]["question"] + ", Answer: " + match["metadata"]["answer"] 
        for match in results["matches"]
    ]
    database_vector_response = " ".join(retrieved_docs)
    input_text = f"Please use this context {database_vector_response} to answer the following question {question}"
    input_ids = tokenizer_flat(input_text, return_tensors="pt").input_ids
    outputs = model_flat.generate(input_ids)
    response_with_rag = tokenizer_flat.decode(outputs[0], skip_special_tokens=True)
    return database_vector_response, response_without_rag, response_with_rag


if __name__ == "__main__":
    load_dotenv()
    question = input("Ingresa la pregunta: ")
    tokenizer_flat = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
    model_flat = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    api_key = os.getenv("PINECONE_KEY")
    index = create_pineco_index(api_key)
    dataset = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages", split='test')
    data = dataset.to_pandas().reset_index()
    batch_size = 100
    index = index_data(data, batch_size, index)
    question = f"Please answer to the following question. {question}"
    database_vector_response, response_without_rag, response_with_rag = get_responses(
        question, index
    )

    print("")
    print("Respuesta sin RAG:", response_without_rag)
    print("")
    print("Respuesta con RAG:", response_with_rag)
    print("")
    # Obtener solo la primera respuesta de las 3 que trae la base de datos
    match = re.search(r"Answer: (.*?)(?:Question:|$)", database_vector_response, re.DOTALL).group(1).strip()
    print("Respuesta de la base de datos vectorial:", match)
