from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

def set_up_model():
    llm = OllamaLLM(model="llama3.2:3b")
    return llm

def set_up_embeddings():
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    return embeddings

def chunking(processed_transcript, chunk_size=200, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_text(processed_transcript)
    return chunks
