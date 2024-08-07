import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from htmlTemplates import bot_template, user_template, css
import os, sys, re
sys.path.append('..')
import config

os.environ['HUGGINGFACEHUB_API_TOKEN'] = config.HUGGINGFACEHUB_API_TOKEN
model_id = "google/flan-t5-xxl"

def get_pdf_text(pdf_files):
    
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_chunk_text(text):
    
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )

    chunks = text_splitter.split_text(text)

    return chunks

def cleans_answer_model(message, question):
    
    word_to_find = "Question: {}".format(question)
    match = re.search(rf"{re.escape(word_to_find)}\s*(.*)", message)
    if match:
        text_following_word = match.group(1)
    
    return text_following_word


def get_vector_store(text_chunks):  
    
    # For OpenAI Embeddings
    
    # embeddings = OpenAIEmbeddings()
    
    # For Huggingface Embeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    vectorstore = Chroma.from_texts(text_chunks, embeddings)
    
    return vectorstore

def get_conversation_chain(vector_store):
    
    # OpenAI Model

    # llm = ChatOpenAI(model_name='gpt-4-1106-preview', temperature=0.3, openai_organization=os.getenv("ORG_ID"))

    # HuggingFace Model

    llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature":0.1})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )

    return conversation_chain

def handle_user_input(question):

    response = st.session_state.conversation({'question':question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write("User Response")
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(f"Model {model_id} response")
            #new_message = cleans_answer_model(message.content, question)
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Ask your PDF")

    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header('Chat with Your own PDFs :books:')
    question = st.text_input("Ask anything to your PDF: ")

    if question:
        handle_user_input(question)
    

    with st.sidebar:
        # st.subheader("Please set up your OpenAI key Here: ")
        # open_IA_key = st.text_input("Click on Enter after set up the key: ")
        # os.environ['OPENAI_API_KEY'] = open_IA_key
        st.subheader("Upload your Documents Here: ")
        pdf_files = st.file_uploader("Choose your PDF Files and Press OK", type=['pdf'], accept_multiple_files=True)

        if st.button("OK"):
            with st.spinner("Processing your PDFs..."):
                # Get PDF Text
                raw_text = get_pdf_text(pdf_files)

                # Get Text Chunks
                text_chunks = get_chunk_text(raw_text)
                # Create Vector Store
                vector_store = get_vector_store(text_chunks)
                st.write("DONE")
                # Create conversation chain
                st.session_state.conversation =  get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()