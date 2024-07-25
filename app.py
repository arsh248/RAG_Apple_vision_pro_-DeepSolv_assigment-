import streamlit as st
import os
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

# Load API keys from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Title of the Streamlit app
st.title("RAG Chatbot- Apple Vision Pro")

# Initialize the language model
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert sales agent. Use the provided context to answer the questions and persuade the user to purchase the product.
    
    User Persona:
    - Name: Sir
    - Profession: Software Developer
    - Looking for: Cutting-edge solutions to improve productivity

    <context>
    {context}
    <context>
    Questions: {input}

    Remember to highlight the unique benefits and features of the product, address any concerns, and encourage the user to make a purchase. 
    Use the knowledge to answer the questions and persuade the user to purchase the product.
    Also dont give big response. use short and sweet response.
    Give answers in short. Dont repeat the answers.
    """
)

# Function to create vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./knowledge_base")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

# Automatically run embedding process in the background
vector_embedding()

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Text input for user question
prompt1 = st.text_input("Enter Your Question From Documents")

# If there's a question, process it and display the conversation history
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    
    # Append the question and response to the conversation history
    st.session_state.history.append({
        "question": prompt1,
        "answer": response['answer'],
        "context": response["context"]
    })

# Display the conversation history
if st.session_state.history:
    for interaction in st.session_state.history:
        st.write(f"**Question:** {interaction['question']}")
        st.write(f"**Answer:** {interaction['answer']}")
