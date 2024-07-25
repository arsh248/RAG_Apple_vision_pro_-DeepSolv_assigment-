# RAG Chatbot - Apple Vision Pro

## Overview

This Streamlit application leverages advanced language processing techniques to create a Retrieval-Augmented Generation (RAG) chatbot. The chatbot is designed to assist users with inquiries related to Apple Vision Pro using a combination of GPT-3.5 turbo and Google Generative AI embeddings. The application also incorporates a FAISS vector database to efficiently handle and retrieve relevant information from a PDF knowledge base.

## Frameworks and Tools

- **Framework**: LangChain
- **Data Sources**: PDF documents and websites (web scraping).
- **Language Model**: GPT-3.5 turbo
- **Embedding**: Google Generative AI embeddings
- **Vector Database**: FAISS

## Setup

### Prerequisites

1. Python 3.8 or higher
2. Streamlit
3. LangChain
4. langchain_core
5. langchain_community
6. langchain_google_genai
7. `python-dotenv`

### Installation

1. Clone this repository:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a `.env` file in the project directory with the following content:

    ```dotenv
    OPENAI_API_KEY=<your-openai-api-key>
    GOOGLE_API_KEY=<your-google-api-key>
    ```

## Usage

1. Ensure you have your `.env` file with the necessary API keys.
2. Place your PDF documents in the `./knowledge_base` directory.
3. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

4. Open the provided URL in your web browser to access the chatbot.

## Code Explanation

- **Initialization**:
  - The `OpenAI` and `GoogleGenerativeAIEmbeddings` instances are initialized with the API keys from the environment variables.
  - The Streamlit app is titled "RAG Chatbot - Apple Vision Pro".

- **Vector Embedding**:
  - The `vector_embedding` function handles the process of loading PDF documents, creating text chunks, generating embeddings, and storing them in a FAISS vector database.

- **Interaction Handling**:
  - User inputs are processed to generate responses based on the context provided by the PDF documents.
  - The chat history is maintained and displayed in the Streamlit interface.

