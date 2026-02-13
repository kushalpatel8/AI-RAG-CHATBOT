# AI RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) Chatbot built using **Streamlit** and **LangChain**. It allows users to ask questions related to a specific PDF document, processing the text to provide accurate answers using the **Groq API** and **HuggingFace embeddings**.

## 🚀 Features

* **RAG Architecture**: Retrieves context from a local PDF document to answer user queries accurately.
* **LLM Integration**: Uses the **Groq API** (specifically the `openai/gpt-oss-120b` model) for generating responses.
* **Vector Search**: Utilizes **ChromaDB** and **HuggingFace Embeddings** (`all-MiniLM-L12-v2`) for efficient similarity search.
* **Interactive UI**: A clean, conversational interface built with **Streamlit** that maintains chat history.

## 🛠️ Tech Stack

* **Python 3.14**
* **Streamlit**: For the web interface.
* **LangChain**: For building the RAG pipeline (Loaders, Splitters, Chains).
* **Groq**: For high-speed LLM inference.
* **ChromaDB**: As the vector store.
* **HuggingFace**: For text embeddings.

## 📋 Prerequisites

Before running the application, ensure you have the following:

1.  **Python 3.14** installed.
2.  A **Groq API Key**. You can obtain one from the [Groq Cloud Console](https://console.groq.com/).
3.  A PDF document named `human_eye.pdf` (or update the code to point to your desired file).

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd AI-RAG-CHATBOT
    ```

2.  **Install Dependencies:**
    This project uses `Pipenv` for dependency management.
    ```bash
    pipenv install
    ```
    *Alternatively, if you prefer `pip`:*
    ```bash
    pip install streamlit langchain-groq langchain-community langchain-classic chromadb sentence-transformers pypdf python-dotenv
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your Groq API key:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

4.  **Add Your Data:**
    Ensure the PDF file you want to query is placed in the root directory. By default, the code looks for:
    ```text
    ./human_eye.pdf
    ```
    *(Note: You can change the `pdf_name` variable in `main.py` to load a different file.)*

## ▶️ Usage

To start the chatbot application, run the following command:

```bash
streamlit run main.py
