# 🛡️ VeriChat: Private AI Document Assistant

**VeriChat** is a local RAG (Retrieval-Augmented Generation) application that allows users to chat with sensitive documents (PDFs) without sending data to external cloud servers.

## 🚀 Features
* **100% Offline:** Runs locally using **Ollama** and **Llama 3**.
* **Privacy-First:** No data leaves the user's machine.
* **RAG Architecture:** Uses **LangChain** and **ChromaDB** for efficient document indexing and retrieval.
* **Interactive UI:** Built with **Streamlit**.

## 🛠️ Tech Stack
* **Python 3.10+**
* **LangChain** (Orchestration)
* **Ollama** (LLM Backend)
* **ChromaDB** (Vector Store)
* **Streamlit** (Frontend)

## 📦 How to Run
1.  Clone the repo:
    ```bash
    git clone [https://github.com/MalindaBotheju/VeriChat-Local-RAG.git](https://github.com/MalindaBotheju/VeriChat-Local-RAG.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the app:
    ```bash
    streamlit run app.py
    ```
