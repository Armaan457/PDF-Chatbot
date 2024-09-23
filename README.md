# PDF Chatbot
A **Retrieval Augmented Generation (RAG)** powered chatbot that enables users to interact with custom PDF documents using an LLM. Utilises **Groq LPU** for fast inference and features a message history for easier reference during interactions. Presently trained on a pdf on Mangonels but but easily adaptable for any PDF using the included `gen_emb.ipynb` notebook.

# Technologies used
**Web Interface**: Streamlit <br>
**Embeddings**: Llama 2 (via Ollama)<br>
**LLM**: Llama3-8b-8192 (via Groq) <br>
**Vector Database**: Chroma <br>

# Setup

Clone the repository:

```sh
> git clone https://github.com/Armaan457/PDF-Chatbot.git
```
Create and activate a virtual environment:

```sh
> python -m venv venv
> venv\Scripts\activate
  ```
Install dependencies:

```sh
> pip install -r requirements.txt
```

Run the development server:

```sh
> streamlit run app.py
```
