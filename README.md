# RAG Medical Chatbot

A web-based medical question-answering chatbot powered by **Retrieval-Augmented Generation (RAG)**. It answers user questions using context retrieved from medical PDF documents stored in a Pinecone vector database.

## How It Works

1. Medical PDFs are loaded from the `data/` folder, split into chunks, and embedded using HuggingFace (`all-MiniLM-L6-v2`).
2. Embeddings are stored in a Pinecone index (`medical-chatbot`).
3. When a user asks a question, the app retrieves the most relevant document chunks and sends them to an LLM (Llama 3.3 70B via Grok) to generate a concise answer.

## Tech Stack

- **Backend:** Flask, LangChain
- **Vector DB:** Pinecone
- **Embeddings:** HuggingFace Sentence Transformers
- **LLM:** Grok (Llama 3.3 70B)
- **Frontend:** HTML, Bootstrap, jQuery

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MOHAAAMEEEEED/RAG_Medical_Chatbot.git
   cd medical-chatbot
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. **Configure environment variables** — create a `.env` file in the project root:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   GROK_API_KEY=your_grok_api_key
   ```

5. **Build the vector index** (first time only):
   ```bash
   python strore_index.py
   ```

6. **Run the app**:
   ```bash
   python app.py
   ```
   Open `http://localhost:8080` in your browser.

## Usage

Type a medical question in the chat interface. The bot retrieves relevant context from indexed documents and returns a short, factual answer. If the answer is not in the documents, it will say it does not know.
