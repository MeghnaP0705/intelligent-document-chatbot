# Intelligent Document & Web Chatbot

A production-ready AI chatbot built using Python, Streamlit, LangChain, and FAISS. It natively supports **Retrieval-Augmented Generation (RAG)** on local documents (PDFs) and dynamically falls back to **Live Web Search** when it lacks specific context. Users can easily toggle between **Concise** and **Detailed** response modes directly from the UI.

---

## 🚀 Features

1. **Retrieval-Augmented Generation (RAG)**: Chat directly with your local PDFs. The app chunks, embeds, and stores documents in a local FAISS database for lightning-fast semantic retrieval.
2. **Live Web Search Fallback**: If the provided documents lack the answer, the bot intelligently searches the live internet (via DuckDuckGo) and summarizes those findings.
3. **Adaptive Response Modes**:
   - `Concise`: Quick summaries and direct answers.
   - `Detailed`: Extensive explanations scaling out the sourced information.
4. **Streamlit Chat UI**:
   - Modern, ChatGPT-style streaming chat interface.
   - Elegant sidebar handling setup, API keys, and configurations dynamically.
5. **Secure & Zero Hardcoding**: Securely reads API credentials via Sidebar UI variables or system environment configurations (`.env`). Supports both Google Gemini and OpenAI GPT models.

---

## 📁 Project Structure

```text
project/
│
├── config/
│   └── config.py          # Environment settings & overarching constants
├── models/
│   ├── llm.py             # Instantiates LLMs (Gemini/OpenAI) using Langchain
│   └── embeddings.py      # Instantiates Embeddings (Gemini/OpenAI)
├── utils/
│   ├── rag_pipeline.py    # Document loading, text splitting, Vector DB logic (FAISS)
│   ├── web_search.py      # DuckDuckGo integration and results evaluation
│   └── helper.py          # UI streaming helpers
├── data/                  # **PLACE YOUR PDFs HERE**
├── app.py                 # The core Streamlit UI and chat management script
├── requirements.txt       # All necessary Python dependencies
└── README.md              # Project documentation (You are here)
```

---

## 🛠️ Step-by-Step Local Deployment

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your machine.

### 2. Install Dependencies
Navigate into the `project` directory and install the required modules.
*(Optional but recommended: do this inside a new virtual environment.)*

```bash
# Create and activate virtual environment
python -m venv venv
# Linux/MacOS
source venv/bin/activate
# Windows
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Setup API Keys
You can either input your API keys directly into the **Streamlit Web UI** Sidebar when you run the app, OR you can export them as system environment variables by creating a `.env` file in the root `project/` directory:
```env
GEMINI_API_KEY=your-gemini-key-here
OPENAI_API_KEY=your-openai-key-here
```

### 4. Provide Documents
Place any `.pdf` documents you wish to chat with inside the `data/` folder.

### 5. Run the Application
Start the Streamlit development server locally.
```bash
streamlit run app.py
```
*Your browser will automatically open to `http://localhost:8501`. In the UI sidebar, click **"Ingest Local Documents"** to generate your database, then start chatting!*

---

## ☁️ Deploying to Streamlit Community Cloud (Public URL)

You can deploy this project publicly and for free using **Streamlit Cloud**:

1. **Upload your code to a GitHub Repository**
   - Ensure you do NOT commit your API keys. Add any `.env` files to `.gitignore`.
2. **Login to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io) and link your GitHub account.
3. **Deploy the App**
   - Click **"New App"**.
   - Select your newly created repository and branch.
   - For **Main file path**, enter: `app.py`
4. **Configure Secrets**
   - Before deploying, click **"Advanced settings"**.
   - Under the **Secrets** section, configure your API keys securely:
     ```toml
     GEMINI_API_KEY = "your-actual-api-key"
     OPENAI_API_KEY = "your-actual-api-key"
     ```
   - Click save and **Deploy**.

*Congratulations! Your Intelligent Chatbot is now live on the internet!*
