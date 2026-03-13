import streamlit as st
import os

# Set page characteristics
st.set_page_config(
    page_title="Intelligent Document Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Relative imports from our modular structure
from utils.rag_pipeline import ask_rag, ingest_documents
from utils.web_search import perform_web_search
from utils.helper import stream_text
from config.config import Config

def main():
    st.title("🤖 Intelligent Document & Web Chatbot")
    st.caption("Ask questions about your local documents, securely augmented by live web searches.")
    
    # -----------------------------
    # Sidebar Configuration
    # -----------------------------
    with st.sidebar:
        st.header("⚙️ Settings")
        
        with st.expander("API Configuration", expanded=True):
            # Safe handling of API key input without hardcoding
            api_key_input = st.text_input(
                "Enter Gemini or OpenAI API Key:",
                type="password",
                placeholder="sk-... or AIza..."
            )
            
            if api_key_input:
                # Basic heuristic check to distribute API keys
                if api_key_input.startswith("sk-") or api_key_input.startswith("proj-"):
                    os.environ["OPENAI_API_KEY"] = api_key_input
                    Config.OPENAI_API_KEY = api_key_input
                    st.success("OpenAI Key Registered")
                else:
                    os.environ["GEMINI_API_KEY"] = api_key_input
                    Config.GEMINI_API_KEY = api_key_input
                    st.success("Gemini Key Registered")
                    
        # Feature 3: Response Modes
        st.subheader("Response Mode")
        mode = st.radio(
            "Select AI Detail Level:",
            options=["Concise", "Detailed"],
            help="Concise gives quick summaries; Detailed provides thorough explanations."
        )
        mode_str = mode.lower()
        
        st.divider()
        
        # Feature 1: Context Management
        st.subheader("📚 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Upload your PDFs here before ingesting them into the database."
        )
        
        if uploaded_files:
            # Ensure the data directory exists
            os.makedirs(Config.DOCUMENTS_DIR, exist_ok=True)
            
            # Save uploaded files to the expected directory
            for uploaded_file in uploaded_files:
                file_path = os.path.join(Config.DOCUMENTS_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.success(f"Ready to ingest {len(uploaded_files)} file(s)!")
        
        if st.button("Ingest Uploaded Documents", use_container_width=True):
            with st.spinner("Processing & embedding PDFs..."):
                try:
                    ingest_documents()
                    st.success("Documents successfully ingested into the Vector DB!")
                except Exception as e:
                    st.error(f"Ingestion failed: {str(e)}")
                    
    # -----------------------------
    # Main Chat Interface (Feature 4)
    # -----------------------------
    # Keep track of conversation history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Introduce the bot
        st.session_state.messages.append(
            {"role": "assistant", "content": "Hello! I am ready to answer questions based on your PDFs. If I don't know the answer, I will search the web for you. Please ensure your API key is provided in the sidebar."}
        )

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Handle user interaction
    if prompt := st.chat_input("Ask a question about your documents..."):
        
        # 1. Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. Check Prerequisites
        if not Config.GEMINI_API_KEY and not Config.OPENAI_API_KEY:
            st.warning("⚠️ Please provide an API key in the sidebar configuration to continue.")
            return
            
        # 3. Process LLM Action
        with st.chat_message("assistant"):
            try:
                # Attempt Document Search (RAG)
                with st.spinner("Searching documents..."):
                    response_text = ask_rag(question=prompt, mode=mode_str)
                    
                # Feature 2: Fallback to Web Search
                if not response_text: # The RAG chain returned None (meaning 'I don't know')
                    with st.spinner("Context not found in documents. Searching the live web..."):
                        response_text = perform_web_search(query=prompt, mode=mode_str)
                        
                # Present output to User incrementally (Streaming)
                # st.write_stream yields chunks, delivering a dynamic "typing" interface
                st.write_stream(stream_text(response_text))
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except FileNotFoundError:
                st.error("No knowledge base found! Please click 'Ingest Local Documents' in the sidebar first.")
            except Exception as e:
                # Graceful Error Handling
                st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
