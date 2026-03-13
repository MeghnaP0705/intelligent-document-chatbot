from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from config.config import Config
import logging

def get_embeddings():
    """
    Initializes and returns the appropriate embedding model based on user configuration.
    Falls back to a local HuggingFace embedding model if no API keys are provided or valid.
    """
    if Config.GEMINI_API_KEY:
        try:
            emb = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=Config.GEMINI_API_KEY
            )
            emb.embed_query("test")
            return emb
        except Exception as e:
            logging.warning(f"Failed to load Gemini Embeddings. Error: {e}")
            pass
            
    if Config.OPENAI_API_KEY:
        try:
            return OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=Config.OPENAI_API_KEY
            )
        except Exception as e:
            logging.warning(f"Failed to load OpenAI Embeddings. Error: {e}")
            pass
            
    # Default Fallback: Local embeddings using sentence-transformers
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
