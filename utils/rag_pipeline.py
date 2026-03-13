import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
from langchain_core.prompts import ChatPromptTemplate
from models.llm import get_llm
from models.embeddings import get_embeddings
from config.config import Config

def ingest_documents() -> bool:
    """
    Reads PDFs from the configured directory, chunks the text, creates embeddings,
    and stores them in a local FAISS index.
    """
    # Ensure data directory exists
    os.makedirs(Config.DOCUMENTS_DIR, exist_ok=True)
    
    documents = []
    # Process all PDFs in the data folder
    for filename in os.listdir(Config.DOCUMENTS_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(Config.DOCUMENTS_DIR, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
    if not documents:
        raise ValueError(f"No PDF documents found in '{Config.DOCUMENTS_DIR}'.")
        
    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Generate and store embeddings
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    # Save the index locally for future use
    vectorstore.save_local(Config.FAISS_INDEX_PATH)
    
    return True

def get_rag_chain(mode: str = "concise"):
    """
    Loads the FAISS index and returns a configured LangChain retrieval chain.
    The response generation behavior changes based on 'mode'.
    """
    if not os.path.exists(Config.FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{Config.FAISS_INDEX_PATH}'. "
            "Please ingest documents first."
        )
        
    embeddings = get_embeddings()
    # Allow dangerous deserialization is required for local FAISS loading in recent versions
    vectorstore = FAISS.load_local(
        Config.FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Configure retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Get appropriately configured LLM
    llm = get_llm(mode=mode)
    
    # Set up conditional prompts
    if mode.lower() == "concise":
        system_prompt = (
            "You are a concise operational assistant. Answer the user's question directly and briefly "
            "using ONLY the provided context.\n"
            "If the answer is not contained within the context, you MUST say exactly: 'I don't know' "
            "so that a web search fallback can be triggered.\n\n"
            "Context: {context}"
        )
    else:
        system_prompt = (
            "You are an analytical and detailed AI assistant. Provide a comprehensive explanation "
            "based ONLY on the provided context.\n"
            "If the answer is not contained within the context, you MUST say exactly: 'I don't know' "
            "so that a web search fallback can be triggered.\n\n"
            "Context: {context}"
        )
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Assemble the LCEL chain
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def ask_rag(question: str, mode: str = "concise") -> str:
    """
    Processes a user's question through the RAG pipeline.
    Returns the answer if found in context, or None if the LLM doesn't know.
    """
    try:
        chain = get_rag_chain(mode=mode)
        answer = chain.invoke(question).strip()
        
        # Determine if we need fallback
        # A crude but effective check based on the strict system prompt instructions
        if "I don't know" in answer or not answer:
            return None
            
        return answer
    except FileNotFoundError as e:
        # Re-raise FileNotFoundError so the UI can prompt the user to ingest docs
        raise e
    except Exception as e:
        print(f"Error within RAG pipeline: {e}")
        return None
