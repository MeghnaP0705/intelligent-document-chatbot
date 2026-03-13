from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate
from models.llm import get_llm

def perform_web_search(query: str, mode: str = "concise") -> str:
    """
    Falls back to a DuckDuckGo web search when the RAG pipeline lacks context.
    Retrieves web results and summarizes them using the LLM.
    """
    try:
        # Run DuckDuckGo text search
        # DuckDuckGoSearchResults returns a unified string containing summaries and links
        search_tool = DuckDuckGoSearchResults(num_results=3)
        raw_results = search_tool.run(query)
        
        if not raw_results or "No good DuckDuckGo Search Result" in raw_results:
            return "Live web search could not find any relevant information for your query."
            
        # Initialize LLM
        llm = get_llm(mode=mode)
        
        if mode.lower() == "concise":
            system_prompt = (
                "You are an assistant. Summarize the following web search results concisely "
                "to answer the user's query."
            )
        else:
            system_prompt = (
                "You are an assistant. Provide a highly detailed and thoroughly explained answer "
                "to the user's query based ONLY on the following web search results."
            )
            
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Query: {query}\n\nSearch Results:\n{results}")
        ])
        
        # Build an LCEL chain for the search summarization
        chain = prompt_template | llm
        response = chain.invoke({"query": query, "results": raw_results})
        
        # Let the user know the source of the information
        final_answer = f"*[Answer sourced via Live Web Search]*\n\n{response.content}"
        return final_answer
        
    except Exception as e:
        return f"An error occurred during the live web search fallback: {str(e)}"
