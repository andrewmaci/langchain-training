import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

def get_config() -> Dict[str, str]:
    """Retrieve and validate configuration from environment variables."""
    config = {
        "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "ministral-3:14b"),
    }
    
    missing = [k for k, v in config.items() if not v and k != "ollama_host"]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    return config

def initialize_agent(config: Dict[str, str]):
    """Initialize the LangChain agent with tools and LLM."""
    try:
        tools = [TavilySearch()]
        llm = ChatOllama(
            base_url=config["ollama_host"],
            temperature=0.1,
            model=config["ollama_model"]
        )
        return create_agent(llm, tools)
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise

def run_query(agent: Any, query: str) -> Dict[str, Any]:
    """Execute a query through the agent and return the result."""
    logger.info(f"Executing query: {query}")
    try:
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        return result
    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        return {"error": str(e)}

def main():
    try:
        config = get_config()
        agent = initialize_agent(config)
        
        query = "What is currently happening in Poland?"
        result = run_query(agent, query)
        
        print("\n--- Agent Result ---")
        print(result)
        print("--------------------\n")
        
    except Exception as e:
        logger.critical(f"Application failed: {e}")

if __name__ == "__main__":
    main()
