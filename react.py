import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

load_dotenv()

@tool
def quadruple(num: float) -> float:
    """
    param num: a number to quadruple
    returns: the quadruple of the input
    """
    return num * 4


tools = [TavilySearch(num_results=3), quadruple]

llm = ChatOpenAI(
    model=os.environ["OPENROUTER_GENERATIVE_MODEL"],
    base_url=os.environ["OPENROUTER_GENERATIVE_BASE_URL"],
    api_key=os.environ["OPENROUTER_API_KEY"]
)

llm_with_tools = llm.bind_tools(tools)

