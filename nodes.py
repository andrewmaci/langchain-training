from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm_with_tools, tools

load_dotenv()


SYSTEM_PROMPT = """
You are a helpful assistant that can use tools to answer questions:
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning loop.
    """
    response = llm_with_tools.invoke([{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]])

    return {"messages": [response]}

tool_node = ToolNode(tools)
