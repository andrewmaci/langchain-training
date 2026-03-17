from dotenv import load_dotenv

from typing import TypedDict,Annotated
from langgraph.graph import StateGraph,END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,HumanMessage

from chains import generate_chain, reflection_chain

load_dotenv()

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph) -> MessageGraph:
    return {"messages": generate_chain.invoke(state["messages"])}

def reflection_node(state: MessageGraph) -> MessageGraph:
    response = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content)]}

def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

def create_graph():
    builder =  StateGraph(state_schema=MessageGraph)
    builder.add_node(GENERATE,generation_node)
    builder.add_node(REFLECT,reflection_node)
    builder.set_entry_point(GENERATE)
    builder.add_conditional_edges(GENERATE,should_continue,path_map={END:END,REFLECT:REFLECT})
    builder.add_edge(REFLECT,GENERATE)

    return builder.compile()


def main():
    graph = create_graph()
    input_messages = HumanMessage(content="Make this tweet better: The sheer willpower it takes to reply \"sounds good!\" to an email that absolutely does not sound good. 🫠 Send help (or a dangerously large coffee). ☕️✉️ #CorporateLife #MondayMotivation")
    response = graph.invoke({"messages": [input_messages]})

if __name__ == "__main__":
    main()
