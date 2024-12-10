import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from access_data import data_retriever
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.graph import END

from langchain_core.messages import SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver

# Initialize API keys and models
OPENAI_KEY = os.getenv('OPENAI_KEY')
tavily_tools = TavilySearchResults(max_results=2, tavily_api_key="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")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_KEY)

# Create the state graph
graph_builder = StateGraph(MessagesState)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    results, serialized = data_retriever(query)
    return serialized, results

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve, tavily_tools])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ToolNode for executing retrieval
tools = ToolNode([retrieve, tavily_tools])

def generate(state: MessagesState):
    """Generate answer based on retrieved content."""
    recent_tool_messages = [msg for msg in reversed(state["messages"]) if msg.type == "tool"]
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n" + docs_content
    )
    
    conversation_messages = [
        msg for msg in state["messages"]
        if msg.type in ("human", "system") or (msg.type == "ai" and not msg.tool_calls)
    ]
    
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}

# Build the graph
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

# Set up memory for conversation state
memory = MemorySaver()
config = {"configurable": {"thread_id": "abc123"}}
graph = graph_builder.compile(checkpointer=memory)

# Main interaction loop
def main():
    print("Chatbot is ready! Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        try:
            response = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
            response_content =response["messages"][-1].content if response["messages"][-1].content !='' else response["messages"][3].content
            print(f"Chatbot: {response_content}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()



# # A chat model:
# import os
# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
# # This model for state is so versatile that LangGraph offers a built-in version for convenience:
# from langgraph.graph import MessagesState, StateGraph
# # Let's turn our retrieval step into a tool:
# from langchain_core.tools import tool
# from access_data import data_retriever
# # We build them below. Note that we leverage another pre-built LangGraph component, ToolNode, that executes the tool and adds the result as a ToolMessage to the state.
# from langchain_core.messages import SystemMessage
# from langgraph.prebuilt import ToolNode
# from langgraph.graph import END
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_community.tools.tavily_search import TavilySearchResults

# tavily_tools = TavilySearchResults(max_results=2,tavily_api_key="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")

# llm = ChatOpenAI(model="gpt-4o-mini",api_key=os.getenv('OPENAI_KEY'))
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.getenv('OPENAI_KEY'))
# graph_builder = StateGraph(MessagesState)


# @tool(response_format="content_and_artifact")
# def retrieve(query: str):
#     """Retrieve information related to a query."""
#     results, serialized=data_retriever(query)
#     return  serialized,results

# # Step 1: Generate an AIMessage that may include a tool-call to be sent.
# def query_or_respond(state: MessagesState):
#     """Generate tool call for retrieval or respond."""
#     llm_with_tools = llm.bind_tools([retrieve,tavily_tools])
#     response = llm_with_tools.invoke(state["messages"])
#     # MessagesState appends messages to state instead of overwriting
#     return {"messages": [response]}

# # Step 2: Execute the retrieval.
# tools = ToolNode([retrieve,tavily_tools])

# # Step 3: Generate a response using the retrieved content.
# def generate(state: MessagesState):
#     """Generate answer."""
#     # Get generated ToolMessages
#     # recent_tool_messages = [msg for msg in reversed(state["messages"]) if msg.type == "tool"]
#     # tool_messages = recent_tool_messages[::-1]

#     recent_tool_messages = []
#     for message in reversed(state["messages"]):
#         if message.type == "tool":
#             recent_tool_messages.append(message)
#         else:
#             break
#     tool_messages = recent_tool_messages[::-1]

#     # Format into prompt
#     docs_content = "\n\n".join(doc.content for doc in tool_messages)
#     system_message_content = (
#         "You are an assistant for question-answering tasks. "
#         "Use the following pieces of retrieved context to answer "
#         "the question. If you don't know the answer, say that you "
#         "don't know. Use three sentences maximum and keep the "
#         "answer concise."
#         "\n\n"
#         f"{docs_content}"
#     )
#     conversation_messages = [
#         message 
#         for message in state["messages"]
#         if message.type in ("human", "system")
#         or (message.type == "ai" and not message.tool_calls)
#     ]
#     prompt = [SystemMessage(system_message_content)] + conversation_messages

#     # Run
#     response = llm.invoke(prompt)
#     return {"messages": [response]}



# graph_builder.add_node(query_or_respond)
# graph_builder.add_node(tools)
# graph_builder.add_node(generate)

# graph_builder.set_entry_point("query_or_respond")
# graph_builder.add_conditional_edges(
#     "query_or_respond",
#     tools_condition,
#     {END: END, "tools": "tools"},
# )
# graph_builder.add_edge("tools", "generate")
# graph_builder.add_edge("generate", END)

# from langgraph.checkpoint.memory import MemorySaver 
# memory = MemorySaver()
# config={"configurable": {"thread_id": "abc123"}}
# graph = graph_builder.compile(checkpointer=memory)

# # graph = graph_builder.compile()

# print("Done")
# if __name__=="__main__":
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Chatbot: Goodbye!")
#             break
#         response= graph.invoke( {"messages": [{"role": "user", "content": user_input}]},config=config,)#,config={"configurable": {"thread_id": "abc123"}})
#         response=response["messages"][-1].content if response["messages"][-1].content !='' else response["messages"][3].content
#         print(f"Chatbot: {response}")
        
        






