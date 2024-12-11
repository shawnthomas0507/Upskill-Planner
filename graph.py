import langgraph
from langchain_groq import ChatGroq
from langgraph.graph import START,END,StateGraph
from classes import MessagesState
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from read_rag import qa_chain
from rag import insert_rag

llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model="llama-3.1-70b-versatile"
)

def rag_retrieval(user_query: str) -> str:
    """
    argument: user_query
    This tool is used to answer questions regarding the user's resume stored in the rag database.
    """
    response = qa_chain.run(user_query)
    return response

llm_with_tools=llm.bind_tools(tools=[rag_retrieval])


def check_rag_for_resume(state:MessagesState):
    name=state["name"]
    query = f"does {name} have a resume in here"
    response = qa_chain.run(query)
    if "No" in response:
        print("seems like I dont have your resume")
        return "path_to_pdf"
    elif "Yes" in response:
        print(f"Welcome {name} I have your resume in here")
        return "human_feedback"

def path_to_pdf(state:MessagesState):
    user_input=input("Enter the path to your resume:")
    return {"path": user_input, "messages": [HumanMessage(content=f"Resume path: {user_input}")]}

def human_feedback(state:MessagesState):
       user_input=input("Enter your query:")
       return {"messages":user_input}

def save_rag(state:MessagesState):
    path=state["path"]
    insert_rag(path)
    return {"messages":"resume saved successfully"}


def agent(state:MessagesState):
        sys_message=SystemMessage(content="You are a helpful assistant tasked with answering the user's questions.")
        ai_response = llm_with_tools.invoke([sys_message]+state["messages"])
        print(ai_response.content)
        ai_response2=llm.invoke([SystemMessage(content=f"You are an intelligent general assistant. You will be given a sentence. You have to tell whether that sentence is an end of conversation sentence or not. If it is an end of conversation sentence just say True else just say False. The sentence is {ai_response.content}")])
        if "true" in ai_response2.content.lower():
             return {"messages":[AIMessage(content=ai_response.content)],"messages": [AIMessage(content="END")]}

        return {"messages": [AIMessage(content=ai_response.content)]}

def should_continue(state:MessagesState):
      message = state["messages"][-1]
      if message.content == "END":
        return END
      else:
        return "human_feedback"

graph=StateGraph(MessagesState)
graph.add_node("agent",agent)
graph.add_node("save_rag",save_rag)
graph.add_node("path_to_pdf",path_to_pdf)
graph.add_node("human_feedback",human_feedback)
graph.add_node("tools",ToolNode([rag_retrieval]))
graph.add_conditional_edges(START,check_rag_for_resume)
graph.add_edge("path_to_pdf","save_rag")
graph.add_edge("save_rag","human_feedback")
graph.add_edge("human_feedback","agent")
graph.add_conditional_edges("agent",should_continue,[END, "human_feedback"])
graph.add_conditional_edges("agent",tools_condition)
graph.add_edge("tools","agent")
memory=MemorySaver()
app=graph.compile(checkpointer=memory)





