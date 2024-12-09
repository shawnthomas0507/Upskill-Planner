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


def check_rag_for_resume(state:MessagesState):
    name=state["name"]
    query = f"does {name} have a resume in here"
    response = qa_chain.run(query)
    if "No" in response:
        print("seems like I dont have your resume")
        return "path_to_pdf"
    elif "Yes" in response:
        print(f"Welcome {name} I have your resume in here")
        return "agent"

def path_to_pdf(state:MessagesState):
    user_input=input("Enter the path to your resume:")
    return {"path": user_input, "messages": [HumanMessage(content=f"Resume path: {user_input}")]}

def save_rag(state:MessagesState):
    path=state["path"]
    insert_rag(path)
    print("resume saved successfully")
    return {"messages":"resume saved successfully"}


def agent(state:MessagesState):
    return {"messages":[llm.invoke(state["messages"]+[SystemMessage(content="You are an intelligent general assistant that is able to answer any question.If the user asks about the resume first ask him the path to his resume. If the user asks to build a timetable first ask him his daily schedule.")])]}


graph=StateGraph(MessagesState)
graph.add_node("agent",agent)
graph.add_node("read_rag",save_rag)
graph.add_node("path_to_pdf",path_to_pdf)
graph.add_conditional_edges(START,check_rag_for_resume)
graph.add_edge("path_to_pdf","read_rag")
graph.add_edge("read_rag","agent")
graph.add_edge("agent",END)
app=graph.compile()


