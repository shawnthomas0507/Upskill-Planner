import langgraph
from langchain_groq import ChatGroq
from langgraph.graph import START,END,StateGraph
from classes import MessagesState
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from read_rag import qa_chain
from rag import insert_rag
from tools import rag_retrieval,timetable_agent,calendar

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
        return "human_feedback"

def path_to_pdf(state:MessagesState):
    user_input=input("Enter the path to your resume:")
    return {"path": user_input, "messages": [HumanMessage(content=f"Resume path: {user_input}")]}

def human_feedback(state:MessagesState):
       user_input=input("Enter your query:")
       return {"messages":user_input,"user_query":user_input}

def save_rag(state:MessagesState):
    path=state["path"]
    insert_rag(path)
    return {"messages":"resume saved successfully"}


def agent(state:MessagesState):
        user_query=state["messages"][-1]
        ai_response2=llm.invoke([SystemMessage(content=f"You are an intelligent general assistant. You will be given a sentence. You have to tell whether the user query is an end of conversation sentence or not or if it is a resume related question or if it is a general question or it is a query where the user describes his/her day . If it is an end of conversation sentence just say True and if it is a resume related question just say resume and if it is a general question just say general and if it is a  query where the user describes his day say daily. The sentence is {user_query}")])
        if "resume" in ai_response2.content.lower():
             return {"messages":[AIMessage(content="rag_node")]}
        elif "daily" in ai_response2.content.lower():
            return {"messages":[AIMessage(content="timetable_node")]}
        elif "general" in ai_response2.content.lower():
             sys_message=SystemMessage(content="You are a helpful assistant tasked with answering the user's questions.If the user asks about creating a timetable or schedule ask him what does your todays schedule look like?")
             ai_response = llm.invoke([sys_message]+state["messages"])
             return {"messages": [AIMessage(content=ai_response.content)]}
        elif "true" in ai_response2.content.lower():
             sys_message=SystemMessage(content="You are a helpful assistant tasked with answering the user's questions.If the user asks about creating a timetable or schedule ask him what does your todays schedule look like?")
             ai_response = llm.invoke([sys_message]+state["messages"])
             return {"messages":[AIMessage(content=ai_response.content)],"messages": [AIMessage(content="END")]}

def should_continue(state:MessagesState):
      message = state["messages"][-1]
      if message.content == "END":
        return END
      elif message.content=="rag_node":
        return "rag_node"
      elif message.content=="timetable_node":
          return "timetable_node"
      else:
        return "human_feedback"




graph=StateGraph(MessagesState)
graph.add_node("agent",agent)
graph.add_node("save_rag",save_rag)
graph.add_node("path_to_pdf",path_to_pdf)
graph.add_node("human_feedback",human_feedback)
graph.add_node("rag_node",rag_retrieval)
graph.add_node("timetable_node",timetable_agent)
graph.add_node("calendar_node",calendar)
graph.add_conditional_edges(START,check_rag_for_resume)
graph.add_edge("path_to_pdf","save_rag")
graph.add_edge("save_rag","human_feedback")
graph.add_edge("human_feedback","agent")
graph.add_conditional_edges("agent",should_continue,[END, "human_feedback","rag_node","timetable_node"])
graph.add_edge("rag_node","human_feedback")
graph.add_edge("timetable_node","calendar_node")
graph.add_edge("calendar_node",END)
memory=MemorySaver()
app=graph.compile(checkpointer=memory)





