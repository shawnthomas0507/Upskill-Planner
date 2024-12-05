import langgraph
from langchain_groq import ChatGroq
from langgraph.graph import START,END,MessagesState,StateGraph
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from tools import extract_skills

llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model="llama-3.1-70b-versatile"
)


tools = [extract_skills]
llm_with_tools=llm.bind_tools(tools=tools)

def agent(state:MessagesState):
    return {"messages":[llm_with_tools.invoke(state["messages"]+[SystemMessage(content="You are an intelligent general assistant that is able to answer any question. Just remember if the user asks whether he can upload a resume just tell him he can upload the resume in the sidebar ")])]}


graph=StateGraph(MessagesState)
graph.add_node("agent",agent)
graph.add_node("tools",ToolNode(tools=tools))
graph.add_edge(START,"agent")
graph.add_conditional_edges("agent",tools_condition)
graph.add_edge("tools","agent")
graph.add_edge("agent",END)

app=graph.compile()


