import langgraph
from langchain_groq import ChatGroq
from langgraph.graph import START,END,MessagesState,StateGraph
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from tools import extract_skills

llm = ChatGroq(
    
)


tools = [extract_skills]
llm_with_tools=llm.bind_tools(tools=tools)

def agent(state:MessagesState):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}


graph=StateGraph(MessagesState)
graph.add_node("agent",agent)
graph.add_node("tools",ToolNode(tools=tools))
graph.add_edge(START,"agent")
graph.add_conditional_edges("agent",tools_condition)
graph.add_edge("tools","agent")
graph.add_edge("agent",END)

app=graph.compile()


