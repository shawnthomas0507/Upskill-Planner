from PyPDF2 import PdfReader
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from classes import Info,Timetable
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
import streamlit as st
from classes import MessagesState
from read_rag import qa_chain

llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model="llama-3.1-70b-versatile"
)

@tool
def resume_helper(text):
    
    """
    Argument: text
    This tool is used to extract the skills from a extracted resume text.
    """
    instructions="""
    You are an intelligent assistant that is able to extract skills and projects from a resume.
    You will be given an extracted resume text and you need to extract the skills and projects of the person from the extracted resume text.
    You need to return the skills and projects in a structured manner. Also describe the projects in detail.
    The resume is as follows: {text}
    """
    sys_message=instructions.format(text=text)
    response=llm.with_structured_output(Info).invoke([SystemMessage(content=sys_message)])
    return response.Information,{"skills_agent_output":response.Information}


@tool
def timetable_agent(text):
    """
    argument:text
    This tool is used to create a daily timetable for the user based on the missing skills, tools and projects.
    """


    instructions="""
    You are an assistant that reads in the user's text containing his daily schedule. You need to extract the time slots and the tasks to be done in the time slots from the text.
    The user's text is as follows: {text}
    """
    user_skills=MessagesState["skills_agent_output"]
    sys_message=instructions.format(text=text)
    response=llm.with_structured_output(Timetable).invoke([SystemMessage(content=sys_message)])
    user_timetable=response.to_dataframe()

    instructions="""
    You are an intelligent assistant that is able to create a daily timetable for the user based on the missing skills, tools and projects.
    The user's daily timetable is as follows: {user_timetable}
    The user's missing skills are as follows: {user_skills}.
    Your first task is to select one tool from the user's missing skills which he can learn in his free time.
    You are to return the time slots and time slots of the user's daily schedule along with the new tool the user needs to learn.
    """
    sys_message=instructions.format(user_timetable=user_timetable,user_skills=user_skills)
    response1=llm.with_structured_output(Timetable).invoke([SystemMessage(content=sys_message)])
    return response1.to_dataframe()


def rag_retrieval(state:MessagesState):
    user_query=state["user_query"][-1]
    response = qa_chain.run(user_query)
    instructions="""
    You are an intelligent assistant that is able to answer questions about the resume.
    You will be given a context and the user query. Using the context ask the user's query.
    The context is as follows: {context}
    The user_query is as follows: {user_query}
    """
    sys_message=instructions.format(user_query=user_query,context=response)
    response=llm.with_structured_output(Info).invoke([SystemMessage(content=sys_message)])
    return {"messages":response.content}






