from PyPDF2 import PdfReader
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from classes import Info,Timetable
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
import streamlit as st

llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model="llama-3.1-70b-versatile"
)


@tool
def extract_skills(text):
    """
    argument: text
    This tool is used to extract the skills from a resume.Call this tool when the user asks for the anything related to the resume.
    """
    
    instructions="""
    You are an intelligent assistant that is able to extract skills and projects from a resume.
    You will be given an extracted resume text and you need to extract the skills and projects of the person from the extracted resume text.
    You need to return the skills and projects in a structured manner. Also describe the projects in detail.
    The resume is as follows: {text}
    """
    sys_message=instructions.format(text=text)
    response=llm.with_structured_output(Info).invoke([SystemMessage(content=sys_message)])
    return response.Information



def timetable_agent(text):
    """
    argument:text
    This tool is used to create a daily timetable for the user based on the missing skills, tools and projects. This tool is only called when the user asks anything related to a timetable.
    """
    instructions="""
    You are an assistant that reads in the user's text containing his daily schedule. You need to extract the time slots and the tasks to be done in the time slots from the text.
    The user's text is as follows: {text}
    """
    sys_message=instructions.format(text=text)
    response=llm.with_structured_output(Timetable).invoke([SystemMessage(content=sys_message)])
    return response.to_dataframe()
