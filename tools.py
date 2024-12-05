from PyPDF2 import PdfReader
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from classes import Info
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
    This tool is used to extract the skills from a resume.Call this tool when the user asks for the skills.
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


