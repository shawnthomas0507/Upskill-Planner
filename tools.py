from PyPDF2 import PdfReader
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from classes import Info,Timetable
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
import streamlit as st
from classes import MessagesState,event
from read_rag import qa_chain
from gcsa.google_calendar import GoogleCalendar
from datetime import datetime
from gcsa.event import Event

llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model="llama-3.1-70b-versatile"
)

def get_events(state:MessagesState):
    calendar = GoogleCalendar('shawnthomas0507@gmail.com')
    events=[]
    for event in calendar:
        events.append(event)
    instructions="""
    You are an intelligent assistant. You will be given a list of events of a google calendar.You have to tell the user the events scheduled in his calendar in a concise way.
    The events list is {events}
    """
    sys_message=instructions.format(events=events)
    response=llm.invoke([SystemMessage(content=sys_message)])
    return {"messages":[AIMessage(content=response.content)]}





def timetable_agent(state:MessagesState):   
    user_query=state["messages"][-2] 
    instructions="""
    You are an assistant that reads in the user's text containing his daily schedule. You need to extract the time slots and the tasks to be done in the time slots from the text. You then need to print out the time slots and tasks in a structured manner that looks like a table.
    The user's text is as follows: {text}.
    Remember i dont want any python code. I just want you to extract the tasks and time slots and print them nicely.
    """
    sys_message=instructions.format(text=user_query)
    response=llm.invoke([SystemMessage(content=sys_message)])

    instructions="""
    You are an intelligent assistant that is able to create a daily timetable for the user based on the missing skills, tools and projects.
    The user's daily timetable is as follows: {user_timetable}
    Your first task is to select one tool from the user's missing skills which he can learn in his free time.
    You are to return the time slots and time slots of the user's daily schedule along with the new tool the user needs to learn in a structured manner that looks like a table.
    Remember, you are to only give only one day's schedule not for the whole week. And you should be more specific on what exactly you can learn about that tool in the specific time slot.
    Also once you give the schedule inform the user that you will be adding it to the calendar.
    """
    sys_message=instructions.format(user_timetable=response.content)
    response1=llm.invoke(state["messages"]+[SystemMessage(content=sys_message)])
    return {"messages":[AIMessage(content=response1.content)]}


def rag_retrieval(state:MessagesState):
    user_query=state["user_query"]
    response = qa_chain.run(user_query)
    instructions="""
    You are an intelligent assistant that is able to answer questions about the resume.
    You will be given a context and the user query. Using the context ask the user's query.
    The context is as follows: {context}
    The user_query is as follows: {user_query}
    """
    sys_message=instructions.format(user_query=user_query,context=response)
    response=llm.invoke([SystemMessage(content=sys_message)])
    return {"messages":[AIMessage(content=response.content)]}



def calendar(state:MessagesState):
    tool=state["messages"][-2].content
    instructions="""
    You are an intelligent assistant. You will be given a sentence. 
    You should extract the tool to be learnt from it and the time the tool will be learnt. Also the location to learn this tool is philadelphia.
    Also you will need todays date and time. todays date and time: {today} 
    text: {text}
    Using the tool and time you need to create a event function.

    I want the output to be in this format:
    event=Event(
    'The Glass Menagerie',
    start=datetime(2020, 7, 10, 19, 0),
    location='Záhřebská 468/21',
    minutes_before_popup_reminder=15
    )

    This format should be strictly followed. Dont output any code or any sort of text just this Event function. Be very mindful of the indentation it should be perfect.
    """
    sys_message=instructions.format(text=tool,today=datetime.now())
    response=llm.invoke([SystemMessage(content=sys_message)])
    

    str1="""
    from gcsa.event import Event
    from gcsa.google_calendar import GoogleCalendar
    from datetime import datetime
    from gcsa.event import Event

    calendar = GoogleCalendar('shawnthomas0507@gmail.com')
    """

    str2=response.content
    if not str2.startswith("event="):
         str2 = "event=" + str2

    str3="\ncalendar.add_event(event)"

    str1+=str2
    str1+=str3
    event_str = "\n".join([line.strip() for line in str1.splitlines()])
    exec(event_str)
    return {"messages":[AIMessage(content=" I have successfully created an event for you in google calendar to study this skill. Good Luck !")]}






