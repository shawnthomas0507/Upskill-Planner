from typing import List,Annotated,TypedDict
from pydantic import BaseModel,Field
import pandas as pd
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage,BaseMessage
from langgraph.graph.message import add_messages
from datetime import datetime
from typing import Optional


class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
    skills_agent_output: str
    name: str
    path: str
    user_query: str
    
class Info(BaseModel):
    skills: str=Field(
        description="skills of the person"
    )
    projects: str=Field(
        description="projects done by the person"
    )
    @property
    def Information(self)->str:
        return f"Skills:{self.skills}\nProjects: {self.projects}"



class objects(BaseModel):
    time_slot: str=Field(
        description="Time slot of the task"
    )
    task: str=Field(
        description="Task to be done in the time slot"
    )

class Timetable(BaseModel):
    work: List[objects]=Field(
        description="comprehensive list of tasks and their time slots"
    )
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.dict())
    

class event(BaseModel):

    tool: str=Field(description="the tool to be learnt")
    start: datetime=Field(description="what does the time slot start")
    location: str=Field(description="location of the user")

    @property
    def repr(self) -> str:
        return (
            f"Event('{self.tool}', start={self.start}, location='{self.location}')"
        )




