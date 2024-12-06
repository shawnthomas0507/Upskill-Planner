from typing import List
from pydantic import BaseModel,Field
import pandas as pd

class Info(BaseModel):
    skills: str=Field(
        description="skills of the person"
    )
    projects: str=Field(
        description="projects done by the person"
    )
    role: str=Field(
        description="Role of the analyst in the context of the topic."
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
        return pd.DataFrame(self.dict()["work"])
