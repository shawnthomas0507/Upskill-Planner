from typing import List
from pydantic import BaseModel,Field


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



