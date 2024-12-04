from typing import List
from pydantic import BaseModel


class Skills(BaseModel):
    technical_skills:List[str]
    projects:List[str]


