from pydantic import BaseModel
from typing import List

class InputData(BaseModel):
    texts: List[str]
    
class OutputData(BaseModel):
    status: dict
    result: dict