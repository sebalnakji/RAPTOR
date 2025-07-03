from pydantic import BaseModel
from typing import List

class RequestModel_llama3(BaseModel):
    texts: List[str]
    segCount: int = -1
    segMaxSize: int = 100
    segMinSize: int
    
class RequestModel_t5(BaseModel):
    texts: List[str]
    segMaxSize: int = 100
    segMinSize: int

class OutputData(BaseModel):
    status: dict
    result: dict