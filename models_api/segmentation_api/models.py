from pydantic import BaseModel
from typing import List

class InputData_llama3(BaseModel):
    texts: List[str]
    segCount: int = -1
    segMaxSize: int = 100
    segMinSize: int

# # 240708
# class InputData_nltk(BaseModel):
#     texts: List[str]
#     chunk_length: int = 100
#     whether_fix_sentence: bool

class InputData_nltk(BaseModel):
    texts: List[str]
    chunk_length: int = 128
    overlap_length: int = 28
    whether_fix_sentence: bool = False

class OutputData(BaseModel):
    status: dict
    result: dict