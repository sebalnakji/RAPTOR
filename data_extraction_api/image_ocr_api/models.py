from pydantic import BaseModel
from typing import Dict

class ImageOcrProcessResponse(BaseModel):
    status: Dict[str, str]
    result: Dict[str, str]
