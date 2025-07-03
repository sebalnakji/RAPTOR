from pydantic import BaseModel
from typing import Dict

class PDFProcessResponse(BaseModel):
    status: Dict[str, str]
    result: Dict[str, str]
