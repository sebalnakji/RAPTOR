from pydantic import BaseModel

class HWP_To_TexT_OutputData(BaseModel):
    status: dict
    result: dict