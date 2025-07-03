from pydantic import BaseModel, Field, HttpUrl
from enum import Enum
from typing import Optional

class FileType(str, Enum):
    PPT = "ppt"
    DOCX = "docx"
    EXCEL = "excel"
    IMG = "img"
    HWP = "hwp"

class ConversionType(str, Enum):
    TO_PDF = "to_pdf"
    EXCEL_TO_CSV = "excel_to_csv"
    HWP_TO_XHTML = "hwp_to_xhtml"

class ConversionRequest(BaseModel):
    file_url: HttpUrl
    file_type: FileType
    conversion_type: ConversionType

class ConversionResponse(BaseModel):
    success: bool
    fileUrl: Optional[HttpUrl] = None

class ConversionStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class DetailConversionResponse(ConversionResponse):
    conversion_time: float = Field(..., description="변환에 걸린 시간(초)")
    file_size: int = Field(..., description="변환된 파일의 크기(바이트)")