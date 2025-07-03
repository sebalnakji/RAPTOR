from fastapi import APIRouter, HTTPException
from .utils import docx_to_pdf_main
from logging_config import converter_docx_logger


docx_to_pdf_router = APIRouter()


@docx_to_pdf_router.get("/docx-to-pdf")
async def convert_docx_to_pdf(catalog_path: str, origin_file_name: str):
    try:
        converter_docx_logger.info(f"DOCX from {catalog_path}/{origin_file_name}")
        result = docx_to_pdf_main(catalog_path, origin_file_name)

        return result

    except Exception as e:
        converter_docx_logger.error(f"DOCX to PDF 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
