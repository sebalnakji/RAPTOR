from fastapi import APIRouter, HTTPException
from .utils import ppt_to_pdf_main
from logging_config import converter_ppt_logger


ppt_to_pdf_router = APIRouter()


@ppt_to_pdf_router.get("/ppt-to-pdf")
async def convert_ppt_to_pdf(catalog_path: str, origin_file_name: str):
    try:
        converter_ppt_logger.info(f"PPT from {catalog_path}/{origin_file_name}")
        result = ppt_to_pdf_main(catalog_path, origin_file_name)

        return result

    except Exception as e:
        converter_ppt_logger.error(f"PPT to PDF 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
