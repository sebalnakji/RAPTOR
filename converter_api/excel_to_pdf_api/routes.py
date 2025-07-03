from fastapi import APIRouter, HTTPException
from .utils import excel_to_pdf_main
from logging_config import converter_excel_to_pdf_logger


excel_to_pdf_router = APIRouter()


@excel_to_pdf_router.get("/excel-to-pdf")
async def convert_excel_to_pdf(catalog_path: str, origin_file_name: str):
    try:
        converter_excel_to_pdf_logger.info(f"Excel from {catalog_path}/{origin_file_name}")
        result = excel_to_pdf_main(catalog_path, origin_file_name)

        return result

    except Exception as e:
        converter_excel_to_pdf_logger.error(f"Excel to PDF 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
