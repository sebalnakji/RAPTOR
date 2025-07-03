from logging_config import extraction_hwp_logger
from fastapi import APIRouter, HTTPException
from .models import HWP_To_TexT_OutputData
from .utils import hwp_to_text


hwp_to_text_router = APIRouter()

description = """
hwp 파일의 text를 추출
단, [배포용 문서] hwp 파일은 추출 불가.
"""


@hwp_to_text_router.get("/hwp-to-text")
async def extract_hwp_data(catalog_path: str, origin_file_name: str):
    try:
        extraction_hwp_logger.info(f"HWP from {catalog_path}/{origin_file_name}")
        result = hwp_to_text(catalog_path, origin_file_name)
        extraction_hwp_logger.info("Processed HWP successfully")
        return result
        # return {
        #     "result": result
        # }
    except HTTPException as e:
        extraction_hwp_logger.error(f"HTTPException: {e.detail}")
        raise e
    except Exception as e:
        extraction_hwp_logger.error(f"Unhandled exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")
