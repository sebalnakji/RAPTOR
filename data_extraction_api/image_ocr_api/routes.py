from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from .utils import image_to_text
from logging_config import extraction_image_logger


image_ocr_router = APIRouter()


@image_ocr_router.get("/image-ocr")
async def extract_image_ocr_data(catalog_path: str, origin_file_name: str):
    try:
        extraction_image_logger.info(f"Image from {catalog_path}/{origin_file_name}")
        result = image_to_text(catalog_path, origin_file_name)
        extraction_image_logger.info("Processed Image successfully")
        return JSONResponse(content=jsonable_encoder(result))
    except HTTPException as e:
        extraction_image_logger.error(f"HTTPException: {e.detail}")
        raise e
    except Exception as e:
        extraction_image_logger.error(f"Unhandled exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")
