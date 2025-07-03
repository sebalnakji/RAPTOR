from fastapi import APIRouter, HTTPException
from .utils import image_to_pdf_main
import os
from minio import Minio
from dotenv import load_dotenv
from logging_config import converter_image_logger


load_dotenv()

endpoint = os.getenv('MINIO_ENDPOINT')
access_key = os.getenv('MINIO_ACCESS_KEY')
secret_key = os.getenv('MINIO_SECRET_KEY')

minio_client = Minio(
    endpoint,
    access_key=access_key,
    secret_key=secret_key,
    secure=False
)

image_to_pdf_router = APIRouter()


@image_to_pdf_router.get("/image-to-pdf")
async def convert_image_to_pdf(catalog_path: str, origin_file_name: str):
    try:
        converter_image_logger.info(f"Image from {catalog_path}/{origin_file_name}")
        presigned_url = image_to_pdf_main(minio_client, catalog_path, origin_file_name)

        return presigned_url
    except Exception as e:
        print(f"에러: {str(e)}")  
        converter_image_logger.error(f"Image to PDF 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
