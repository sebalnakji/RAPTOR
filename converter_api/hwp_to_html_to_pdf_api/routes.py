from fastapi import APIRouter, HTTPException
from .utils import hwp_to_xhtml, xhtml_to_pdf
import requests
import os
from datetime import timedelta
from minio import Minio
from dotenv import load_dotenv
from urllib.parse import urlparse
import io

load_dotenv()

access_key = os.getenv('MINIO_ACCESS_KEY')
secret_key = os.getenv('MINIO_SECRET_KEY')

minio_client = Minio(
    "sbtglobal2.iptime.org:17355",
    access_key=access_key,
    secret_key=secret_key,
    secure=False
)

hwp_to_xhtml_router = APIRouter()
xhtml_to_pdf_router = APIRouter()

@hwp_to_xhtml_router.get("/hwp-to-xhtml")
async def convert_hwp_to_xhtml(fileUrl: str):   
     
    try:
        response = requests.get(fileUrl)
        response.raise_for_status()
        file_content = response.content

        parsed_url = urlparse(fileUrl)
        path_components = parsed_url.path.split('/')
        bucket_name = path_components[1]
        file_name = os.path.basename(parsed_url.path)
        object_name = f"temp/{file_name}"

        minio_client.put_object(bucket_name, object_name, io.BytesIO(file_content), length=len(file_content))

        converted_dir = hwp_to_xhtml(minio_client, bucket_name, object_name)

        return {"bucket_name": bucket_name, "converted_dir": converted_dir}
    
    except Exception as e:
        print(f"에러: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        minio_client.remove_object(bucket_name, object_name)


@xhtml_to_pdf_router.get("/xhtml-to-pdf")
async def convert_xhtml_to_pdf(bucket_name: str, converted_dir: str):
    expiry_time = timedelta(minutes=10)
    
    try:
        object_name = xhtml_to_pdf(minio_client, bucket_name, converted_dir)

        # presigned_url = minio_client.get_presigned_url(minio_client, bucket_name, object_name, expiry=expiry_time)
        presigned_url = minio_client.presigned_get_object(bucket_name, object_name, expires=expiry_time)
        
        return {"presigned_url": presigned_url}
    
    except Exception as e:
        print(f"에러: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
