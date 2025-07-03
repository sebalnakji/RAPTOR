from fastapi import APIRouter, HTTPException, Query
from .utils import excel_to_csv, upload_file_to_minio, get_presigned_url
import requests
import os
from datetime import datetime
from datetime import timedelta
from minio import Minio
from dotenv import load_dotenv
from urllib.parse import urlparse

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

excel_to_csv_router = APIRouter()

@excel_to_csv_router.get("/excel-to-csv")
async def convert_excel_to_csv(fileUrl: str):
    temp_file_path = f"/tmp/{datetime.now().strftime('excel-to-csv_%Y_%m_%d_%H%M')}.xlsx"
    csv_file_path = None
    expiry_time = timedelta(minutes=10)
    try:
        response = requests.get(fileUrl)
        response.raise_for_status()
        with open(temp_file_path, "wb") as f:
            f.write(response.content)
        csv_file_path = excel_to_csv(temp_file_path)
        object_name = "converted-files/excel-to-csv/" + os.path.basename(csv_file_path)

        parsed_url = urlparse(fileUrl)
        path_components = parsed_url.path.split('/')
        bucket_name = path_components[1]

        upload_file_to_minio(minio_client, csv_file_path, bucket_name, object_name)
        presigned_url = minio_client.presigned_get_object(bucket_name, object_name, expires=expiry_time)
        print(f"Presigned URL: {presigned_url}")
        return {"presigned_url": presigned_url}
    except Exception as e:
        print(f"에러: {str(e)}")  
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_file_path)
        if csv_file_path and os.path.exists(csv_file_path):
            os.remove(csv_file_path)
