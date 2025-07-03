import os
import subprocess
import tempfile
import datetime
from minio import Minio
from dotenv import load_dotenv

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


def excel_to_pdf_main(catalog_path, origin_file_name):
    split_path = catalog_path.split('/', 1)
    bucket_name = split_path[0]
    dir_path = split_path[1] if len(split_path) > 1 else ""
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_file_name = f"{os.path.splitext(origin_file_name)[0]}_{current_time}"
    pdf_object_name = f"{dir_path}/{unique_file_name}.pdf"

    pdf_file_path = excel_to_pdf(bucket_name, dir_path, origin_file_name)
    minio_client.fput_object(bucket_name, pdf_object_name, pdf_file_path)

    return {
        "catalog_path": catalog_path,
        "converted_file_name": f"{unique_file_name}.pdf",
    }


def excel_to_pdf(bucket_name, dir_path, origin_file_name):
    ext = os.path.splitext(origin_file_name)[1]
    temp_dir = tempfile.mkdtemp()
    base_name = os.path.splitext(os.path.basename(f"{dir_path}/{origin_file_name}"))[0]
    temp_file_path = os.path.join(temp_dir, f"{base_name}.{ext}")
    minio_client.fget_object(bucket_name, f"{dir_path}/{origin_file_name}", temp_file_path)

    result = subprocess.run(
        ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', temp_dir, temp_file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise Exception(f"파일 변환 중 오류 발생: {result.stderr.decode()}")

    pdf_file_path = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(temp_file_path))[0]}.pdf")

    return pdf_file_path
