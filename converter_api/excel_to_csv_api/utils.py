import subprocess
from minio import Minio
import os
from datetime import datetime

def excel_to_csv(input_file_path):
    output_dir = "/tmp"
    result = subprocess.run(
        ['libreoffice', '--headless', '--convert-to', 'csv', '--outdir', output_dir, input_file_path, '--infilter=CSV:44,34,76,1,,0,true,true,true'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        raise Exception(f"파일 변환 중 오류 발생: {result.stderr.decode()}")
    csv_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file_path))[0] + ".csv")
    return csv_file_path

def upload_file_to_minio(client, file_path, bucket_name, object_name):
    with open(file_path, "rb") as file_data:
        client.put_object(bucket_name, object_name, file_data, os.stat(file_path).st_size)

def get_presigned_url(client, bucket_name, object_name, expiry=600):
    print(f"get_presigned_url called with expiry: {expiry} (type: {type(expiry)})")
    if not isinstance(expiry, int):
        raise ValueError("Expiry must be an integer representing seconds.")
    url = client.presigned_get_object(bucket_name, object_name, expires=expiry)
    print(f"Presigned URL: {url}")
    return url