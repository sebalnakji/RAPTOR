import os
import subprocess
from weasyprint import HTML
from minio import Minio
import tempfile
import shutil
import re

def hwp_to_xhtml(minio_client, bucket_name, object_name):
    temp_dir = tempfile.mkdtemp()
    base_name = os.path.splitext(os.path.basename(object_name))[0]
    
    temp_file_path = os.path.join(temp_dir, base_name + ".hwp")
    minio_client.fget_object(bucket_name, object_name, temp_file_path)

    result = subprocess.run(
        ["hwp5html", "--output", temp_dir, temp_file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise Exception(f"파일 변환 중 오류 발생: {result.stderr.decode()}")

    # Upload all generated files to MinIO
    for root, dirs, files in os.walk(temp_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, temp_dir)
            object_name = f"converted-files/hwp-to-html/{base_name}/{relative_path}"
            minio_client.fput_object(bucket_name, object_name, file_path)

    shutil.rmtree(temp_dir)

    return f"converted-files/hwp-to-html/{base_name}"

def xhtml_to_pdf(minio_client, bucket_name, converted_dir):
    temp_dir = tempfile.mkdtemp()
    base_name = os.path.basename(converted_dir)
    
    objects = minio_client.list_objects(bucket_name, prefix=converted_dir, recursive=True)
    for obj in objects:
        object_name = obj.object_name
        file_name = object_name.replace(converted_dir + '/', '')
        file_path = os.path.join(temp_dir, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        minio_client.fget_object(bucket_name, object_name, file_path)

    xhtml_file = next(f for f in os.listdir(temp_dir) if f.endswith('.xhtml') or f.endswith('.html'))
    css_file = next(f for f in os.listdir(temp_dir) if f.endswith('.css'))

    xhtml_path = os.path.join(temp_dir, xhtml_file)
    css_path = os.path.join(temp_dir, css_file)
    pdf_path = os.path.join(temp_dir, f"{base_name}.pdf")

    HTML(xhtml_path).write_pdf(pdf_path, stylesheets=[css_path], margin_left=5)

    pdf_object_name = f"converted-files/hwp-to-html-to-pdf/{base_name}.pdf"
    minio_client.fput_object(bucket_name, pdf_object_name, pdf_path)

    shutil.rmtree(temp_dir)

    return pdf_object_name

def get_presigned_url(client, bucket_name, object_name, expiry):
    if not isinstance(expiry, int):
        raise ValueError("Expiry must be an integer representing seconds.")
    url = client.presigned_get_object(bucket_name, object_name, expires=expiry)
    return url