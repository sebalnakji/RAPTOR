import os
import tempfile
from dotenv import load_dotenv
from matplotlib import pyplot as plt
# import cv2

# from typing import Union
# from fastapi import FastAPI
from paddleocr import PaddleOCR
from minio import Minio
from logging_config import extraction_image_logger

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


def image_to_text(catalog_path, origin_file_name):
    split_path = catalog_path.split('/', 1)
    bucket_name = split_path[0]
    dir_path = split_path[1] if len(split_path) > 1 else ""
    ext = os.path.splitext(origin_file_name)[1][1:]

    image_path = get_path(bucket_name, dir_path, origin_file_name)
    json_result = image_ocr(image_path)

    result = {
        "json": json_result,
        "type": "words",
        "finalFileType": ext,
    }

    extraction_image_logger.info("Image processing completed successfully")
    return result


def image_ocr(image_path):
    img = plt.imread(image_path)
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ocr = PaddleOCR(lang="korean", use_gpu=True)
    
    result = ocr.ocr(img, cls=False)

    size = img.shape
    word = []
    box = []
    page_word = []
    json = []

    for j in result[0]:
        word.append(j[-1][0])
        box.append(j[0][0]+j[0][2])

    for j in result[0]: #페이지 별 단어
        vertices = j[0][0]+j[0][2]
        words = {
            "text": j[-1][0],
            "boundingBox":{
                "vertices": [
                    {"x":vertices[0], "y":vertices[1]}, #x0, y0
                    {"x":vertices[2], "y":vertices[1]}, #x1, y0
                    {"x":vertices[2], "y":vertices[3]}, #x1, y1
                    {"x":vertices[0], "y":vertices[3]}, #x0, y1
                ]
            }
        }
        page_word.append(words)

    page_json = {
        "width": size[1],
        "height": size[0],
        "text": ' '.join(word),
        "words": page_word
    }
    json.append(page_json)

    return json

def get_path(bucket_name, dir_path, origin_file_name):
    ext = os.path.splitext(origin_file_name)[1]
    temp_dir = tempfile.mkdtemp()
    base_name = os.path.splitext(os.path.basename(f"{dir_path}/{origin_file_name}"))[0]
    temp_file_path = os.path.join(temp_dir, f"{base_name}{ext}")

    minio_client.fget_object(bucket_name, f"{dir_path}/{origin_file_name}", temp_file_path)

    return temp_file_path
