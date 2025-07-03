import os
import tempfile
from dotenv import load_dotenv
import re
from minio import Minio
from logging_config import extraction_hwp_logger
from llama_index.readers.file import HWPReader
from pathlib import Path

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


def hwp_to_text(catalog_path, origin_file_name):
    split_path = catalog_path.split('/', 1)
    bucket_name = split_path[0]
    dir_path = split_path[1] if len(split_path) > 1 else ""

    hwp_path = get_path(bucket_name, dir_path, origin_file_name)

    file = Path(hwp_path)
    reader = HWPReader()
    reader.load_data(file=file)

    result = {
        "json": preprocess_text(reader.get_text()),
        "type": "lines",
        "finalFileType": "hwp",
    }

    extraction_hwp_logger.info("Image processing completed successfully")

    return result


def preprocess_text(text):
    json = []
    page_text = ""
    page_word = []

    # 바이너리 데이터와 불필요한 제어 문자를 제거하는 정규 표현식
    # \x00-\x1F는 ASCII 제어 문자 범위, \x7F-\x9F는 제어 문자로 사용된 확장 ASCII 범위

    lines = text.splitlines()
    for i in lines:
        cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', i)  # 바이너리 데이터 제거

        # 한자를 제거하는 정규식 패턴
        pattern = re.compile(
            r'[\u3400-\u4DBF'  # 확장 A
            r'\u4E00-\u9FFF'  # 기본 한자
            r'\uF900-\uFAFF]'  # 호환 한자
            #r'\u20000-\u2A6DF'  # 확장 B
            #r'\u2A700-\u2B73F'  # 확장 C
            #r'\u2B740-\u2B81F'  # 확장 D
            #r'\u2B820-\u2CEAF'  # 확장 E
            #r'\u2F800-\u2FA1F]'  # 호환 한자 보충
        )

        # 텍스트에서 한자 제거
        cleaned_text = re.sub(pattern, '', cleaned_text)

        # 특정 문자 'ȃ'를 제거하는 정규식 패턴
        remove_ȃ_pattern = re.compile(r'ȃ')

        # 텍스트에서 ȃ 제거
        cleaned_text = re.sub(remove_ȃ_pattern, '', cleaned_text)

        # 여러 공백을 하나로
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

        # 앞뒤 공백 제거
        cleaned_text = cleaned_text.strip()

        if cleaned_text:
            words = {
                "text": cleaned_text,
            }
            page_word.append(words)
            
            if page_text:
                page_text += '\n'
            page_text += cleaned_text

    page_json = {
        "text": page_text,
        "words": page_word,
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
