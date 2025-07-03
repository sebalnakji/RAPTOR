import os
import tempfile
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from minio import Minio
import pypdfium2 as pdfium
from logging_config import extraction_pdf_logger

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


def pdf_to_text(catalog_path, origin_file_name):
    split_path = catalog_path.split('/', 1)
    bucket_name = split_path[0]
    dir_path = split_path[1] if len(split_path) > 1 else ""

    pdf_bytes = get_pdf_bytes(bucket_name, dir_path, origin_file_name)
    text_result = pdf2txt(pdf_bytes)

    result = {
        "json": [],
        "type": "",
        "finalFileType": "pdf",
    }

    if text_result["txtpercent"] < 0.2:
        ocr_result = pdf2ocr(pdf_bytes)
        result["json"] = extract_json(ocr_result["txt"], text_result["size"], ocr_result["word"], ocr_result["box"])
        result["type"] = "words"
    else:
        result["json"] = extract_json(text_result["txt"], text_result["size"], text_result["lines"], text_result["box"])
        result["type"] = "lines"

    extraction_pdf_logger.info("PDF processing completed successfully")
    return result


def get_pdf_bytes(bucket_name, dir_path, origin_file_name):
    temp_dir = tempfile.mkdtemp()
    base_name = os.path.splitext(os.path.basename(f"{dir_path}/{origin_file_name}"))[0]
    temp_file_path = os.path.join(temp_dir, f"{base_name}.pdf")

    minio_client.fget_object(bucket_name, f"{dir_path}/{origin_file_name}", temp_file_path)

    with open(temp_file_path, 'rb') as file:
        file_bytes = file.read()

    return file_bytes


def pdf2txt(path): #텍스트 유무 확인 포함
    txt = []
    box = []
    lines = []
    size = []
    pdf = pdfium.PdfDocument(path)
    count = 0 #텍스트 페이지 카운팅
    for i in pdf:
        page = i.get_textpage()
        size.append(i.get_size()) #각 페이지 사이즈
        if page.get_text_bounded() != '':
            count += 1
            txt.append(page.get_text_bounded()) #txt
            page_lines = page.get_text_bounded().splitlines()
            lines.append(page_lines) #개행 단위
            
            page_box =[]
            after_last_box_end = 0
            for idx, j in enumerate(page_lines):
                try:
                    curr_idx = page.search(j, match_whole_word=True, match_case=False, consecutive=False).get_next() #개행 좌표 찾기
                    page_height = i.get_mediabox()[3]

                    charboxes = [page.get_charbox(k) for k in range(curr_idx[0], curr_idx[0] + curr_idx[1])]
                    min_y = min(min(box[1] for box in charboxes), min(box[3] for box in charboxes))
                    max_y = max(max(box[1] for box in charboxes), max(box[3] for box in charboxes))
                    min_x = min(min(box[0] for box in charboxes), min(box[2] for box in charboxes))
                    max_x = max(max(box[0] for box in charboxes), max(box[2] for box in charboxes))

                    page_box.append((min_x, page_height - min_y, max_x, page_height - max_y)) #개행 좌표
                    after_last_box_end = curr_idx[0]+curr_idx[1]+1

                except:
                    try:
                        charboxes = [page.get_charbox(k) for k in range(curr_idx[0], curr_idx[0] + curr_idx[1] - 1)]
                        min_y = min(min(box[1] for box in charboxes), min(box[3] for box in charboxes))
                        max_y = max(max(box[1] for box in charboxes), max(box[3] for box in charboxes))
                        min_x = min(min(box[0] for box in charboxes), min(box[2] for box in charboxes))
                        max_x = max(max(box[0] for box in charboxes), max(box[2] for box in charboxes))

                        page_box.append((min_x, page_height - min_y, max_x, page_height - max_y)) #개행 좌표
                        after_last_box_end = curr_idx[0]+curr_idx[1]+1

                    except:
                        # 이전 loop의 "idx[0]+idx[1] + 1"과 다음 loop의 "idx[0] - 1"을 각각 "idx[0]"과 "idx[0] + [1]"로 간주하여 page_box에 append합니다.
                        try:
                            next_j = page_lines[idx + 1] if idx + 1 < len(page_lines) else page_lines[len(page_lines)-1]
                            next_idx = page.search(next_j, match_whole_word=True, match_case=False, consecutive=False).get_next() #개행 좌표 찾기

                            charboxes = [page.get_charbox(k) for k in range(after_last_box_end + 1, next_idx[0] - 1)]
                            min_y = min(min(box[1] for box in charboxes), min(box[3] for box in charboxes))
                            max_y = max(max(box[1] for box in charboxes), max(box[3] for box in charboxes))
                            min_x = min(min(box[0] for box in charboxes), min(box[2] for box in charboxes))
                            max_x = max(max(box[0] for box in charboxes), max(box[2] for box in charboxes))

                            page_box.append((min_x, page_height - min_y, max_x, page_height - max_y)) #개행 좌표

                        except:
                            try:
                                charboxes = [page.get_charbox(k) for k in range(after_last_box_end + 1, next_idx[0] - 2)]
                                min_y = min(min(box[1] for box in charboxes), min(box[3] for box in charboxes))
                                max_y = max(max(box[1] for box in charboxes), max(box[3] for box in charboxes))
                                min_x = min(min(box[0] for box in charboxes), min(box[2] for box in charboxes))
                                max_x = max(max(box[0] for box in charboxes), max(box[2] for box in charboxes))

                                page_box.append((min_x, page_height - min_y, max_x, page_height - max_y)) #개행 좌표


                            except:
                                page_box.append((0, 0, 0, 0)) #개행 좌표

            box.append(page_box)
        else:
            lines.append([])
            box.append([])
            txt.append("")
            
    # txt = ' '.join(txt)
    # txt_clean = re.sub('\r\n', '', txt)
    txtpercent = count/len(pdf)
    return {
        "txt": txt,
        "box": box,
        "lines": lines,
        "size": size,
        "txtpercent": txtpercent
    } #텍스트, 박스좌표, 텍스트 비율


def pdf2ocr(path):
    txt = []
    box = []
    word = []
    pdf = pdfium.PdfDocument(path)
    ocr = PaddleOCR(lang="korean", use_gpu=True)
    for i in pdf:
        page_box =[]
        page_txt = []
        pg = i.render().to_numpy()
        result = ocr.ocr(pg, cls=False)
        try:
            for j in result[0]:
                page_txt.append(j[-1][0])
                page_box.append(j[0][0]+j[0][2])
            word.append(page_txt)
            box.append(page_box)
            txt.append(' '.join(page_txt))
        except: #페이지 없는 경우 제외
            word.append([])
            box.append([])
            txt.append("")
            pass
    # txt = ' '.join(txt)
    return {
        "txt": txt,
        "box": box,
        "word": word
    }


def extract_json(txt, size, word, box):
    json = []
    for i in range(len(word)):#페이지
        page_word = []
        for j in range(len(word[i])): #페이지 별 단어
            words = {
                "text": word[i][j],
                "boundingBox":{
                    "vertices": [
                        {"x":box[i][j][0], "y":box[i][j][1]}, #x0, y0
                        {"x":box[i][j][2], "y":box[i][j][1]}, #x1, y0
                        {"x":box[i][j][2], "y":box[i][j][3]}, #x1, y1
                        {"x":box[i][j][0], "y":box[i][j][3]}, #x0, y1
                    ]
                }
            }
            page_word.append(words)
        
        page_json = {
            "width":size[i][0], #size x
            "height":size[i][1], #size y
            "text":txt[i],
            "words":page_word
        }
        json.append(page_json)

    return json
