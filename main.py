import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from models_api.embedding_api.routes import embedding_router
from models_api.summarize_api.routes import summarize_router
from models_api.segmentation_api.routes import segmentation_router

from converter_api.ppt_to_pdf_api.routes import ppt_to_pdf_router
from converter_api.doc_to_pdf_api.routes import docx_to_pdf_router
from converter_api.img_to_pdf_api.routes import image_to_pdf_router
from converter_api.excel_to_csv_api.routes import excel_to_csv_router
from converter_api.excel_to_pdf_api.routes import excel_to_pdf_router
# from converter_api.hwp_to_html_to_pdf_api.routes import hwp_to_xhtml_router, xhtml_to_pdf_router

from data_extraction_api.image_ocr_api.routes import image_ocr_router
from data_extraction_api.hwp_to_text_api.routes import hwp_to_text_router
from data_extraction_api.pdf_to_text_api.routes import pdf_to_text_router


load_dotenv()


SERVER_ENDPOINT = os.getenv('SERVER_ENDPOINT')


class TimeoutMiddleware:
    def __init__(self, app, timeout=600):
        self.app = app
        self.timeout = timeout

    async def __call__(self, scope, receive, send):
        request = Request(scope, receive=receive)
        try:
            await asyncio.wait_for(self.app(scope, receive, send), timeout=self.timeout)
        except asyncio.TimeoutError:
            response = JSONResponse(
                status_code=504,
                content={"message": "요청 시간이 초과되었습니다. 나중에 다시 시도해 주세요."},
            )
            await response(scope, receive, send)


app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[SERVER_ENDPOINT],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 타임아웃 미들웨어 설정
app.add_middleware(TimeoutMiddleware, timeout=1200)

# 라우터 등록
app.include_router(embedding_router, prefix="/data-preprocessing/v1/embedding", tags=["Embedding API"])
app.include_router(segmentation_router, prefix="/data-preprocessing/v1/segmentation", tags=["Segmentation API"])
app.include_router(summarize_router, prefix="/data-preprocessing/v1/summarize", tags=["Summarize API"])
app.include_router(docx_to_pdf_router, prefix="/data-preprocess/v1/convert", tags=["Converter API"])
app.include_router(excel_to_pdf_router, prefix="/data-preprocess/v1/convert", tags=["Converter API"])
# app.include_router(hwp_to_xhtml_router, prefix="/data-preprocess/v1/convert", tags=["Converter API"])
# app.include_router(xhtml_to_pdf_router, prefix="/data-preprocess/v1/convert", tags=["Converter API"])
app.include_router(image_to_pdf_router, prefix="/data-preprocess/v1/convert", tags=["Converter API"])
app.include_router(ppt_to_pdf_router, prefix="/data-preprocess/v1/convert", tags=["Converter API"])
app.include_router(excel_to_csv_router, prefix="/data-preprocess/v1/convert", tags=["Converter API"])
app.include_router(image_ocr_router, prefix="/data-preprocess/v1/convert", tags=["Data Extraction API"])
app.include_router(pdf_to_text_router, prefix="/data-preprocess/v1/convert", tags=["Data Extraction API"])
app.include_router(hwp_to_text_router, prefix="/data-preprocess/v1/convert", tags=["Data Extraction API"])
