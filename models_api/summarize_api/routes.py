from logging_config import models_summarize_logger
from fastapi import APIRouter, HTTPException
from .models import RequestModel_llama3, RequestModel_t5, OutputData
from .utils_llama3 import summarize_llama3
from .utils_t5 import summarize_t5


summarize_router = APIRouter()

description_llama3 = """
1. segCount: 요약할 문장 목록을 분리할 문단 수 (-1일 경우 최적 문단 수로 분리, default=-1)\n
2. segMaxSize: 한 문단에 포함될 최대 문자열의 글자 수 (1~3000, default=100)\n
3. segMinSize: 한 문단에 포함될 최소 문자열의 글자 수 (0~segMaxSize)\n
4. text: 요약할 텍스트"""
                        
description_t5 = """
1. segMaxSize: 한 문단에 포함될 최대 문자열의 글자 수 (1~3000, default=100)\n
2. segMinSize: 한 문단에 포함될 최소 문자열의 글자 수 (0~segMaxSize)\n
3. text: 요약할 텍스트"""

@summarize_router.post("/llama3", response_model=OutputData, description=description_llama3)
async def summarize_route_llama3(data: RequestModel_llama3):
    models_summarize_logger.info(f"LLAMA3 요약 요청 받음: segCount={data.segCount}, segMaxSize={data.segMaxSize}, segMinSize={data.segMinSize}")
    try:
        result = summarize_llama3(data)
        models_summarize_logger.info(f"LLAMA3 요약 완료")
        return {
            "status": {
                "code": 20000,
                "message": "OK"
            },
            "result": result
        }
    except Exception as e:
        models_summarize_logger.error(f"LLAMA3 요약 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@summarize_router.post("/t5", response_model=OutputData,description=description_t5)
async def summarize_route_t5(data: RequestModel_t5):
    models_summarize_logger.info(f"T5 요약 요청 받음: segMaxSize={data.segMaxSize}, segMinSize={data.segMinSize}")
    try:
        result = summarize_t5(data)
        models_summarize_logger.info(f"T5 요약 완료")
        return {
            "status": {
                "code": 20000,
                "message": "OK"
            },
            "result": result
        }
    except Exception as e:
        models_summarize_logger.error(f"T5 요약 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
