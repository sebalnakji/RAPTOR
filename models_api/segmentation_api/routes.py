from logging_config import models_segmentation_logger
from fastapi import APIRouter, HTTPException
from .models import InputData_llama3, OutputData, InputData_nltk
from .utils_llama3 import segment_llama
from .utils_nltk import segment_nltk


segmentation_router = APIRouter()

description_llama3 = """
1. segCount: 분할할 문장 목록을 분리할 문단 수 (-1일 경우 최적 문단 수로 분리, default=-1)\n
2. segMaxSize: 한 문단에 포함될 최대 문자열의 글자 수 (1~3000, default=100)\n
3. segMinSize: 한 문단에 포함될 최소 문자열의 글자 수 (0~segMaxSize)\n
4. text: 분할할 텍스트"""
                        
description_nltk = """
1. chunk_length: 한 문단에 포함될 최대 토큰 수 (100~1000, default=100)\n
2. overlap_length: 문단 별 겹치는 토큰 수 (0 ~ (chunk_length)-1, default=28) \n
3. whether_fix_sentence: 완전한 문장으로 분할할 것인지 여부(True/False)\n
4. text: 분할할 텍스트"""

@segmentation_router.post("/llama3", response_model=OutputData, description=description_llama3)
async def segment_text_route_llama3(data: InputData_llama3):
    models_segmentation_logger.info(f"LLAMA3 세그멘테이션 요청 받음: segCount={data.segCount}, segMaxSize={data.segMaxSize}, segMinSize={data.segMinSize}")
    try:
        result = segment_llama(data)
        models_segmentation_logger.info(f"LLAMA3 세그멘테이션 완료")
        return {
            "status": {
                "code": 20000,
                "message": "OK"
            },
            "result": result
        }
    except Exception as e:
        models_segmentation_logger.error(f"LLAMA3 세그멘테이션 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@segmentation_router.post("/nltk", response_model=OutputData, description=description_nltk)
async def segment_text_route_nltk(data: InputData_nltk):
    models_segmentation_logger.info(f"NLTK 세그멘테이션 요청 받음: chunk_length={data.chunk_length}, whether_fix_sentence={data.whether_fix_sentence}")
    try:
        result = segment_nltk(data)
        models_segmentation_logger.info(f"NLTK 세그멘테이션 완료")
        return {
            "status": {
                "code": 20000,
                "message": "OK"
            },
            "result": result
        }
    except Exception as e:
        models_segmentation_logger.error(f"NLTK 세그멘테이션 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
