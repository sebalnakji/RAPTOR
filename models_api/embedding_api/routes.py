from logging_config import models_embedding_logger
from fastapi import APIRouter, HTTPException
from .models import InputData, OutputData
from .utils import BGE, MiniLM, KRsBert


embedding_router = APIRouter()

description_bge = """
[Hyper Parameter]\n
1. batch size: 한 번에 모델이 학습하는 문장의 개수\n
2. max_length: 처리 가능한 최대 토큰 수 (1~8192)\n\n
[API]\n
1. batch size: 12 (default)\n
2. max_length: 1000"""
                        
description_minilm = """
max_length: 128\n
segmentation model의 max_length 조절 필요"""

description_krsbert = """
max_length: 128\n
segmentation model의 max_length 조절 필요"""


# FastAPI 엔드포인트 정의
@embedding_router.post("/bge", response_model=OutputData, description=description_bge)
async def get_bge_embedding(data: InputData):
    models_embedding_logger.info(f"BGE 임베딩 요청 받음")
    try:
        result = BGE(data)
        models_embedding_logger.info(f"BGE 임베딩 완료")
        return {
            "status": {
                "code": 20000,
                "message": "OK"
            },
            "result": result
        }

    except Exception as e:
        models_embedding_logger.error(f"BGE 임베딩 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))    

@embedding_router.post("/minilm", response_model=OutputData, description=description_minilm)
async def get_minilm_embedding(data: InputData):
    models_embedding_logger.info(f"MiniLM 임베딩 요청 받음")
    try:
        result = MiniLM(data)
        models_embedding_logger.info(f"MiniLM 임베딩 완료")
        return {
            "status": {
                "code": 20000,
                "message": "OK"
            },
            "result": result
        }

    except Exception as e:
        models_embedding_logger.error(f"MiniLM 임베딩 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))   

@embedding_router.post("/krsbert", response_model=OutputData, description=description_krsbert)
async def get_krsbert_embedding(data: InputData):
    models_embedding_logger.info(f"KRsBERT 임베딩 요청 받음")
    try:
        result = KRsBert(data)
        models_embedding_logger.info(f"KRsBERT 임베딩 완료")
        return {
            "status": {
                "code": 20000,
                "message": "OK"
            },
            "result": result
        }

    except Exception as e:
        models_embedding_logger.error(f"KRsBERT 임베딩 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))     
