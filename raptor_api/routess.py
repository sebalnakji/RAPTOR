# asyncio 비동기

import logging
import aiohttp
import asyncio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from .models import InputData, OutputData, CallBack, CallBackResult
from .utils_raptor import raptor

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

raptor_router = APIRouter()

description_raptor = """
## RAPTOR - 실행요청\n
1. databaseName: Milvus 데이터베이스명(워크스페이스 관리ID)\n
2. summarizeModel: 요약 모델명\n
3. embeddingModel: 임베딩 모델명\n
4. targetVectorTables: RAPTOR 대상 벡터테이블 목록\n
    A. vectorTableName: 벡터 테이블명(컬렉션명)\n
    B. vectorColumnName: 벡터 컬럼명\n 
    C. textColumnName: 벡터화 되기전 텍스트컬럼명\n
    D. vectorTableIdx: 벡터테이블 일련번호(datalake DB 일련번호)\n
5. newVectorTableName: RAPTOR 처리 후 결과 벡터 테이블명\n
6. workerId: 요약 모델 할당 ID
"""

CALLBACK_BASE_URL = "http://sbtglobal2.iptime.org:17301/datalake/v1/data/raptor/callback"


async def send_callback(result: CallBack, raptorIdx: int):
    callback_url = f"{CALLBACK_BASE_URL}/{raptorIdx}"
    async with aiohttp.ClientSession() as session:
        async with session.post(callback_url, json=result.model_dump()) as response:
            if response.status == 200:
                logger.info(f"콜백 전송 성공: raptorIdx={raptorIdx}")
            else:
                logger.error(f"콜백 전송 실패: {response.status}, raptorIdx={raptorIdx}")


async def run_raptor(data: InputData, raptorIdx: int):
    try:
        raptor_result = await raptor(data)
        logger.info(f"RAPTOR 완료: newVectorTableName={data.newVectorTableName}")
        
        callback_data = CallBack(
            status={
                "code": 200,
                "message": "OK"
            },
            result=CallBackResult(
                databaseName=data.databaseName,
                newVectorTableName=data.newVectorTableName,
                successful_tables=raptor_result["successfulTables"],
                failed_tables=raptor_result["failedTables"]
            )
        )
        logger.info(f"{callback_data}")
        await send_callback(callback_data, raptorIdx)
    except Exception as e:
        logger.error(f"RAPTOR 실행 중 오류 발생: {str(e)}")
        
        error_data = CallBack(
            status={
                "code": 500,
                "message": str(e)
            },
            result=CallBackResult(
                databaseName=data.databaseName,
                newVectorTableName=data.newVectorTableName,
                successful_tables=[],
                failed_tables=[table.vectorTableName for table in data.targetVectorTables]
            )
        )
        logger.info(f"{error_data}")
        await send_callback(error_data, raptorIdx)


@raptor_router.post("/run/{raptorIdx}", response_model=OutputData, description=description_raptor)
async def raptor_route(data: InputData, raptorIdx: int, request: Request): 
    
    logger.info(f"""
                RAPTOR 요청 받음: 
                databaseName={data.databaseName}, 
                summarizeModel={data.summarizeModel}, 
                embeddingModel={data.embeddingModel},
                newVectorTableName={data.newVectorTableName},
                raptorIdx={raptorIdx},
                number of table: {len(data.targetVectorTables)},
                worker_id: {data.workerId}
                """)
    try:
        
        asyncio.create_task(run_raptor(data, raptorIdx))
        
        logger.info(f"RAPTOR 요청 완료: newVectorTableName={data.newVectorTableName}, raptorIdx={raptorIdx}")
        return JSONResponse(
            status_code=200,
            content={
                "status": {
                    "code": 200,
                    "message": "OK"
                },
                "result": {
                    "status": "RAPTOR 실행 요청이 접수되었습니다."
                }
            }
        )
        
    except Exception as e:
        logger.error(f"RAPTOR 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))