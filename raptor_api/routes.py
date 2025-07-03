# background task 비동기
import os
import logging
import aiohttp
import asyncio
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from .models import InputData, OutputData, ResultCallBack, ResultCallBackResult, StatusCallBack, StatusCallBackResult
from .utils_raptor import raptor


load_dotenv()

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


RESULT_CALLBACK_BASE_URL = os.getenv('RESULT_CALLBACK_BASE_URL')
STATUS_CALLBACK_BASE_URL = os.getenv('STATUS_CALLBACK_BASE_URL')


async def send_result_callback(result: ResultCallBack, raptorIdx: int):
    callback_url = f"{RESULT_CALLBACK_BASE_URL}/{raptorIdx}"
    timeout = aiohttp.ClientTimeout(total=30)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(callback_url, json=result.model_dump()) as response:
                if response.status == 200:
                    logger.info(f"결과 콜백 전송 성공: raptorIdx={raptorIdx}, result={result}")
                else:
                    logger.error(f"결과 콜백 전송 실패: {response.status}, raptorIdx={raptorIdx}, result={result}")
                    
    except asyncio.TimeoutError:
        logger.error(f"결과 콜백 전송 타임아웃: raptorIdx={raptorIdx}")
        
    except Exception as e:
        logger.error(f"결과 콜백 전송 중 오류 발생: {str(e)}, raptorIdx={raptorIdx}")


async def send_status_callback(result: StatusCallBack, raptorIdx: int):
    callback_url = f"{STATUS_CALLBACK_BASE_URL}/{raptorIdx}"
    timeout = aiohttp.ClientTimeout(total=30)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(callback_url, json=result.model_dump()) as response:
                if response.status == 200:
                    logger.info(f"상태 콜백 전송 성공: raptorIdx={raptorIdx}, result={result}")
                else:
                    logger.error(f"상태 콜백 전송 실패: {response.status}, raptorIdx={raptorIdx}, result={result}")
                    
    except asyncio.TimeoutError:
        logger.error(f"상태 콜백 전송 타임아웃: raptorIdx={raptorIdx}")
        
    except Exception as e:
        logger.error(f"상태 콜백 전송 중 오류 발생: {str(e)}, raptorIdx={raptorIdx}")


async def run_raptor(data: InputData, raptorIdx: int):
    async def callback(vectorTableName: str, completeWork: bool):
        status_callback = StatusCallBack(
            status={"code": 200, "message": "OK"},
            result=StatusCallBackResult(
                vectorTableName=vectorTableName,
                completeWork=completeWork
            )
        )
        logger.info(f"{status_callback}")
        await send_status_callback(status_callback, raptorIdx)
        
    try:
        raptor_result = await raptor(data, callback)
        logger.info(f"RAPTOR 완료: newVectorTableName={data.newVectorTableName}")
        
        result_callback = ResultCallBack(
            status={
                "code": 200,
                "message": "OK"
            },
            result=ResultCallBackResult(
                databaseName=data.databaseName,
                newVectorTableName=data.newVectorTableName,
                successfulTables=raptor_result["successfulTables"],
                failedTables=raptor_result["failedTables"]
            )
        )
        logger.info(f"{result_callback}")
        await send_result_callback(result_callback, raptorIdx)
        
    except Exception as e:
        logger.error(f"RAPTOR 실행 중 오류 발생: {str(e)}")
        
        error_status_callback = StatusCallBack(
            status={"code": 500, "message": str(e)},
            result=StatusCallBackResult(
                vectorTableName=data.targetVectorTables[-1].vectorTableName if data.targetVectorTables else "",
                completeWork=False
            )
        )
        logger.info(f"{error_status_callback}")
        await send_status_callback(error_status_callback, raptorIdx)
        
        # 오류가 발생해도 raptor_result를 반환할 수 있도록 설정
        raptor_result = raptor_result if 'raptor_result' in locals() else {"successfulTables": [], "failedTables": []}
        
        error_result_callback = ResultCallBack(
            status={
                "code": 500,
                "message": str(e)
            },
            result=ResultCallBackResult(
                databaseName=data.databaseName,
                newVectorTableName=data.newVectorTableName,
                successfulTables=raptor_result["successfulTables"],
                failedTables=raptor_result["failedTables"]
            )
        )
        logger.info(f"{error_result_callback}")
        await send_result_callback(error_result_callback, raptorIdx)


@raptor_router.post("/run/{raptorIdx}", response_model=OutputData, description=description_raptor)
async def raptor_route(data: InputData, raptorIdx: int, background_tasks: BackgroundTasks):
    
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
        
        background_tasks.add_task(run_raptor, data, raptorIdx)
        
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