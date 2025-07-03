import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from raptor_api.routes import raptor_router


load_dotenv()


SERVER_ENDPOINT = os.getenv('SERVER_ENDPOINT')


class TimeoutMiddleware:
    def __init__(self, app, timeout=2592000):
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


raptor = FastAPI()

# CORS 설정
raptor.add_middleware(
    CORSMiddleware,
    allow_origins=[SERVER_ENDPOINT],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 타임아웃 미들웨어 설정
raptor.add_middleware(TimeoutMiddleware, timeout=2592000)

# 라우터 등록
raptor.include_router(raptor_router, prefix="/data-preprocessing/v1/raptor", tags=["RAPTOR API"])