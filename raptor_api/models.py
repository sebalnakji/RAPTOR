from pydantic import BaseModel
from typing import List


class VectorTableInfo(BaseModel):
    vectorTableName: str
    vectorColumnName: str
    textColumnName: str
    vectorTableIdx: int
    

class InputData(BaseModel):
    databaseName: str
    summarizeModel: str
    embeddingModel: str
    targetVectorTables: List[VectorTableInfo]
    newVectorTableName: str
    workerId: int
    
    
class OutputData(BaseModel):
    status: dict
    result: dict
    
    
class ResultCallBackResult(BaseModel):
    databaseName: str
    newVectorTableName: str
    successfulTables: List[str]
    failedTables: List[str]


class ResultCallBack(BaseModel):
    status: dict
    result: ResultCallBackResult


class StatusCallBackResult(BaseModel):
    vectorTableName: str
    completeWork: bool


class StatusCallBack(BaseModel):
    status: dict
    result: StatusCallBackResult
